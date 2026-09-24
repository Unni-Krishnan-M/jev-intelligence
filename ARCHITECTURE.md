# JEV Architecture

JEV is a domain-independent decision and early-warning engine (`ml/jev_ml/core`) with domain adapters. The movie recommender is the first adapter. The platform contract is in [docs/platform.md](docs/platform.md).

```
                 ┌──────────────────────────── browser ─────────────────────────────┐
                 │  Next.js 16 (App Router, TS, Tailwind v4, shadcn/ui, Recharts)    │
                 └──────────────┬────────────────────────────────────────────────────┘
                                │ same-origin /api/*  (httpOnly JWT cookie + CSRF header)
                 ┌──────────────▼──────────────┐
                 │ Next.js server (rewrites)   │  proxy.ts: optimistic route guard
                 └──────────────┬──────────────┘
                                │ HTTP
┌───────────────────────────────▼──────────────────────────────────────────────────────┐
│ FastAPI  (backend/jev_api)                                                           │
│  middleware: request-id · JSON logs (secret redaction) · rate limit · sec headers    │
│  routers: auth · users · me · movies · recommendations · models/experiments ·        │
│           health · intel (operator console) · admin · events · governance ·          │
│           experiments/online · security (service tokens)                             │
│  services: profile · recommend (strategy decision → experiment arm → serve/persist)  │
│            events (append-only log + projections) · intel (gather → run → persist)   │
│            governance (snapshot → train → gate → promote/rollback) · experiments     │
│  EngineHolder ──► RecommendationEngine (jev_ml.engine)  ◄── models/<version>/        │
│  jev_ml.core.run_domain(adapter) ◄── data files + event log cut at as_of + manifest   │
└───────┬──────────────────────────────┬──────────────────────────────┬───────────────┘
        │ SQLAlchemy 2 + Alembic       │ redis-py                     │ files (npz/json)
┌───────▼────────┐             ┌───────▼───────┐             ┌────────▼────────────────┐
│ PostgreSQL 17  │             │ Redis 7        │             │ models/ experiments/     │
│ (SQLite in dev)│             │ (memory in dev)│             │ data/raw data/processed  │
└────────────────┘             └───────────────┘             └────────▲────────────────┘
                                                                      │ writes
                              ┌───────────────────────────────────────┴───────────────┐
                              │ Offline ML pipeline (jev_ml, scripts/)                  │
                              │ download → validate → clean → features → split → tune   │
                              │ → evaluate → train → save artifacts → register version  │
                              └─────────────────────────────────────────────────────────┘
```

## Principles

1. **Training and inference are separate.** `jev_ml.training` writes versioned artifacts. `jev_ml.engine`
   only reads them. The API never trains. The two share just the model classes and `signals.py` (how a
   rating, favourite or watch becomes a weight), so there is one definition of the business logic.
2. **Every app user is an "unseen" user.** Profiles are built per request from the user's database events.
   Item-kNN and content work from those events directly, and ALS folds the user in against the frozen item
   factors. There is no nightly per-user job and nothing goes stale: a new rating changes the next list.
3. **Portable artifacts.** Everything is `.npz` or `.json`, with no pickles. The content featurizer stores its
   vocabulary and IDF, so it can vectorise brand-new movies at inference time (item cold start).
4. **Runs on any laptop.** The dataset is MovieLens *small* (100k ratings) and everything runs on CPU with numpy,
   scipy and scikit-learn. A full tuned pipeline takes about 4 minutes, a `--quick` run about 21 s (peak RSS about 620 MB), and the
   serving model loads in about 0.1s with about 25ms per recommendation request. Without Docker the API uses
   SQLite plus an in-process cache.
5. **Files are the source of truth for models.** The API mirrors `models/*/manifest.json` and `experiments/*/metrics.json`
   into `ModelVersion` / `Experiment` / `EvaluationMetric` on startup and when the admin pages load.

## Request path: `GET /recommendations`

1. Auth dependency resolves the user from the bearer token or the cookie. Cookie writes require `X-JEV-CSRF`.
2. The cache key is `rec:{user}:{profile_version}:{model_version}:{diversity}:{context}:{limit}:{offset}:{filters}`.
   `profile_version` is bumped on every taste event, so a key is never invalidated, only superseded.
3. On a miss: load ratings, favourites, watches, negative feedback and genre picks, then build a `UserProfile`
   and run `HybridRanker.rank` (see [docs/recommendation-algorithms.md](docs/recommendation-algorithms.md)).
4. Persist every served item (`recommendations` table: rank, score, reason, reason code, per-signal breakdown,
   request id, model version, context), then cache the response and return it.

## Request path: `POST /intel/runs`

1. Admin guard + CSRF. A lock allows one run at a time (409 otherwise); `as_of` is validated against the data range.
2. `services/intel.py` gathers `PipelineInputs`: processed MovieLens data, app ratings / feedback / served
   recommendations from the DB, the active model manifest and its experiment, and the keys of warnings dismissed
   inside the suppression window.
3. `jev_ml.core.run_domain(adapter, as_of, now)` runs every stage using only data ≤ `as_of` (the event log is cut on
   event time and knowledge time, docs/STREAMING_ARCHITECTURE.md §4). It is pure and deterministic for a fixed code
   version, data files and active model, with no DB or web imports. The run records `pipeline_version`,
   `config_hash`, `input_fingerprint`, `model_version` and the event watermark.
4. The service persists the run (`intel_runs`, full result JSON, normalised objects and evidence). A run without
   `as_of` is `mode = live`: it upserts warnings by dedup key (an event row per status change) and auto-resolves stale
   ones. A run with `as_of` is `mode = replay`: its warnings stay in its result and live state is untouched.
5. List endpoints read the latest successful **live** run (or `?run_id=`, `?mode=replay|any`). Lifecycle tables
   (warnings, decisions, feedback, scenarios) are queried directly.

A run on the real data takes about 0.9 s including persistence. The API starts one in a background thread at
startup when the latest live run is older than `JEV_INTEL_MIN_INTERVAL_HOURS`, and a debounced refresher starts one
after fresh events.

## Write path: `POST /events` and the member write endpoints

Every member interaction (rating, favourite, watch, feedback) and every pushed observation of a generic domain is a
row in the append-only `events` table, with `event_time` (when it happened) and `ingested_at` (when JEV knew). The
current-state tables (`ratings`, `favorites`, …) are projections folded from the log in the same transaction, so there
is no dual write. An `Idempotency-Key` header (or per-item keys in a batch) makes retries exactly-once:
`idempotency_keys` stores the first response. `POST /events/replay` recomputes the projections from the log and
reports drift. Detail: [docs/STREAMING_ARCHITECTURE.md](docs/STREAMING_ARCHITECTURE.md).

## Model lifecycle: governance

`POST /governance/retrain` (or `scripts/retrain.py`, or the opt-in scheduler driven by the `retrain_model` decision)
takes one job lock and runs: **snapshot** (MovieLens + app feedback as pseudo-ratings, content-hashed) → **train** a
candidate (never activated) → **gate** (both training recipes refit on the candidate snapshot's frozen split; paired
bootstrap non-inferiority on NDCG@10, Recall@10 and cold start, with power-derived margins; coverage, calibration,
latency and an artifact self-check). Promotion requires a gate that passed against the model active *now* (or `force`
with a reason); rollback restores the previous active version. `EngineHolder` swaps engines under a lock and other
workers follow `registry.json` by polling. Every step is audited and every version carries lineage (snapshot, config
hash, seed, git commit, job). Detail: [docs/RETRAINING_AND_MODEL_GOVERNANCE.md](docs/RETRAINING_AND_MODEL_GOVERNANCE.md).

## Serving path with experiments

`GET /recommendations`: the member's `recommendation_strategy` decision is computed (and persisted in
`member_decisions`); if a running experiment covers the surface, the member is assigned by a salted hash (sticky in
`ab_assignments`) and the arm's config is applied; the list is served and every exposure, cache hits included, is
logged. Outcomes are attributed within a window; the results endpoint reports per-arm metrics, SRM, Bonferroni-adjusted
tests, guardrails and power. Detail: [docs/EXPERIMENTATION.md](docs/EXPERIMENTATION.md).

## Security boundaries

Browser → web (`/api/*` rewrite, per-request nonce CSP, signs the client address into `x-jev-client` with
`JEV_PROXY_SECRET`) → API (not published in compose). Sessions are JWTs with `jti` and a per-user `token_version`, so
logout, logout-all and password change revoke them. Machines use hashed, scoped, expiring service tokens that can
only post observations. Admin checks re-read `is_admin` on every request; a route walker test fails on any
`/intel*` or `/admin*` route without an admin guard. Detail: [docs/SECURITY_AUDIT_PHASE2.md](docs/SECURITY_AUDIT_PHASE2.md).

## Code map

| Path | Role |
|---|---|
| `ml/jev_ml/data/` | download (checksum-verified), Wikidata enrichment, validation + preprocessing, loaders |
| `ml/jev_ml/features.py` | per-field TF-IDF featurizer |
| `ml/jev_ml/models/` | `popularity`, `content`, `itemknn`, `als`, `hybrid` (ranking pipeline) |
| `ml/jev_ml/explain.py` | reasons from model contributions |
| `ml/jev_ml/evaluation/` | split strategies, metrics, evaluator, report/plots |
| `ml/jev_ml/training.py` | experiment pipeline, tuning, artifact writing |
| `ml/jev_ml/registry.py` | model registry (active pointer) |
| `ml/jev_ml/engine.py` | inference engine |
| `ml/jev_ml/core/` | JEV core, domain-independent: types, adapter protocol, series builder, trends, anomalies, forecasts, scenarios, risk, decision framework, early-warning decision, warnings, actions, signals, drift tests, `run_domain` |
| `ml/jev_ml/domains/movie/` | movie adapter: ingest/validation, genre series, raters, lapse, model governance, user intelligence (preference drift, strategy decision, preference scenarios) |
| `ml/jev_ml/domains/generic/` | generic adapter: any long CSV + `configs/domains/*.yaml` |
| `ml/jev_ml/intel/` | compatibility layer: `run_pipeline(PipelineInputs)` and re-exports for the movie domain, plus offline evaluation |
| `ml/jev_ml/governance/` | snapshots, candidate training, the promotion gate (`gates.py`), promotion blockers |
| `ml/jev_ml/evaluation/{benchmark,leakage,stats,cold_start}.py` | two-protocol benchmark, leakage controls, bootstrap and permutation statistics |
| `backend/jev_api/services/{events,governance,experiments,experiment_stats}.py` | event log and projections; retrain jobs, lock, promotion; online experiments and their statistics |
| `backend/jev_api/` | FastAPI app, ORM models, Alembic migrations, routers, services |
| `frontend/src/` | Next.js pages (`app/`), components (`components/jev`, `components/ui`), API client (`lib/`) |
| `scripts/` | pipeline entry points, acceptance test, screenshot capture |
| `tests/` | unit / ml / integration (pytest), browser journey via `scripts/capture_screenshots.py` |

## Data model

`users` 1─* `ratings`, `watch_history`, `favorites`, `user_genre_preferences`, `recommendations`, `recommendation_feedback`.
`movies` *─* `genres` via `movie_genres`. `recommendation_feedback.recommendation_id` → `recommendations` (SET NULL).
`intel_runs` 1─* `intel_decisions`; `intel_warnings` 1─* `intel_warning_events` (audit trail), with at most one
open warning per key (partial unique index); `intel_feedback` references decisions, warnings, actions or forecasts by id;
`intel_scenarios` stores saved what-if analyses.
`experiments` *─1 `model_versions`, and `evaluation_metrics` *─1 `experiments` (unique per model/protocol/metric/K).
Phase 2 (migrations 0006–0010): `intel_runs.mode` (live | replay) and `event_watermark`; `events`,
`idempotency_keys`, `event_daily_counts`; `dataset_snapshots`, `training_jobs`, `model_governance`, `governance_locks`;
`ab_experiments`, `ab_variants`, `ab_assignments`, `ab_exposures`, `ab_outcomes`, `member_decisions`;
`service_tokens`, `revoked_tokens` and `users.token_version`.
The schema uses only portable types, with JSON in place of JSONB, so it runs on PostgreSQL and SQLite. Constraints include
the rating range (0.5–5), the allowed feedback kinds, and uniqueness of (user, movie) for ratings and favourites.
Indexes cover every per-user timeline query.

## Frontend design system

The frontend is styled as a festival programme printed for a dark screening room. It uses warm near-black and paper
tones (with a light "paper" theme), a single tungsten-amber accent, Instrument Serif for display text, IBM Plex Sans for
UI and IBM Plex Mono for data, hairline rules instead of shadows, and frosted glass only on the top bar. No posters are
used: every cover is typeset from real metadata (genre palette, year, director) in one of three deterministic layouts.
That is cheap to render on any GPU and avoids image licensing. The six signals keep the same colour everywhere, taken
from a CVD-validated palette, and every signal chart also has a legend or value table.
