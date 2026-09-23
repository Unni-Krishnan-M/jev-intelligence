# JEV Architecture

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
│  routers: auth · users · movies · recommendations · models/experiments · health      │
│  services: profile (DB → UserProfile) · recommend (serve/persist/cache) · sync       │
│  EngineHolder ──► RecommendationEngine (jev_ml.engine)  ◄── models/<version>/        │
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
   scipy and scikit-learn. A full tuned pipeline takes about 4 minutes, a `--quick` run under 20s, and the
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
| `backend/jev_api/` | FastAPI app, ORM models, Alembic migrations, routers, services |
| `frontend/src/` | Next.js pages (`app/`), components (`components/jev`, `components/ui`), API client (`lib/`) |
| `scripts/` | pipeline entry points, acceptance test, screenshot capture |
| `tests/` | unit / ml / integration (pytest), browser journey via `scripts/capture_screenshots.py` |

## Data model

`users` 1─* `ratings`, `watch_history`, `favorites`, `user_genre_preferences`, `recommendations`, `recommendation_feedback`.
`movies` *─* `genres` via `movie_genres`. `recommendation_feedback.recommendation_id` → `recommendations` (SET NULL).
`experiments` *─1 `model_versions`, and `evaluation_metrics` *─1 `experiments` (unique per model/protocol/metric/K).
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
