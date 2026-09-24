# Changelog

## 1.3.0 — 2026-09-25
Phase 2: from a single-process demo to a governed decision engine. Every headline number was re-checked by an
independent evaluation review (docs/EVALUATION_AND_VALIDITY_REVIEW.md), and the docs now state what the evidence
does and does not show.
### Added
- **Append-only event log** (`events`, migration 0007) with bitemporal reads (`event_time`, `ingested_at`),
  `Idempotency-Key` handling, projections folded in the same transaction, `POST /events`, `POST /events/replay`,
  `GET /events/health`, pushed observations for generic domains and a debounced live-run refresher.
- **Gated retraining** (migration 0008): content-hashed snapshots that include app feedback, candidate training that
  never activates, a promotion gate (paired bootstrap non-inferiority on NDCG@10, Recall@10 and cold start; coverage,
  calibration, latency, artifact self-check), manual or opt-in automatic promotion, rollback, a retrain lock, a
  decision-driven scheduler and full lineage. `/governance/*` API and `scripts/retrain.py`.
- **Online experiments** (migration 0009): salted-hash sticky assignment, exposure logging including cache hits,
  attributed outcomes, SRM, Bonferroni-adjusted tests, guardrails and power warnings; persisted member strategy
  decisions; `/experiments/online*`; an offline replay demo labelled as such (`scripts/simulate_ab_replay.py`).
- **Replay isolation and lineage**: `intel_runs.mode` (live | replay, migration 0006); replays never touch live
  warnings; reads default to the latest live run; decision and warning lineage resolved to data sources.
- **Early-warning policy 1.1** (`ewl-1.1.0`): recovery-aware trends; change-point null with φ from history (false
  alarms 9.9 % → 1.3 % at nominal 1 %); stale-warning auto-resolution after K live runs.
- **Second reference domain**: Chicago Transit Authority daily ridership (`generic:cta-ridership`), daily and weekly
  series, seasonal forecasting with a holiday calendar, publication lag.
- **Recommender evaluation rigour**: a two-protocol benchmark (per-user and global temporal splits), leakage checks
  and the leak-free tag rebuild, bootstrap CIs, paired permutation tests with Holm adjustment, cold-start stages and
  a logistic calibrator (both shipped off by default: they failed their pre-declared adoption rules).
- **Security**: JWT revocation (logout, logout-all, password change via `token_version`), scoped hashed service tokens
  (migration 0010), a per-account login throttle, a signed client address from the web proxy, pip-audit and
  pnpm audit in CI.
- `scripts/repair_dev_db.py`: repairs a SQLite dev database stamped at head while migrations 0006–0010 were empty
  stubs (backup, rebuild at head, copy every row back; aborts if any user row would be lost).
- Warning evaluations store per-unit rows and a month-cluster bootstrap CI of the lift plus a year-stratified lift
  (`core.evaluation.lift_uncertainty`, `scripts/evaluate_domains.py`).
- Docs: INTELLIGENCE_PIPELINE, FINAL_VERIFICATION_MATRIX, FINAL_PRINCIPAL_REVIEW, PROJECT_COMPLETION_REPORT and the
  Phase 2 workstream reports.
### Changed
- **Promotion gate `gate-1.1.0`**: margins derived from a power rule (80 % pass rate for an equivalent candidate at
  the observed paired SE): NDCG@10 0.005 → 0.010, Recall@10 0.01 → 0.011, cold start 0.006 (was the NDCG margin). The
  quick mode keeps B = 2000 bootstrap resamples (was 300). Each gate records its SE, `power_if_equivalent` and
  `margin_for_80pct`. Both refits use leak-free tags when the raw tags of the same dataset are available.
- Benchmark reports star a warm comparison only when its Holm-adjusted p < 0.05 (was: unadjusted CI excludes 0);
  stored `REPORT.md` files were regenerated from their `results.json`.
- API: training-job, snapshot, governed-model and experiment datetimes always carry a UTC offset; a rollback requires
  a `reason`; a 409 from promote/activate keeps `detail` a string and lists `blockers` in their own field;
  `GET /governance/models` marks each entry `source: "job" | "artifact"`; the public `GET /health/ml` no longer echoes
  the model-load exception.
- Compose passes one `JEV_PROXY_SECRET` to web and, as `JEV_SECURITY_PROXY_SECRET`, to the API.
- A movie replay records the model version it read; `load_default_inputs(model_version=...)` can pin it.
- Docs corrected after the evaluation review: the recommender headline is qualified by protocol, the CTA warning lift
  is reported with its CI and an untouched window (no demonstrated warning skill), the drift precision is stated at its
  synthetic prevalence, calibration claims are stated relative to the base rate, and replay determinism is scoped.

## 1.2.0 — 2026-09-24
JEV becomes a domain-independent decision and early-warning engine. The movie recommender is now its first domain adapter.
### Added
- **JEV core** (`ml/jev_ml/core`), which never imports recommender code: typed records, the `DomainAdapter` protocol,
  a series builder driven by `SeriesSpec`s, and `run_domain(adapter, as_of, …)`.
- **Early-warning decision** `early_warning_level` (`NO_ACTION | MONITOR | WARNING | URGENT_ACTION`, policy
  `ewl-1.0.0`). It is monotone, has a margin confidence and abstains on skipped evidence. Warnings are raised only
  from it and carry `decision_id`.
- **Generic structured-dataset adapter**: any long CSV + YAML config. The demo domain `generic:us-unemployment`
  covers BLS rates via FRED (public domain) and is fetched with checksums by `scripts/download_domain_data.py`.
  Publication lag is modelled for replays.
- **Preference drift** (`core/drift.py`): day-level permutation tests across six aspects with Holm correction.
  - **Per-member recommendation strategy decision:** `standard | adapt_to_recent | explore`. Recommendations are
    served downstream of it.
  - **Per-member preference what-if:** real model rankings under projected preferences.
- **Evaluations**, all with real experiments:
  - per-domain forecasts (MAE, RMSE, coverage);
  - warning precision and false-positive rate from monthly leak-free replays;
  - early-warning decision flip rate;
  - drift precision and recall on labelled splices of real histories;
  - drift adaptation effect on NDCG@10.
- **Web app** reorganised around the engine: platform landing page, Intelligence-first navigation with Movies as one
  domain, a console domain switcher with capability gating, a four-level early-warning decision view, and a new
  "My intelligence" page.
- `npm run dev` starts the API and the web app together.
### Changed
- Early-warning questions and titles now read as plain language (e.g. "Should CA unemployment rate trigger an early
  warning?"). The situation label is part of the hashed decision state, so `early_warning_level` state hashes from
  pre-release 1.2 builds differ from 1.2.0 hashes for identical evidence. The policy (`ewl-1.0.0`) is unchanged.
- `jev_ml.intel.run_pipeline` is a compatibility shim over `run_domain(MovieAdapter)`. Movie outputs are unchanged
  (golden tests on synthetic and real data), apart from the additional early-warning decisions and batch.

## 1.1.0 — 2026-09-23
### Added
- Intelligence & early-warning layer (`ml/jev_ml/intel`), run leak-free as of any date:
  - validation and freshness;
  - monthly series and signals;
  - trends (Mann–Kendall, Theil–Sen, BH FDR) and change points (AR(1) null);
  - robust-z series anomalies and IsolationForest rater anomalies;
  - damped-Holt/MA/naive forecasts with finite-sample intervals, and a calibrated lapse model;
  - risk scoring;
  - typed, versioned decisions with labelled confidence kinds and abstention;
  - early warnings, an action planner and what-if scenarios.
- Offline evaluation of the layer: forecast backtests, lapse holdout with calibration, a shilling-injection study,
  series-anomaly and change-point studies, and latency (`scripts/evaluate_intelligence.py`).
- Migration `0002_intelligence`: runs, warnings (lifecycle + audit trail), decisions, scenarios and feedback.
  `/intel/*` API and `/admin/metrics`.
- Intelligence console in the web app: 13 sections plus warning, decision and series detail pages.
- GitHub Actions CI.
### Added in v1.1 (same release)
- **Score decisions.** A new decision kind `score` with a numeric `answer`, a `scale`, an `answer_interval`, and the
  confidence kind `interval` (the nominal coverage). The first real one is `editorial_slot_share`, the recommended
  % of home-rail slots per genre, from the 3-month forecast share with a conformal 80 % interval. It abstains when the
  backtested coverage is below 0.6.
- **Decision batches** (`ml/jev_ml/intel/batches.py`): `model_governance`, `genre_programming` and `audience`. Each
  answers several questions against one deep-copied, sha1-hashed state snapshot, atomically: a raise, a foreign key or
  a mutated state makes every question abstain.
- **Recommendation confidence.** A calibrated P(the user rates the film ≥ 4 among their next 5 ratings), fitted by
  isotonic regression per profile-size stratum on the validation split (`ml/jev_ml/calibration.py`,
  `scripts/calibrate_recommendations.py`, `train_models.py --calibrate`). Stored per model as `calibration.json`. It is
  null when uncalibrated, never guessed. Test ECE ≤ 0.0014.
- **Migration `0003_intel_normalized`:**
  - `intel_signals`, `intel_trends`, `intel_anomalies`, `intel_forecasts`, `intel_risks` and the polymorphic
    `intel_evidence`, all with CASCADE to their run;
  - `audit_logs` and `intel_evaluation_runs`;
  - `intel_decisions.batch_id/answer_value/answer_interval/scale`;
  - `recommendations.confidence/confidence_kind`;
  - the `interval` confidence kind is allowed.
- **Audit log** (`backend/jev_api/services/audit.py`): login success and failure, register, intelligence runs, warning
  transitions, feedback, saved scenarios and model activation. It is written in the caller's transaction and redacted,
  and it never breaks a request.
- **New endpoints:** `GET /intel/evidence`, `/intel/history/{entity}`, `/intel/recommendations` (recommender
  monitoring), `/intel/evaluation/runs`, `/intel/decisions/batches`, `/admin/audit`, and a `batch_id` filter on
  `/intel/decisions`. Recommendation items and history gain `confidence`/`confidence_kind`, `/health/ml` and
  `/models/active/summary` gain `calibration`, and `/admin/metrics` gains audit and persisted-row counters.
- **New console pages:** Evidence explorer, Recommender monitoring and Audit log. Score decisions render on their
  scale with the interval, decisions group by batch, and signals and risks show their history across runs. Members see
  confidence as "≈ n % chance you rate it 4★+".
- **Frontend unit tests:** Vitest + Testing Library in jsdom (`pnpm test`), 65 tests in 6 files.
- **Security pass:**
  - admin guards on every intel/admin route (tested over the route table);
  - CSRF on cookie-authenticated intel writes;
  - bounds on ids, strings, dates and NaN/Infinity;
  - LIKE wildcard escaping;
  - sanitised run errors and a generic 500 envelope;
  - a per-admin rate limit on manual runs;
  - the client address from `request.client` only (the backend never parses `X-Forwarded-For`);
  - server-generated `X-Request-ID` (a client value is only logged, as `upstream_request_id`);
  - a per-request nonce Content-Security-Policy on every page (pages now render dynamically);
  - HSTS and `upgrade-insecure-requests` behind the new runtime `JEV_HTTPS=true` on web;
  - safe post-login redirects;
  - security regression tests (`test_security.py`, `test_security_review2.py`; one test needs PostgreSQL).
- **Migration `0004_feedback_dedup`.** It removes duplicate recommendation feedback and adds partial unique indexes:
  one verdict and one click per user per recommendation (or per movie without one). `POST
  /recommendations/feedback` is now an upsert (latest verdict wins, same 201 body), and the acceptance test checks
  that a repeated dislike counts once.
- **Acceptance test.** The intelligence layer is covered end to end, with 58 checks through the web proxy. It passes
  locally and in Docker, and can be re-run. `docs/demo.md` is a guided walkthrough, and screenshots 22–24 are new.
### Changed in v1.1
- **Docker:**
  - The API is no longer published to the host (`expose` only), and uvicorn no longer trusts forwarded headers from
    any address (no `--forwarded-allow-ips "*"`). By default compose trusts none, because the Next.js proxy forwards
    `X-Forwarded-For` verbatim.
  - The API and trainer run as the host uid, so they can read and write the bind mounts (`registry.json` is written
    0600).
  - The compose file passes `JEV_PROCESSED_DIR`/`JEV_MODELS_DIR`/`JEV_EXPERIMENTS_DIR`, the `JEV_INTEL_*` settings
    and the rate limits.
  - The trainer now calibrates and runs the intelligence evaluation, and `TRAIN_ARGS` reaches it.
  - `.dockerignore` excludes the frontend tests.
  - Compose now defaults to `JEV_ENV=production` (`.env.example` keeps `development` for local demos) and no longer
    sets `JEV_TRUST_PROXY`. `web` gets `JEV_HTTPS` (default false). `JEV_API_URL` stays a build arg (`http://api:8000`).

## 1.0.0 — 2026-09-23
### Added
- Data pipeline: checksum-verified MovieLens download, resumable Wikidata enrichment, validation and preprocessing.
- Recommenders: popularity (+ trending, Bayesian average), per-field TF-IDF content model, item-kNN collaborative
  filtering, implicit ALS with fold-in and exact score attribution.
- Hybrid ranking engine: candidate generation, normalisation, interaction-adaptive weights, quality floor, MMR diversity,
  genre cap, filters and pagination. Explanations built only from model contributions.
- Evaluation: per-user temporal split, P/R/F1/NDCG/MAP/HitRate@K, coverage, diversity, novelty, cold-start protocol,
  validation tuning, reports and plots. Model registry with versioned, pickle-free artifacts.
- FastAPI backend with SQLAlchemy + Alembic (PostgreSQL / SQLite), JWT auth, CSRF protection, rate limiting,
  Redis cache with in-memory fallback, structured logging, admin model activation.
- Next.js frontend: landing, auth, 3-step onboarding, home shelves, ranked list with "Why this?", movie pages,
  discover, taste profile, history, favourites, admin overview, models and experiments. Typeset covers, dark and paper themes.
- Docker Compose stack, acceptance test script, browser journey with screenshots, and test suite.
### Fixed during development
- Quadratic label filtering in the featurizer (250 s → 0.9 s).
- Genre normalisation turned "science fiction film" into "science"; MovieLens Sci-Fi and Wikidata now share a token.
- The metric-key parser rejected `f1@K`; F1 now syncs to the database.
