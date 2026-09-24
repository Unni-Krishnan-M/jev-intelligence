# Changelog

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
