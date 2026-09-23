# Changelog

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
