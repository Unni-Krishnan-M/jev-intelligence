# Changelog

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
