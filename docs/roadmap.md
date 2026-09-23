# JEV Roadmap

JEV is built in 25 phases. Each phase is done only after it has been run and verified.
Live status is tracked in [progress.md](progress.md).

| # | Phase | Deliverable |
|---|-------|-------------|
| 1 | Audit environment & repository | Tooling inventory, constraints |
| 2 | Architecture & project structure | `ARCHITECTURE.md`, directory layout |
| 3 | Dependencies & environments | `pyproject.toml` (uv), `frontend/package.json`, `.env.example` |
| 4 | Dataset download & preprocessing | `scripts/download_data.py`, `scripts/preprocess_data.py`, Wikidata enrichment |
| 5 | Popularity baseline | `jev_ml.models.popularity` |
| 6 | Content-based recommender | TF-IDF over genres/title/tags/director/cast/keywords, cosine |
| 7 | Collaborative filtering | Item-item KNN over sparse implicit matrix |
| 8 | Matrix factorization | Implicit ALS (Hu–Koren–Volinsky) with fold-in for unseen users |
| 9 | Hybrid ranking engine | Candidate gen → normalization → weighted scoring → MMR diversity |
| 10 | Evaluation framework | Temporal per-user split, P/R/F1/NDCG/MAP/HR@K, coverage, diversity, novelty |
| 11 | Train & evaluate all models | Artifacts in `models/`, experiment runs in `experiments/` |
| 12 | PostgreSQL database | SQLAlchemy models + Alembic migrations |
| 13 | FastAPI backend | REST API, structured logging, error handling |
| 14 | Auth & user profiles | JWT (httpOnly cookie + bearer), Argon2 hashing, taste profile |
| 15 | Recommendation API | Personalized / similar / trending / feedback |
| 16 | Next.js frontend | All pages, responsive, laptop-first |
| 17 | Recommendation explanations | Reasons generated only from real contributing signals |
| 18 | Model/experiment dashboard | Admin pages with metric charts, active-model switching |
| 19 | Redis caching | Rec cache + rate limiting, in-memory fallback |
| 20 | Docker environment | `docker-compose.yml`, Dockerfiles |
| 21 | Full test suite | unit / integration / ML / e2e |
| 22 | Security review | `SECURITY.md` |
| 23 | End-to-end demo | Acceptance scenario executed |
| 24 | Fix discovered issues | — |
| 25 | Final documentation | README and docs/* |

## Beyond 1.0

- Sequence-aware models (SASRec-style) once more interaction data exists
- Learned-to-rank hybrid (LightGBM LambdaMART) on logged feedback
- Online A/B testing using the experiment table
- Poster art via optional TMDB integration (API key, user-provided)
- Scheduled incremental retraining
