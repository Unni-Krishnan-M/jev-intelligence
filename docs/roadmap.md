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

## Beyond 1.0 (original list)

- Sequence-aware models (SASRec-style) once more interaction data exists
- Learned-to-rank hybrid (LightGBM LambdaMART) on logged feedback
- Online A/B testing using the experiment table
- Poster art via optional TMDB integration (API key, user-provided)
- Scheduled incremental retraining

## 1.1 (done)
The intelligence and early-warning layer, then v1.1:
- score decisions and decision batches;
- calibrated recommendation confidence;
- normalised run tables, evidence and history;
- the audit log, recommender monitoring and evaluation runs;
- Vitest;
- a security pass;
- a Docker hardening pass (internal-only API, no trusted forwarded headers by default, host-uid bind mounts).

See [progress.md](progress.md) for the evidence.

## Next steps (ordered by value, from what 1.1 measured)
| # | Item | Why (measured) | Sketch |
|---|---|---|---|
| 1 | **Auto-resolution of stale warnings** | Open warnings are never closed when a run stops producing them, so the queue only grows (DEFERRED since 1.0 of the layer) | Resolve after N consecutive successful runs without the key, with a `system` event and an audit row; keep manual resolution as the default for high severity |
| 2 | **Better recommendation discrimination** | Confidence is well calibrated (ECE ≤ 0.0014) but ranks weakly: AUC 0.57–0.63, Brier skill +0.45 % warm, about 0 for short profiles | Learn per-stage hybrid weights on the validation split; add features (profile size, item popularity, signal agreement) to the calibrator; report AUC as a release gate |
| 3 | **Learning-to-rank on logged feedback** | Popularity still beats the hybrid at cold start (NDCG@10 0.046 vs 0.035), and weights are hand-set per stage | LightGBM LambdaMART over the six signals and context, trained on held-out ratings and logged `recommendation_feedback`; offline A/B against the current hybrid |
| 4 | **Change-point false-alarm rate** | 7 % false alarms against 1 % nominal on synthetic AR(1) series; detection 32.5 % | Estimate AR(1) by a bias-corrected estimator or block bootstrap for the null; require a minimum segment length; re-run `evaluate_intelligence.py` and publish the new rate |
| 5 | **Live-traffic validation** | Every live signal says "cannot be assessed" on the static 2018 snapshot; lapse and warning precision are only measured offline or synthetically | Collect a few weeks of app ratings and feedback; compare forecasts with actuals (`prediction` feedback); report warning precision from operator verdicts once ≥ 5 exist |
| 6 | Rater-detector budget | The 2 % review budget is partly spent on genuine heavy users | Add tenure and diversity features; calibrate against the injection study per attack type |
| 7 | Client-address attribution behind the web proxy | Next.js rewrites forward `X-Forwarded-For` verbatim, so the API trusts no forwarded header and all clients share one rate-limit bucket (docs/deployment.md) | Put an edge proxy that overwrites the header in the compose stack, or set the header from the socket peer in `frontend/src/proxy.ts` |
| 8 | Scheduled retraining with app interactions | App users are served by fold-in and never enter training | Nightly trainer job gated by the `retrain_model` decision |
| 9 | Sequence-aware models, larger MovieLens variants, optional TMDB artwork | From the 1.0 list | — |
