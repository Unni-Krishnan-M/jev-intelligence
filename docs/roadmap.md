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

## 1.2 and 1.3 (done)
- 1.2: the domain-independent core, the generic CSV + YAML adapter (US unemployment), preference drift and the
  per-member recommendation strategy decision.
- 1.3 (Phase 2): the append-only event log with idempotency and bitemporal replay; gated retraining with promotion,
  rollback and lineage; online experiments; replay isolation and decision lineage; stale-warning auto-resolution
  (done, was item 1 below in 1.1); the change-point false-alarm fix (9.9 % → 1.3 %, was item 4); scheduled,
  decision-driven retraining that includes app interactions (was item 8); client-address signing behind the web
  proxy (was item 7); the CTA ridership domain; the two-protocol recommender benchmark; the security hardening.
  Evidence: [PROJECT_COMPLETION_REPORT.md](PROJECT_COMPLETION_REPORT.md).

## Next steps (ordered by value, from what 1.3 measured)
| # | Item | Why (measured) | Sketch |
|---|---|---|---|
| 1 | **Live traffic** | Every online-experiment result is an offline replay; movie live signals are "cannot be assessed" on the 2018 snapshot | Run one experiment on real members; compare forecasts and warnings with realised outcomes |
| 2 | **Paired offline experiment replay** | The between-arm replay needs ~13,400 members per arm; arm baselines move ~0.01 NDCG@10 by assignment alone | Score every member under every variant in `scripts/simulate_ab_replay.py` and test within member |
| 3 | **Warning skill** | Unemployment lift 1.41 (2019–21: 1.04); CTA lift CI 0.52–2.36 and 1.10 on an untouched window; movie lapse uninformative, genre decline never fires | Persistence rule for daily anomalies; per-domain thresholds on separate tuning windows; a baseline rule (Sahm rule); lift CIs in every report |
| 4 | **Recommender power and recency** | Leak-free protocol has 28 users; a 90-day popularity baseline is 0.044 NDCG@10 ahead (n.s.); popularity leads at cold start | ML-1M/25M re-run (same code); decayed popularity inside the hybrid; learning-to-rank on logged feedback |
| 5 | **Governance statistics** | The calibration gate's ECE margin cannot fail at ~1 % base rates; the joint pass rate of three accuracy gates is below 80 % | Shared calibration set with Brier skill or relative calibration; consider a joint (intersection-union) power target |
| 6 | **Replay pinning in the API** | A movie replay reads the active model (recorded, not pinned) | Accept `model_version` on `POST /intel/runs` and thread it to `load_default_inputs` |
| 7 | **Operations** | Docker acceptance not re-run for 1.3; `idempotency_keys` / `revoked_tokens` not pruned; per-process run lock | Re-run acceptance on compose; a `retention.prune` job; a shared lock |
| 8 | Rater-detector budget | The 2 % review budget is partly spent on genuine heavy users | Tenure and diversity features |
| 9 | Sequence-aware models, optional TMDB artwork | From the 1.0 list | — |
