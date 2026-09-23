# JEV Progress

_Last updated: 2026-09-23_

## Completed
- **Phase 1 — Audit.** Linux x86-64 dev box; Python 3.12 via uv, Node 26, pnpm, Docker 29 / Compose v5. No local
  PostgreSQL/Redis (they come from Compose). Empty repository.
- **Phase 2 — Architecture.** `ARCHITECTURE.md`; single uv project with `ml/jev_ml` + `backend/jev_api`, Next.js in `frontend/`.
- **Phase 3 — Dependencies.** `pyproject.toml` + `uv.lock`, `frontend/package.json` + `pnpm-lock.yaml`, `.env.example`.
- **Phase 4 — Data.** MovieLens small downloaded and MD5-verified; Wikidata enrichment (9,651/9,742 matched); validation
  passes with 0 errors / 0 warnings; processed tables + `dataset_meta.json` (`ml-latest-small-31a303aa-wd-2bb80720a955`).
- **Phases 5–9 — Models.** Popularity, content TF-IDF, item-kNN, implicit ALS (fold-in + attribution), hybrid ranker
  (adaptive weights, quality floor, MMR, filters, pagination), grounded explanations.
- **Phase 10 — Evaluation framework.** Temporal splits, 9 metric families, cold-start protocol, tuning, reports and plots.
- **Phase 11 — Trained & evaluated.** Run `jev-hybrid-v1-20260923T095709Z` → model `jev-20260923T100141Z-bbb2e4c9`
  (active). Test NDCG@10: hybrid 0.1214 · item-kNN 0.1006 · ALS 0.0956 · popularity 0.0775 · content 0.0060 · random 0.0024.
- **Phases 12–15 — Database + API.** SQLAlchemy models, Alembic migration `0001`, JWT (cookie + bearer) with CSRF,
  Argon2id, profiles, recommendation serving with persistence, feedback, admin registry, health endpoints.
- **Phases 16–18 — Frontend.** All 14 pages, explanation badges, "Why this?" breakdowns, admin models + experiments
  dashboards. `pnpm build`, `tsc` and `eslint` all clean.
- **Phase 19 — Cache.** Redis (rec responses keyed by profile version, trending, similar, rate limits) with in-memory fallback.
- **Phase 20 — Docker.** `docker-compose.yml` (db, cache, api, web, trainer profile), non-root images.
- **Phase 21 — Tests.** 56 pytest tests (unit, ML, API integration, real-model checks) pass; ruff, ruff format and mypy clean.
- **Phase 22 — Security review.** `SECURITY.md`.
- **Phase 24 — Fixes found by testing:** quadratic featurizer loop, genre normalisation bug, F1 metric-key parsing,
  stale tuning cache key, synthetic-data timestamp ordering, email-validator rejecting `.local`.
- **Phase 25 — Docs.** README, ARCHITECTURE, SECURITY, CHANGELOG, docs/*.

## In progress
- Phase 20/23 — Docker Compose build + acceptance scenario against the containerised stack.

## Blocked
- (none)

## Next
- Re-run the browser journey against Docker; refresh screenshots with the final model.
- Roadmap items (cold-stage weights / learning-to-rank) — see [roadmap.md](roadmap.md).
