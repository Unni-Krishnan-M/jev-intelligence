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

---

# Upgrade: Intelligence & early-warning layer (started 2026-09-23)

Spec: upgrade JEV into an intelligent decision & early-warning engine without discarding the recommender.
Design contract: [intelligence.md](intelligence.md).

## Audit (before any change)
- Baseline green: 57 pytest tests pass, ruff and mypy clean.
- Complete: data ingestion + validation (MovieLens + Wikidata), 5 recommenders + hybrid, evaluation framework,
  model registry, FastAPI (auth, CSRF, rate limits, request ids, structured logs), recommendation feedback,
  admin model/experiment dashboards, Docker Compose, docs.
- Missing relative to the spec: signals, trends/change points, anomaly detection, forecasting, risk scoring,
  typed decision layer, early warnings with lifecycle, scenarios, action planner, operator feedback,
  intelligence evaluation, metrics endpoint, operator console.
- Interpretation: "JEV" in the spec is the decision layer. Here it is implemented as typed, versioned decision
  policies over computed evidence (no external LLM, no API key; runs offline on any laptop).

## Status
- [x] ML intelligence core (`ml/jev_ml/intel`) + evaluation: 28 tests; ~0.85 s per run; report in experiments/intel-eval-20260923T134257Z
- [x] Persistence (migration 0002, 6 tables) + `/intel/*` API + `/admin/metrics`: 14 API tests; real-data smoke test clean
- [x] Intelligence console (frontend): 13 sections + 3 detail routes; tsc, lint, build clean 
- [x] Integration: real-browser review against real data found 8 UI issues (freshness labels, stage order, confidence kinds,
  number formats, protocol notes, mobile overflow). All fixed and re-verified: 0 px overflow at 390/1366, both themes
- [x] CI workflow (`.github/workflows/ci.yml`); clean-copy run: 94 passed, 5 skipped (need real data)
- [x] README + ARCHITECTURE updated
- [x] Screenshots 14–21 (console) + refreshed 01–13; final report below

## Upgrade report (2026-09-23)
| Area | Status | Evidence |
|---|---|---|
| Data ingestion, validation, freshness, provenance | DONE | `intel/ingest.py`; 10 weighted checks; per-source SLA |
| Signals (dedup, strength, evidence) | DONE | `intel/signals.py`; 18 signals on real data |
| Trends + change points | DONE | Mann–Kendall/Theil–Sen + BH FDR; AR(1)-null change points (7 % false alarms, above nominal) |
| Anomalies (series + raters + live feedback) | DONE / PARTIAL | live-feedback test only exercised on synthetic app data |
| Forecasts + lapse prediction | DONE | median MASE 0.93 vs naive 1.15; lapse AUC 0.886, ECE 0.061 |
| Risk engine | DONE | 8 kinds; some impact weights are declared estimates (labelled) |
| Decision layer ("JEV") | DONE | 5 typed specs, versioned policies, confidence kinds, abstention |
| Early warnings + lifecycle | DONE | dedup, suppression, reopen, audit trail; 14 API tests |
| Scenarios, action planner, feedback | DONE | effort per action type is a declared estimate |
| API, persistence, metrics | DONE | migration 0002 (SQLite + PostgreSQL 17 verified); `/admin/metrics` |
| Operator console | DONE | 13 sections + 3 detail pages; real-browser verified |
| CI | DONE | `.github/workflows/ci.yml` (clean-copy run green) |
| Auto-resolution of stale warnings | DEFERRED | open warnings are never auto-resolved |
| Frontend unit tests | MISSING | no JS test runner in the project; covered by tsc/lint/build + browser checks |
