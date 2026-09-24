# JEV Progress

_Last updated: 2026-09-24_

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
- (none; Phase 20/23, the Docker Compose build and the acceptance scenario against the containerised stack, was
  completed on 2026-09-24. See the implementation report below.)

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

## Upgrade report (updated 2026-09-24)
| Area | Status | Evidence |
|---|---|---|
| Data ingestion, validation, freshness, provenance | DONE | `intel/ingest.py`; 10 weighted checks; per-source SLA; freshness card verified in the browser |
| Signals (dedup, strength, evidence) | DONE | `intel/signals.py`; 18 signals on real data (21 as of 2017-07-01) |
| Trends + change points | DONE / PARTIAL | Mann–Kendall/Theil–Sen + BH FDR; AR(1)-null change points over-alarm (7 % vs 1 % nominal) |
| Anomalies (series + raters + live feedback) | DONE / PARTIAL | Horror spike 2017-05 (robust z 7.85, high) in the replay, checked by the acceptance test; the live-feedback test has only run on synthetic app data |
| Forecasts + lapse prediction | DONE | median MASE 0.93 vs naive 1.15; lapse AUC 0.886, ECE 0.061 |
| Risk engine | DONE | 8 kinds; some impact weights are declared estimates (labelled) |
| Decision layer ("JEV") | DONE | 6 typed specs incl. the `score` kind (`editorial_slot_share`, 80 % interval), 3 atomic batches, abstention |
| Early warnings + lifecycle | DONE | dedup, suppression, reopen, audit trail; transition + history checked end to end |
| Scenarios, action planner, feedback | DONE | effort per action type is a declared estimate |
| API, persistence, metrics | DONE | migrations 0002–0004 on SQLite and PostgreSQL 17 (Docker, first start); `/admin/metrics` |
| Normalised tables, evidence, history (v1.1) | DONE | migration 0003; `/intel/evidence`, `/intel/history/*` checked end to end |
| Audit log (v1.1) | DONE | login, runs, transition, feedback and scenario rows matched by request id; no password in any row |
| Recommendation confidence (v1.1) | DONE / PARTIAL | calibrated (ECE ≤ 0.0014), but discrimination is weak (AUC 0.57–0.63) |
| Recommender monitoring (v1.1) | DONE | `/intel/recommendations`; served rows all carry a confidence |
| Operator console | DONE | 16 sections + 3 detail pages; the demo flow was walked in a real browser on 2026-09-24 |
| Frontend unit tests | DONE | Vitest: 65 tests in 6 files (was MISSING) |
| Docker stack incl. intelligence layer | DONE | `docker compose up -d --build` → acceptance 58/58 (production mode); 0 ERROR log lines |
| CI | DONE | `.github/workflows/ci.yml` (not re-run for this report; the local gates are listed below) |
| Auto-resolution of stale warnings | DEFERRED | open warnings are never auto-resolved |
| Client IP behind the web proxy | PARTIAL | not spoofable (no forwarded header is trusted), but all clients share web's rate-limit bucket; see known limitations |

---

# Implementation report: v1.1 upgrade (spec phase 17, 2026-09-24)

Status words: **DONE** (implemented and verified by a test, an end-to-end run or a browser check), **PARTIAL** (works,
with a measured gap), **BROKEN**, **MISSING**, **DEFERRED** (consciously postponed). Every status below rests on
evidence gathered while writing this report. Anything not re-run is marked as such.

## 1. Repository audit summary
- Before v1.1 (commit `2ac635d`):
  - hybrid recommender, evaluation framework, FastAPI + Alembic 0001/0002, Next.js app with the 13-section
    intelligence console, Docker Compose, CI;
  - 99 pytest tests (94 pass + 5 skipped without data);
  - no frontend test runner;
  - Docker acceptance had never been completed (listed under "In progress").
- Working tree now: 74 files changed against HEAD (+5,201 / −367), plus 26 new files or directories (lists below).
- Toolchain: Python 3.12 (uv), Node 26 / pnpm 11, Docker 29 / Compose v5, Chrome for Playwright. Ports 5432/6379 on
  this machine belong to another compose project (`jev`, in ~/Desktop/JEV). It was left untouched; this stack does not
  publish db/cache ports.

## 2. Features
### Completed before the upgrade (re-verified)
| Feature | Status | Evidence |
|---|---|---|
| Hybrid recommender, explanations, cold start | DONE | acceptance part 1 (23 checks, local and Docker); hybrid NDCG@10 0.1214 |
| Intelligence pipeline (ingest → feedback) | DONE | acceptance part 2; run ≈ 1.0 s local, 1.1 s in Docker |
| Warning lifecycle, scenarios, operator feedback | DONE | acceptance: transition + history, saved scenario, feedback counts |
| Offline intelligence evaluation | DONE | `intel-eval-20260923T134257Z` served by `/intel/evaluation` and `/intel/evaluation/runs` |

### Completed during the upgrade
| Feature | Status | Evidence |
|---|---|---|
| Score decisions (`kind: score`, `confidence_kind: interval`) | DONE | e.g. Western 1.36 % [0.74, 2.04]; Horror 7.06 % [3.87, 13.3]; 15/18 genres answered, 3 abstain (coverage < 0.6) |
| Decision batches (one hashed state, atomic) | DONE | model_governance (2), genre_programming (36, 3 abstained), audience (4); `batch_id` filter checked |
| Recommendation confidence (isotonic, validation split) | DONE | 20/20 served items carry it (0.0072–0.0189); test ECE 0.00075 (full profile) |
| Normalised tables + evidence + history (0003) | DONE | 179 evidence rows per latest-data run (202 for the replay); history over 2+ runs |
| Audit log | DONE | every expected action found by request id; admin password absent |
| Recommender monitoring page and endpoint | DONE | served/with-confidence, feedback by reason code, 10-bin histogram |
| Feedback dedup / upsert (0004) | DONE | a repeated dislike counts once (acceptance, Docker and local) |
| Security pass | DONE | 25 + 16 regression tests; CSP nonce, HSTS behind `JEV_HTTPS`, server-generated request ids |
| Frontend unit tests (Vitest) | DONE | 65 tests, 6 files |
| Acceptance test covering the intelligence layer | DONE | 58 checks; local 58/58 twice, Docker 57/57 ×4 then 58/58 |
| Docker: intelligence layer, calibration, hardening | DONE | see section 7 |
| Demo walkthrough | DONE | `docs/demo.md`; flow walked in Chrome on a fresh DB (feedback, acknowledge, replay, scenarios) |
| Screenshots 22–24 | DONE | evidence, recommender, audit regenerated and inspected |

### Partial
- **Recommendation confidence discrimination.** AUC 0.57–0.63, Brier skill +0.45 % warm. It is calibrated, but it
  barely ranks.
- **Change points.** 32.5 % detection, 7 % false alarms against 1 % nominal.
- **Live signals.** Freshness for the app source "cannot be assessed" until real traffic exists. The live-feedback
  anomaly path has only run on synthetic data.
- **Client attribution through the web proxy.** Secure by default (not spoofable), but per-IP limits are shared by
  all clients (see limitations).

### Deferred
- Auto-resolution of stale warnings.
- Learning-to-rank; sequence models; scheduled retraining with app interactions.
- Live-traffic validation.

## 3. Files
### Created (untracked in git)
- **Backend:** `backend/jev_api/migrations/versions/0003_intel_normalized.py`, `0004_feedback_dedup.py`,
  `backend/jev_api/services/audit.py`, `backend/jev_api/services/feedback.py`
- **ML:** `ml/jev_ml/calibration.py`, `ml/jev_ml/intel/batches.py`; `scripts/calibrate_recommendations.py`
- **Frontend:**
  - pages: `frontend/src/app/intel/{audit,evidence,recommendations}/`;
  - components: `frontend/src/components/jev/intel/history-timeline.tsx`, `frontend/src/components/jev/rec-confidence.tsx`;
  - lib: `frontend/src/lib/{csp,safe-redirect,version}.ts`;
  - tests: `frontend/src/__tests__/` (6 test files + setup + render helper), `frontend/vitest.config.mts`.
- **Tests:** `tests/integration/test_intel_v11.py`, `test_security.py`, `test_security_review2.py`,
  `tests/intel/test_intel_decisions_v11.py`, `tests/unit/test_calibration.py`
- **Docs:** `docs/demo.md`; `docs/screenshots/22-intel-evidence.png`, `23-intel-recommender.png`, `24-intel-audit.png`
- **Artefact (not in git):** `models/jev-20260923T100141Z-bbb2e4c9/calibration.json`

### Modified (`git diff --stat HEAD`)
- **Backend (17):** `__init__.py`, `config.py`, `deps.py`, `main.py`, `models.py`, `schemas.py`,
  `routers/{admin,auth,health,intel,movies,recommendations,users}.py`, `services/{intel,ml,recommend,sync}.py`
  (`routers/intel.py` +443, `services/intel.py` +299, `models.py` +266)
- **ML (10):** `__init__.py`, `data/dataset.py`, `engine.py`, `models/hybrid.py`,
  `intel/{config,decisions,forecast,pipeline,series,trends}.py` (`decisions.py` +358)
- **Frontend (22):**
  - `package.json`, `pnpm-lock.yaml`, `README.md`, `src/proxy.ts`, `src/app/layout.tsx`;
  - `src/app/intel/{layout,page}.tsx`;
  - `src/app/intel/{decisions,decisions/[id],evaluation,risks,series/[id],signals}/page.tsx`;
  - components `auth/auth-form`, `providers`, `why-dialog`, `intel/{badges,charts,states}`;
  - `lib/{intel,intel-types,types}.ts`.
- **Scripts (3):** `acceptance_test.py` (+385), `capture_screenshots.py`, `train_models.py`
- **Tests (4):** `conftest.py`, `intel/test_intel_pipeline.py`, `intel/test_intel_real_data.py`, `ml/test_real_artifacts.py`
- **Ops (6):** `docker-compose.yml`, `docker/api.Dockerfile`, `docker/web.Dockerfile`, `.dockerignore`, `.env.example`,
  `.github/workflows/ci.yml`
- **Docs (11):** `README.md`, `CHANGELOG.md`, `SECURITY.md`, `docs/{api,deployment,development,evaluation,intelligence,recommendation-algorithms,roadmap,progress}.md`
- **Build (2):** `pyproject.toml`, `uv.lock` (version 1.1.0)

## 4. Database migrations
| Revision | Content | Verified |
|---|---|---|
| `0002_intelligence` | runs, warnings (+ events, partial unique open-key index), decisions, scenarios, feedback | SQLite (tests, local stack), PostgreSQL 17 (Docker first start) |
| `0003_intel_normalized` | `intel_signals/trends/anomalies/forecasts/risks/evidence` (CASCADE), `audit_logs`, `intel_evaluation_runs`; `intel_decisions.batch_id/answer_value/answer_interval/scale`; `recommendations.confidence(_kind)`; `interval` confidence kind | round-trip test on SQLite; upgrade on PostgreSQL 17 in Docker |
| `0004_feedback_dedup` | removes duplicate recommendation feedback; partial unique indexes (one verdict + one click per user per recommendation or movie) | SQLite tests; PostgreSQL 17 upgrade in Docker; acceptance dedup check |

## 5. API changes
- **New (admin only):**
  - `GET /intel/evidence`, `GET /intel/history/{signals|risks|trends|anomalies}`;
  - `GET /intel/recommendations`, `GET /intel/evaluation/runs`, `GET /intel/decisions/batches`;
  - `GET /admin/audit`.
- **Changed responses:**
  - Decisions gain `batch_id, answer_value, answer_interval, scale`, and `answer` is a number for score decisions.
  - Recommendation items, `SimpleRecItem` and history items gain `confidence`/`confidence_kind`.
  - `/health/ml` and `/models/active/summary` gain `calibration`.
  - `/admin/metrics` gains `audit_entries_by_action`, `audit`, `intel_rows_persisted`,
    `intel_evidence_rows_per_run` and `intel_pipeline.last_evidence_rows`.
  - `/intel/decisions` gains a `batch_id` filter.
  - `POST /recommendations/feedback` is an upsert (same 201 body).
  - `X-Request-ID` is always generated by the server.
- Full reference: [api.md](api.md).
- Doc nit found while testing: warning `history` rows use `from_status`/`to_status`, while api.md's warning line says
  "from, to". The acceptance test uses the real keys.

## 6. Frontend, ML and decision-layer changes
- **Frontend:**
  - Evidence explorer, Recommender monitoring and Audit log pages.
  - Score decisions drawn on their scale with the interval; the decision log grouped by batch.
  - History timelines on signals and risks.
  - Member-facing confidence ("≈ n % chance you rate it 4★+").
  - Per-request nonce CSP; HSTS behind `JEV_HTTPS`; safe redirects.
  - Vitest.
- **ML:**
  - `calibration.py`: an isotonic calibrator per profile-size stratum, fitted on the validation split and evaluated on
    test.
  - Engine `confidence`.
  - Forecast window-mean conformal intervals for the slot share.
- **Decision layer:**
  - The `editorial_slot_share` score policy (`slot-share-1.0.0`).
  - `batches.py`: 3 atomic batches with a sha1 state hash and mutation detection.
  - Replays make retrain and serving abstain.

## 7. Docker and operations (changed in this pass)
- **API not published.** `api` uses `expose: 8000`. The browser, and the acceptance test, reach it only through web
  (`:3000/api`). `curl localhost:8000` fails, as intended.
- **Proxy headers.**
  - `--forwarded-allow-ips "*"` was removed. `FORWARDED_ALLOW_IPS` comes from `JEV_FORWARDED_ALLOW_IPS`, default
    `127.0.0.1` (trust nobody).
  - `web` has a fixed address, `172.29.84.10` on subnet `172.29.84.0/24`, for operators who add an edge proxy.
  - Reason: Next.js rewrites forward `X-Forwarded-For` verbatim. With web trusted, a spoofed `6.6.6.6` reached the
    audit log. With the default, it is attributed to `172.29.84.10`.
- **API and trainer run as the host uid** (`JEV_UID`/`JEV_GID`, default 1000). The first `up` failed with
  `PermissionError: /app/models/registry.json`: the registry is written 0600, and the image uid is 10001.
- **Intelligence layer paths.** `JEV_PROCESSED_DIR`, `JEV_MODELS_DIR` and `JEV_EXPERIMENTS_DIR` are set explicitly,
  and the `JEV_INTEL_*` settings and rate limits are passed through.
- **Trainer.**
  - It now runs `train_models.py --calibrate` and then `evaluate_intelligence.py`.
  - `TRAIN_ARGS` is now passed into the container. Before, it was never set inside it, so `--quick` was silently
    ignored.
  - Checked without training: the uid can write `models`, `experiments` and `data/processed`, and both scripts
    parse their arguments.
- **Other compose defaults.** `JEV_ENV` defaults to production. `web` gets `JEV_HTTPS` (default false; HSTS checked
  present when true, absent by default). `JEV_API_URL=http://api:8000` stays a build arg.
- **`.dockerignore`** excludes `frontend/src/__tests__`, `vitest.config.mts`, `tsconfig.tsbuildinfo`,
  `jev.egg-info` and `.github`.
- **Measurements:**
  - build 2 min 18 s (a rebuild plus up took 2 min 50 s);
  - images: api 1.52 GB (356 MB compressed), web 303 MB;
  - healthy about 10 s after `up`;
  - idle memory: api 343 MiB, web 51 MiB, db 52 MiB, cache 12 MiB.
- Runs used a separate project name (`-p jevaccept`) with a scratch `--env-file` (no secrets in the repo). They were
  torn down with `down -v`, and the images were removed. The existing `jev-recsys` volume and containers were not
  touched.

## 8. Tests added
| Suite | Count | Delta vs HEAD |
|---|---|---|
| pytest total | **174 collected: 173 passed, 1 skipped** (PostgreSQL-only, needs `JEV_TEST_POSTGRES_URL`) | from 99 |
| unit / ml / intel / integration | 47 / 9 / 39 / 79 | |
| new files | test_calibration 11, test_intel_decisions_v11 11, test_intel_v11 11, test_security 25, test_security_review2 16 | +74; plus +1 in the modified real-data tests |
| Vitest | **65 in 6 files** | from 0 |
| Acceptance | **58 checks** (23 recommender + 35 intelligence) | from 23 |

## 9. Validation results (run on 2026-09-24)
| Gate | Result |
|---|---|
| `uv run pytest -q` | 173 passed, 1 skipped (≈25 s) |
| `uv run ruff check .` / `ruff format --check .` / `mypy` | clean / 107 files formatted / no issues in 77 files |
| `pnpm test` / `tsc --noEmit` (after `next typegen`) / `eslint src` / `pnpm build` | 65 passed / clean / clean / ok |
| `pnpm audit` (incl. dev deps) and `--prod` | No known vulnerabilities found |
| Acceptance, local (SQLite, memory cache, `pnpm start`) | 57/57 ×2 and 58/58 ×2 (re-runs on the same DB); 0 ERROR log lines |
| Acceptance, Docker (PostgreSQL 17, Redis 7) | 57/57 ×4 (development mode; two image builds), then 58/58 (production mode); 0 ERROR log lines |
| Browser | demo flow walked in Chrome; screenshots 22–24 inspected |
| CI on GitHub | not re-run (nothing pushed) |

## 10. Known limitations
- MovieLens is a 2018 snapshot, so live signals need real traffic. The genre trend stream is thin (10–15 raters a
  month).
- Change points over-alarm (7 % vs 1 % nominal). The rater detector spends budget on heavy genuine users. The lapse
  base rate is 74 %.
- Recommendation confidence is calibrated but weakly discriminative (AUC 0.57–0.63). Popularity beats the hybrid at
  cold start.
- Open warnings are never auto-resolved.
- **Behind the Next.js proxy, the API cannot see client addresses.**
  - Rate limits (240/min general, 20/min auth) are shared by all clients of one web container.
  - A burst of logins from many users can therefore hit 429.
  - Fix: an edge proxy that overwrites `X-Forwarded-For` plus `JEV_FORWARDED_ALLOW_IPS`, or set the header in
    `src/proxy.ts`.
- `models/registry.json` is written with mode 0600 (`ml/jev_ml/registry.py`, `tempfile.mkstemp`). Any container or
  service user other than the file's owner cannot read it. Compose works around this by running as the host uid.
- JWTs are not revocable before expiry; there is no password reset.
- Posters are typeset.

## 11. Next recommended engineering tasks
See [roadmap.md](roadmap.md#next-steps-ordered-by-value-from-what-11-measured):
1. auto-resolution of stale warnings;
2. better recommendation discrimination;
3. learning-to-rank on logged feedback;
4. change-point false-alarm rate;
5. live-traffic validation;
6. rater-detector budget;
7. client-address attribution behind the web proxy (and `chmod 0644` on the registry write);
8. scheduled retraining with app interactions.

---

# Platform migration: JEV as a domain-flexible engine (v1.2, started 2026-09-24)

Contract: [platform.md](platform.md). Goal: make JEV itself the reusable decision and early-warning engine, with the
movie recommender as the first domain adapter and a generic structured-dataset adapter as the second.

## Audit (before any change)
- Baseline green:
  - pytest 174 passed / 1 skipped; ruff, format and mypy clean;
  - frontend typegen, tsc, lint, Vitest (65) and build clean.
- The intelligence pipeline (`ml/jev_ml/intel`, about 5.6k lines) is movie-coupled:
  - ingest validates ratings;
  - series are hard-wired to genres and platform;
  - raters, lapse, model governance and genre-programming decisions are inline;
  - warnings come straight from risk and anomaly levels, not from a decision.
- Generic already: robust-z series anomalies, trends and change points, forecasting, scenarios, the decision
  framework (confidence kinds, abstention, batches), the warning lifecycle, feedback, evidence and audit
  persistence.
- Missing: core/domain separation, a second domain, an early-warning decision gating warnings, preference drift,
  recommendations downstream of a decision, per-user intelligence and scenarios, and a platform-first UI.

## Status
- [x] Core extraction + movie adapter + generic adapter (FRED unemployment) + early-warning decision + evaluation: golden tests pass (synthetic + real); movie ≈1 s, unemployment ≈0.2 s
- [x] Preference drift + recommendation strategy + user scenarios + drift evaluation: 37 tests; detector precision 0.92, recall 0.29 (splice 20); adaptation not proven (7 users) → policy serves standard
- [x] Frontend platform restructure (identity, nav, domain switcher, /me/intelligence): Vitest 88, tsc/lint/build clean (signed-in pages await the backend)
- [x] Backend: domain dimension (migration 0005, SQLite + PostgreSQL 17), /intel/domains, /me/intelligence*, recommendations intelligence block: 276 passed; acceptance 69/69; strategy step p50 0.37 ms cached, ~5.5 ms uncached
- [x] Integration, end-to-end demo, docs, final report (see "Final report: platform migration v1.2.0")

## Final report: platform migration v1.2.0 (2026-09-24)
**Before:**
- JEV was a movie recommender. An intelligence layer (`ml/jev_ml/intel`) was hard-wired to ratings, genres, raters,
  lapse and model governance.
- Warnings came straight from risk and anomaly levels.
- The UI was a movie app with an admin console.

**Preserved:**
- All recommender models (popularity, TF-IDF, item-kNN, ALS, hybrid with adaptive weights and MMR).
- Evaluation, calibration, registry and artifacts.
- Every v1.1 endpoint, table and page.
- The `jev_ml.intel` API, now a compatibility shim.
- Movie intelligence outputs: golden tests confirm the same ids, keys, answers, warning set and numbers on synthetic
  and real data. The only differences are the additive fields and the new early-warning decisions.

**Added:**

| Area | Status | Evidence |
|---|---|---|
| Domain-independent core (`ml/jev_ml/core`) + `DomainAdapter` protocol + registry | DONE | never imports recommender code (checked by grep); `run_domain` runs both domains |
| Movie adapter | DONE | golden tests (4 synthetic variants + 2 real runs) |
| Generic CSV + YAML adapter; US unemployment (BLS via FRED) | DONE | real runs: default, 2008-06-01 (10 warnings, IL/NY critical) and 2020-05-01 |
| Early-warning decision gating warnings (`ewl-1.0.0`) | DONE | randomised monotonicity tests; every warning carries `decision_id` |
| Preference drift, recommendation strategy, preference scenarios | DONE | 38 tests; `/recommendations` is served downstream of the strategy decision |
| Evaluations | DONE | `platform-eval-20260924T051637Z`, `drift-eval-20260924T045528Z` (README tables) |
| API: `/intel/domains`, `?domain=`, `/me/intelligence*`, recommendations `intelligence` block | DONE | 16 platform API tests; acceptance 69/69 on real data |
| Migration 0005 (domain column, per-domain open-warning index, user_intel_feedback) | DONE | round trip on SQLite and PostgreSQL 17 |
| Platform-first UI: landing, navigation, domain switcher, `/me/intelligence` | DONE | real-browser walk of both domains; 0 overflow on 132 page visits; Vitest 91+ |
| Code review (`/code-review high`) and security review | DONE | security: no findings; code review: 10 findings, 9 fixed, 1 documented (state hash) |
| Drift adaptation enabled by default | DEFERRED | the effect is shown on only 7 users; the policy requires ≥ 30 |
| Domain-specific stages for generic datasets | DEFERRED | the generic adapter runs core stages only |
| Warning quality on the generic domain | PARTIAL | precision 0.36 and recall 0.52; warnings lag turning points |

**Known limitations:** see README → Limitations.
