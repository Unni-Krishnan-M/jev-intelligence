# JEV Phase 2 gap matrix (Phase 0 forensic audit)

_Audit date: 2026-09-24. Commit `997f2ad` (v1.2.0), clean working tree. No code was changed._

Status words:
- **working**: implemented and backed by a test or a measured run.
- **partial**: works, with a measured or structural gap.
- **broken**: behaves incorrectly today.
- **missing**: not implemented.

Priorities:
- **P0**: blocks Phase 2 or is a correctness bug.
- **P1**: needed for Phase 2 to be credible.
- **P2**: improvement.

Every entry was checked against code or command output, not only against the docs. Paths are relative to the
repository root. Line numbers refer to `997f2ad`.

## 0. Baseline (run during this audit)

| Gate | Command | Result |
|---|---|---|
| Python tests | `uv run pytest -q` | **277 passed, 2 skipped** in 33.7 s. Both skips are PostgreSQL-only (`JEV_TEST_POSTGRES_URL` not set): `tests/integration/test_platform_api.py:590`, `tests/integration/test_security_review2.py:387`. Real-data tests ran because `data/` and `models/` exist locally. |
| Test distribution | `pytest --collect-only` | core 43, domains 7, drift 38, integration 95, intel 39, ml 9, unit 48 = 279 |
| Real-data-dependent tests | `tests/ml/test_real_artifacts.py`, `tests/intel/test_intel_real_data.py`, `tests/core/test_movie_golden.py` (real variants), `tests/drift/test_user_intel.py::test_real_users_if_available` | 30 collected. They **skip in CI**, because `data/*`, `models/*` and `experiments/*` are gitignored (`.gitignore:22-30`). |
| Lint | `uv run ruff check .` | All checks passed |
| Format | `uv run ruff format --check .` | 161 files already formatted |
| Types | `uv run mypy` | Success: no issues found in 120 source files |
| Web types | `pnpm exec tsc --noEmit` | exit 0 |
| Web lint | `pnpm lint` | exit 0 |
| Web tests | `pnpm test` | **95 passed in 7 files** |
| GitHub CI | `gh run list -L 5` | The last 5 runs on `main` succeeded (the v1.2.0 run took 1 m 26 s) |
| Not run in CI | `.github/workflows/ci.yml` | Tests that need PostgreSQL, the acceptance test (`scripts/acceptance_test.py`), the Docker build, `pnpm audit` and `pip-audit`, the browser journey, and the real-data tests |
| Docs vs reality | README.md:254, 256 | README says 174 pytest and 65 Vitest tests; the actual counts are 279 and 95 |

## 1. Recommender ML

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Base models (popularity, TF-IDF content, item-kNN, ALS, hybrid + MMR) | working | `ml/jev_ml/models/*.py`. The README table (NDCG@10 0.1214) matches `experiments/jev-hybrid-v1-20260923T095709Z/metrics.json`. | Yes, all | None to the models themselves | — |
| Cold start (3 interactions) | partial | README:125: popularity NDCG@10 0.046 vs hybrid 0.035. The hybrid blends cold users only through a linear `behavioral_ramp` plus `cold_start_boost` (`models/hybrid.py:46,54,148-155`). | Ramp and boost code, cold-start protocol (`configs/experiment.yaml:15`) | Tune stage-specific weights on validation per profile-size bucket (0, 1-3, 4-10, >10). The acceptance bar is that the hybrid ≥ popularity at every bucket, or the serving path falls back to popularity below the crossover. No new models. | P0 |
| Evaluation split | partial | `configs/experiment.yaml:7` uses `user_temporal`, which lets other users' future interactions leak into training (`evaluation/split.py:8-11`). `global_temporal_split` exists (`split.py:62-73`), but no experiment in `experiments/` uses it. | `global_temporal_split`, `make_split` | Run and report both protocols. Make global temporal the headline split. Add a leakage-check test that asserts max(train.ts) < min(test.ts) for the global protocol. | P0 |
| Leakage checks | missing | No test asserts the temporal ordering of splits or of calibration inputs. The drift evaluation labels the served model "active (leaky)" (`docs/platform.md:334`). | `tests/unit/test_metrics_split.py` | Add property tests: split ordering, no test rows in the calibration fit (the manifest already records `test_rows_used_for_fit: 0`, so assert it), and no future rows in replays of app data (see §2). | P1 |
| Confidence calibration | partial | ECE 0.00075, but AUC 0.567-0.629 and Brier skill 0.0045 (`models/jev-20260923T100141Z-bbb2e4c9/calibration.json` headline and strata). The calibrator is a 1-D isotonic map on score or rank (`calibration.py:30,93`). Served values are about 0.007-0.019. | Calibrator format, strata, serving interpolation | Add a small logistic/GBM calibrator over profile size, item popularity, signal agreement and rank (sklearn only). Gate the release on AUC ≥ 0.65 on the global temporal split. | P1 |
| Hyper-parameter tuning | working | `training.py:178` (`tune`), validation split only | Yes | Retune on global temporal validation | P2 |
| Learning to rank on logged feedback | missing | roadmap.md:178. There is no LTR code. | The six signals are already logged per served item (`Recommendation.signals`) | sklearn `HistGradientBoosting` or a logistic ranker. LightGBM only if it measurably beats that. | P2 |
| Name collision `jev_ml/signals.py` vs `core/signals.py` vs `domains/movie/signals.py` | partial | Three unrelated modules share one name (interaction weights vs intelligence signals) | — | Rename `jev_ml/signals.py` to `jev_ml/interactions.py` and keep a re-export shim | P2 |

## 2. Intelligence engine (core + movie adapter)

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Core/adapter separation | working | `ml/jev_ml/core/*`. A test forbids core imports of recommender modules (`docs/platform.md:389-390`). Golden tests pass. | Yes | — | — |
| **Replay vs live: warnings** | **broken** | `services/intel.py:805` calls `_upsert_warnings` for every run, including `as_of` replays. A 2017 replay therefore opens or updates *live* warnings with `detected_at=now` (`services/intel.py:825-905`). | Upsert logic | Add `IntelRun.mode ∈ {live, replay}` (migration). Replays persist their result but never touch the warning lifecycle, or write to a replay namespace. | **P0** |
| **Replay vs live: "latest run"** | **broken** | `latest_run` orders by `started_at` whatever the `as_of` (`services/intel.py:948-959`). After a replay, every `/intel/*` read defaults to the replay (`routers/intel.py:133-143`). `_startup_refresh` also treats the replay as a fresh live run and skips the refresh (`services/intel.py:1025-1033`). | — | "Latest" should mean the latest *live* run. Replays are reachable only by `run_id`. | **P0** |
| Replay vs live: app-event clock | broken | App ratings are validated against `now_ts`, not `as_of_ts` (`domains/movie/ingest.py:390`). The live-feedback test windows on `now` (`domains/movie/raters.py:211-215`). A replay therefore mixes 2026 app events into a 2017 analysis. | `_validate_ratings` already takes a cutoff | Filter app events at `as_of`, or mark live stages `skipped: not replayable` in replays (the offline evaluation already excludes them: `experiments/platform-eval-*/REPORT.md`). | P0 |
| Replay vs live: suppression | partial | Replays use suppressions computed at wall-clock `now` (`services/intel.py:517-530`). Offline evaluation passes none. | — | Replays ignore operator suppression (it is live state) and record that in `run.validation` | P1 |
| Offline eval vs API path | partial | `scripts/evaluate_domains.py` calls `run_domain` directly without DB inputs. The API path adds app events and suppression (`services/intel.py:532-548`), so the published precision numbers are not produced by the code path that serves. | — | One `RunContext` builder shared by the API and the evaluators | P1 |
| Warning precision (generic) | partial | Precision 0.36 at base rate 0.30 (lift 1.2), FPR 0.39. 2019-21: precision 0.11 (`experiments/platform-eval-20260924T051637Z/REPORT.md`). | ewl policy, confirmation rules | Report lift over base rate and a baseline rule (the Sahm rule for unemployment). Tune the ewl thresholds per domain on a separate replay window. Add trend-reversal damping so warnings stop after peaks. | P1 |
| Warning precision (movie) | partial | Lapse precision 1.0 with a base rate of 1.0, so it is uninformative. Genre decline never fires (recall 0). Same report. | — | Evaluate lapse warnings on lift at a fixed flag rate, or retire the lapse *warning* (keep the risk) | P1 |
| Forecasts (weak domains) | partial | Unemployment MASE 1.52 vs naive 1.41, beats naive on 3/17 series, 80 % coverage 0.61. There are only 3 models (`core/forecast.py:30`) and no seasonal or theta model. | Backtest harness, MASE, conformal intervals | Add seasonal-naive and theta models (hand-rolled numpy, like Holt). Select on MASE with a "naive unless better by ε" rule. Recalibrate intervals from backtest residual quantiles. | P1 |
| Change points | partial | 7 % false alarms vs 1 % nominal (README Limitations) | AR(1) null | Bias-corrected φ or block bootstrap (roadmap.md:179) | P2 |
| Drift detector recall | partial | Recall 0.29 at precision 0.92 with m=20 (`docs/platform.md:318`); per-aspect recall 0.10-0.19 | Session-permutation design | Report a PR curve over α. Add a pooled omnibus aspect (Fisher/Stouffer across aspects) before Holm. Keep FPR ≤ 0.05. | P2 |
| Preference-drift validation | partial | Adaptation effect measured on **7** users. The policy requires ≥ 30 (`domains/movie/user_intel.py:113,154`). | `scripts/evaluate_drift.py` | Evaluate on ML-latest (full, ~330k users) or ML-25M offline. Keep the policy gate as is. | P2 |
| Auto-resolution of stale warnings | missing | No auto-resolve path in `_upsert_warnings`. roadmap.md:176. | Warning events table | Resolve after N consecutive live runs without the key, as a `system` event plus an audit row | P1 |
| Member strategy decision lineage | partial | The `recommendation_strategy` decision is only cached (`services/user_intel.py:167`), never persisted. Recommendation rows and `user_intel_feedback` store a `decision_id` that cannot be resolved after the cache expires. | `IntelDecision` table and serializer | Persist member decisions (domain `movie`, entity `user:<id>`), deduplicated by decision_id, or add a slim `member_decisions` table | P1 |
| DB CHECK vs core enum | broken (latent) | Core `CONFIDENCE_KINDS` includes `"evidence"` (`core/decisions.py:30`). The DB CHECK allows only 4 kinds (`models.py:294,463`). The first persisted decision with `evidence` confidence fails the **whole run transaction**. | — | Add `evidence` in the same migration that adds the run mode | P0 |
| Lineage identifiers | partial | `IntelRun` has pipeline, data and model versions, but no config hash, input fingerprint or code commit (`models.py:333-363`). Evidence rows link owner ids (good). | `training._git_commit()` | Add `config_hash`, `input_fingerprint` (row counts plus max timestamp per source) and `code_version` to runs | P1 |

## 3. Events / streaming

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Event log | **missing** | Interactions are mutable state rows. A rating is overwritten in place (`routers/movies.py:145-153`) and a delete erases it (`:156-159`). There is no append-only event table (`models.py` table list). | Tables stay as projections | Add an append-only `events` table in Postgres (event_id UUID, `idempotency_key` UNIQUE, type, user, entity, payload, occurred_at, received_at, monotonically increasing `seq`). Current tables become projections written in the same transaction. **No Kafka.** | **P0** |
| Idempotency | missing | `POST /movies/{id}/watch` appends a row on every retry (`routers/movies.py:179`). There is no `Idempotency-Key` handling anywhere (grep found nothing). Feedback is deduplicated by unique indexes (0004), which is the only idempotent write. | 0004 partial-index pattern | `Idempotency-Key` header, optional at first and required for the batch ingest endpoint; stored in a unique column | P0 |
| Event time | partial | The profile uses `Rating.updated_at` as event time (`services/profile.py:37,59`), so a re-rate moves the event. | — | `occurred_at` from the event, `received_at` from the server | P1 |
| External ingestion endpoint | missing | The only writers are the UI endpoints. There is no batch or stream ingest API for a domain. | Generic adapter CSV loader | `POST /events:batch` (admin/service token), validated by domain schema, idempotent | P1 |
| Replay from the log | missing | Replays read the MovieLens snapshot plus current DB state, not an event log | `as_of` machinery | Replay = fold events with `occurred_at ≤ as_of` and `received_at ≤ as_of` (bitemporal, which also models publication lag) | P1 |
| Live signals | partial | Freshness says "cannot be assessed" without traffic (README Limitations). App ratings never enter genre series, only freshness, the staleness count and live feedback (`domains/movie/modelstats.py:98-105`, `raters.py:211`). | — | Series builder reads the event projection | P2 |
| profile_version bump | partial | Read-modify-write `user.profile_version += 1` (`routers/movies.py:32-35`) can lose an update under concurrency (a stale cache key) | — | `UPDATE users SET profile_version = profile_version + 1 RETURNING` | P2 |

## 4. Retraining / model governance

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Feedback → training | **missing** | Training reads only `data/processed/interactions.csv` (`training.py:399`). No ML or script module touches the DB (grep for `sqlalchemy|jev_api` in `ml/` and `scripts/` finds nothing). README:"their feedback is not yet fed back into retraining". | `signals.py` weights; `Interaction` | An export job that turns event projections into a training snapshot with a dataset version, merged with MovieLens under a declared weighting | P0 |
| Scheduled retraining | missing | `INTEL_RUN_TRIGGERS` has `"schedule"` (`models.py:288`) but nothing schedules anything. The only automatic run is the startup refresh (`services/intel.py:1017`). The compose trainer is a manual profile. | `retrain_model` decision (movie decisions), trainer image | A compose `scheduler` service (a Python loop with a Postgres advisory lock), or host cron. Retraining is gated by the existing `retrain_model` decision. No Airflow or Celery. | P1 |
| Promotion gate | **missing** | `run_pipeline(..., activate=True)` activates every new model unconditionally (`training.py:385`, `registry.py:38-44`). `POST /models/{id}/activate` has no metric check (`routers/admin.py:119-141`). | Metrics in the manifest; `EngineHolder.activate` loads before swapping (`services/ml.py:44-50`) | A `governance.promote(candidate, incumbent)` gate: NDCG@10 non-inferiority (paired bootstrap), calibration AUC and ECE bounds, cold-start bucket check. Register as `candidate` by default. | P0 |
| Rollback | partial | Manual only: activate an older version (`routers/admin.py:119`) | `set_active`, audit `model.activate` | Keep `previous` in the registry. `POST /models/rollback` plus an automatic rollback when post-promotion monitors breach | P1 |
| Multi-worker model swap | partial | `EngineHolder` is per process. Activation in one worker does not reload the others (`services/ml.py:18-50`). | — | Workers poll the `registry.json` mtime, or read a DB `active` row, on each request (cheap) | P1 |
| Registry file mode | working (docs stale) | `registry.py:25-26` now chmods 0644. `docker-compose.yml:68`, `docs/deployment.md:35` and `progress.md:295` still say 0600. | — | Fix the docs | P2 |

## 5. Experimentation

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Online A/B | **missing** | `Experiment` holds offline runs only (`models.py:246-265`). There is no assignment, arm or exposure code (grep for variant/bucket/assign/treatment found nothing relevant). The admin "experiments" pages show offline metrics. | `recommendations` table (exposure log), `recommendation_feedback` (outcomes), `engine_with_overrides` (`domains/movie/user_intel.py`) | Postgres tables `online_experiments` and `experiment_arms`. Deterministic assignment by sha256(salt, user_id). Add `experiment_id` and `arm` to `recommendations`. SQL plus paired-bootstrap analysis. No feature-flag service. | P1 |
| Exposure logging | partial | A cache hit returns the cached `request_id` and `recommendation_id`s, and no new rows are written (`services/recommend.py:102-104`). Impressions are therefore under-counted, and feedback attaches to an old exposure. | — | Log exposure on every served response (a slim row or event), including cache hits | P1 |
| Guardrails / stopping rules | missing | — | Bootstrap CIs in drift eval | Fixed-horizon analysis, SRM check, guardrail metrics (negative feedback rate, latency) | P2 |

## 6. Second domain

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Generic CSV + YAML adapter | working | `domains/generic/*`, 7 synthetic tests (`tests/domains/test_generic_adapter.py`), leak-free test at `:207` | Yes | — | — |
| us-unemployment demo quality | partial | Forecasts are no better than naive. Warnings lag turning points. Current FRED vintage only (`docs/platform.md:498-504`). | Config, download script | Compare against the Sahm-rule baseline. Add weekly initial claims (ICSA) as a leading series. Use ALFRED vintages for honest replays. | P1 |
| A "serious" second domain | missing | Only one real non-movie dataset. The generic adapter adds no domain stages. | Generic adapter, weekly series support (`docs/platform.md:419`) | CDC ILINet (FluView, weekly, public domain), which has an official epidemic baseline as ground truth for warnings and strong seasonality (a real forecasting test). Implement through YAML plus a small adapter hook for the seasonal baseline. | P1 |
| Domain data in Docker | missing | The compose trainer never runs `download_domain_data.py` (`docker-compose.yml:116-123`), so the generic domain is unavailable in a fresh stack | Script exists | Add it to the trainer command | P2 |

## 7. Backend / API

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Routers, auth, CSRF, error envelope | working | `main.py:137-153`; `tests/integration/test_security*.py` | Yes | — | — |
| Run lock | partial | `threading.Lock` per process (`services/intel.py:450`). SECURITY.md:256 lists this as a residual risk. | — | A Postgres advisory lock (`pg_try_advisory_lock`) with the thread lock kept for SQLite | P1 |
| Metrics | partial | In-process JSON registry, reset on restart, one worker's view (`metrics.py:1-5`). No Prometheus exposition. | Registry API | `/metrics` in Prometheus text format from the same registry (no new dependency), or `prometheus_client` with multiprocess mode | P1 |
| Monitoring / alerting | missing | No SLOs or alert rules. `/health` checks the DB and cache, and `/health/ml` checks the engine (`routers/health.py:15-37`). Nothing watches model quality or run freshness over time. | `/admin/metrics` | Readiness vs liveness split, model-quality monitors (served-confidence drift, negative-feedback rate) feeding the rollback gate | P1 |
| Auto-migrate on startup | partial | `auto_migrate=True` by default (`config.py:47`, `main.py:62-63`). Several workers would race. | — | A migration job in compose (`api` command `alembic upgrade head` once) and `auto_migrate=false` in production | P2 |
| Retention | missing | Runs (about 0.5 MB JSON each), audit rows and recommendation rows are never pruned (SECURITY.md:258) | — | A retention script plus a scheduled job | P2 |
| Hand-maintained enum duplication | partial | Allowed values live in `models.py:287-312` **and** as string copies in each migration (`0005_domains.py:24-31`) **and** in `services/recommend.py:36` | — | A single `enums.py`. Migrations keep literal copies (correct), but a test compares the migrated DB CHECK text with the models. | P1 |

## 8. Frontend

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Pages and console | working | 34 pages. tsc, eslint and 95 Vitest tests are clean. | Yes | — | — |
| API types | partial | 1,157 hand-written lines in `src/lib/intel-types.ts` plus `types.ts`. No OpenAPI codegen, so types drift silently. | OpenAPI is served in development | Generate types from `/openapi.json` in CI (`openapi-typescript`, a dev dependency only) and diff them | P1 |
| Error/loading boundaries | missing | No `error.tsx`, `loading.tsx` or `not-found.tsx` under `src/app` (find found none) | `components/jev/states.tsx` | Add route-segment boundaries for `/intel`, `/me` and the root | P1 |
| E2E in CI | missing | Only `scripts/capture_screenshots.py` (local Playwright) | Playwright is already a dev dependency | A smoke E2E job against `docker compose` | P2 |
| Replay UX | partial | The console shows the latest run, which after a replay is the replay (see §2) | — | A "Replay" badge and a separate replay view once runs carry a `mode` | P1 |
| New surfaces for Phase 2 | missing | No pages for experiments (online), model promotion or rollback, event ingestion health | Admin pages | New pages under `/admin/*` and `/intel/*` | P1 |

## 9. Security

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Passwords, JWT, CSRF, CSP, headers | working | SECURITY.md; `security.py`; tests | Yes | — | — |
| Token revocation / logout | missing | There is no `jti` and no denylist. Logout only clears the cookie. A JWT lasts 24 h (`security.py:31-43`, `config.py:26`). | Redis cache | A `jti` plus a Redis denylist on logout and demotion, or a `token_version` column checked each request (the user is already re-read every request) | P1 |
| Password reset / email verification | missing | SECURITY.md:210 | — | Out of scope unless the product needs it. At minimum, admin-initiated reset. | P2 |
| Rate limiting behind the proxy | partial | All clients share the web container's IP bucket (progress.md §10). 20 logins/min is **global**, so one client can lock everyone out. | `client_address` | An edge proxy that overwrites XFF, or set XFF in `proxy.ts` from the socket peer. Add a per-account login throttle. | P1 |
| Admin bootstrap | partial | `ensure_admin` promotes an *existing* account with `JEV_ADMIN_EMAIL` to admin without resetting its password (`services/sync.py:249-251`). Whoever registered that email first becomes admin. | — | Refuse to promote and log an error, or require the configured password to match | P1 |
| Service authentication for ingestion | missing | Only user JWTs exist | — | Scoped service tokens (hashed in the DB) for `/events:batch` and the scheduler | P1 |
| Supply-chain checks in CI | missing | `pip-audit` and `pnpm audit` were run by hand (SECURITY.md:238). There is no Dependabot or CodeQL. | — | Add both audits as CI steps and Dependabot config | P2 |
| Default `env` | partial | `Settings.env` defaults to `development` (`config.py:22`), which exposes `/docs`. Compose defaults to production. | — | Default to production and opt in to development | P2 |

## 10. Testing / CI / release

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| PostgreSQL in CI | missing | 2 tests skip without `JEV_TEST_POSTGRES_URL`. The CI job has no `services: postgres`. | Existing tests | A `postgres:17` service plus the env var | P0 |
| Real-data tests in CI | missing | 30 tests skip in CI (data gitignored) | Download scripts are checksum-verified | A cached data job: download MovieLens-small (MD5) and the FRED CSVs, run `train --quick`, then run the real-data tests. Nightly if too slow. | P1 |
| Migration gate | partial | Round trips exist for 0004 and 0005 on SQLite. There is no check for a single head or for parity between models and migrations. | `test_migration_0005_roundtrip_sqlite` | CI: `alembic heads` has exactly one head, `upgrade head → downgrade base → upgrade head` on SQLite **and** PostgreSQL, and autogenerate produces an empty diff | P0 |
| Acceptance test in CI | missing | `scripts/acceptance_test.py` runs by hand only | Script | A compose job (nightly or on tag) | P2 |
| Evaluation as a gate | missing | Evaluation reports are generated by hand into gitignored `experiments/` | Evaluators | Commit small `report.json` summaries to `docs/eval/` or attach them to releases. Gate on thresholds. | P1 |
| Release | partial | No git tags (`git tag -l` is empty). The version is duplicated in 4 places (`pyproject.toml`, `jev_api/__init__.py:3`, `jev_ml/__init__.py:3`, `frontend/package.json` plus `version.ts`). | Vitest version check | Tag releases. One version source, checked by a test. | P2 |

## 11. Docs and positioning

| Component | Status | Evidence | Reuse possible | Required change | Pri |
|---|---|---|---|---|---|
| Architecture docs | partial | ARCHITECTURE.md:22 still shows `jev_ml.intel.run_pipeline` as the entry point. The data model (ARCHITECTURE.md:100-110) omits v1.2 (`domain` column, `user_intel_feedback`, audit, evidence tables): grep finds 0 mentions. | platform.md is accurate | Regenerate the architecture map (see PHASE2_ARCHITECTURE_AUDIT.md §1) | P1 |
| Stale counts and claims | broken | README test counts (174/65 vs 279/95). Registry 0600 claims (see §4). roadmap.md "Next steps" dates from 1.1. | — | Correct the numbers. Generate counts in CI or drop them. | P1 |
| GitHub positioning | broken | The repo description is "Hybrid movie recommendation engine…" and the topics are all recommender (`gh repo view`). `pyproject.toml` description is "Intelligent Hybrid Recommendation Engine". The README title is the decision engine. | README | Update the description, topics (`decision-intelligence`, `early-warning`, `anomaly-detection`, `forecasting`) and pyproject description | P1 |
| Doc volume | partial | 3,380 lines across 15 docs. intelligence.md (759) and platform.md (522) overlap, and progress.md holds three release reports. | — | A one-page "How JEV works" plus a reference split. Move release reports to CHANGELOG or `docs/releases/`. | P2 |
