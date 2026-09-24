# JEV Phase 2 architecture audit (Phase 0)

_Audit date: 2026-09-24. Commit `997f2ad` (v1.2.0). This is a read-only audit; no code was changed._

The gap-by-gap evidence is in [PHASE2_GAP_MATRIX.md](PHASE2_GAP_MATRIX.md). This document covers:
- the architecture as it really is (§1);
- technical debt and duplication (§2 and §3);
- the traps (§4);
- the Phase 2 roadmap and its acceptance criteria (§5 and §6);
- how 5–6 engineers can work in parallel without touching the same files (§7);
- where each gap is closed without new infrastructure (§8).

## 1. Architecture map (as built, verified against code)

```
Browser ── Next.js 16 (frontend/src; 34 pages; SWR; hand-written types in lib/*types.ts)
            proxy.ts: per-request CSP nonce + cookie-presence guard; /api/* rewrite
                 │ same-origin, httpOnly JWT cookie + X-JEV-CSRF
FastAPI (backend/jev_api, one uvicorn process, per-process state)
  main.py  middleware: server request id · rate limit (cache.incr_window) · security headers · JSON logs
  routers: auth · users · me (/me/intelligence*) · movies (+ rate/favorite/watch writes) ·
           recommendations · admin (/models, /experiments = OFFLINE runs, /admin/metrics|audit) ·
           intel (/intel/* operator console, 30 routes) · health
  services:
    profile.py      DB rows → UserState → jev_ml UserProfile          (event time = Rating.updated_at)
    recommend.py    strategy decision → cache key → engine.recommend → persist rows → cache
    user_intel.py   recommendation_strategy decision (CACHED ONLY, never persisted)
    feedback.py     upsert on partial unique indexes (0004)
    intel.py        IntelService: domains registry, gather inputs (MovieLens files + app DB),
                    threading.Lock run, persist run/decisions/normalised rows/evidence, warning lifecycle,
                    scenarios, startup refresh thread
    ml.py           EngineHolder (per-process active engine; load-before-swap)
    sync.py         files → DB mirror (catalogue, model_versions, experiments, intel evaluations), ensure_admin
    audit.py        audit_logs writer (redaction, never fails the request)
  metrics.py        in-process counters/summaries (reset on restart)
  models.py         ALL ORM tables + ALL enum tuples feeding CHECK constraints (787 lines)
  migrations 0001 → 0002 → 0003 → 0004 → 0005 (linear, single head)
        │                               │                               │
   PostgreSQL 17 / SQLite           Redis 7 / in-memory              files: data/ models/ experiments/
                                    (rec + strategy cache, rate limits)  (gitignored; source of truth for models)
jev_ml (pure Python, no DB imports)
  recommender: data/ · features.py · models/{popularity,content,itemknn,als,hybrid} · explain.py ·
               evaluation/{split,metrics,evaluator,report} · training.py · calibration.py · registry.py ·
               engine.py (inference only) · signals.py (interaction weights)
  core/ (domain-independent): types · adapter · config · series · quality · trends · anomalies · forecast ·
               scenario · risk · decisions · batches · early_warning · warnings · actions · signals · drift ·
               evaluation · pipeline.run_domain
  domains/movie/: ingest · series · raters · lapse · modelstats · risks · decisions · actions · signals ·
               evaluation · user_intel · user_scenario · adapter (MovieAdapter)
  domains/generic/: config (YAML schema) · adapter (GenericAdapter)   configs/domains/us-unemployment.yaml
  intel/ (v1.1 compatibility shim: re-exports + run_pipeline = run_domain(MovieAdapter))
scripts/ (offline entry points): download_* · preprocess · train_models · calibrate · evaluate_{models,
         intelligence,domains,drift} · run_intelligence · acceptance_test · capture_screenshots · dev.mjs
Ops: docker-compose (db, cache, api [expose only], web [fixed IP], trainer [manual profile]) · CI (ruff,
     format, mypy, pytest on SQLite without data; tsc, eslint, vitest, build)
```

**Data flows that matter for Phase 2**
1. *Member write*: `POST /movies/{id}/rate` upserts a `ratings` row in place (no history) and bumps
   `profile_version`. The next `GET /recommendations` rebuilds the profile from the rows.
2. *Serving*: the strategy decision (cached) produces a cache key; on a miss the engine ranks, and every
   item is persisted to `recommendations` (model version, decision id, confidence). A cache hit writes
   nothing.
3. *Intelligence run*: admin, startup, or a script → `IntelService.run`, which takes the per-process lock
   → gathers MovieLens files, the app DB and suppressions → `run_domain` → one transaction writes the run,
   decisions, normalised rows and evidence, then upserts warnings **for every run, replay or not**.
4. *Training*: the manual CLI or the compose trainer reads `data/processed/*.csv` only → artifacts in
   `models/<v>/` → `registry.json` (active by default) → the API mirrors it at startup. The API never sees
   a new model until restart or activation.
5. *Evaluation*: separate scripts write `experiments/<prefix>-<ts>/report.json`, and the API globs them.
   These are not the code path that serves (no DB inputs, no suppressions).

## 2. Technical debt (ranked)

1. **Replays share live state.** The replay path writes to the live warning lifecycle, becomes "latest",
   and suppresses the startup refresh. App-event stages use a `now` clock inside an `as_of` replay. Evidence:
   matrix §2, P0.
2. **No event log.** Interaction tables are mutable projections without event ids, idempotency or event
   time. Everything Phase 2 wants (feedback → retraining, A/B analysis, replayable live signals) needs an
   append-only history first.
3. **Governance is file-based and unconditional.** A new model activates by default. There is no promotion
   gate, no candidate state, no automatic rollback, and a model swap in one process does not propagate.
4. **Enum/CHECK drift.** Allowed values are duplicated across `models.py`, each migration and some services.
   `CONFIDENCE_KINDS` has already diverged (core 5, DB 4).
5. **Per-process singletons.** Run lock, metrics, engine and in-memory cache are all per process. They are
   fine for one uvicorn worker and wrong for more than one.
6. **Evaluation is out of band.** Four evaluator scripts with four report formats sit in a gitignored
   directory. CI never runs them, and no threshold gates a merge or a model.
7. **Lineage holes.** Member strategy decisions are not persisted. Runs have no config hash or input
   fingerprint. Recommendation rows lack the calibration version and the experiment arm.
8. **Frontend contract drift risk.** About 1,400 lines of hand-written API types, and no generated client.
9. **Monolithic shared files.** `models.py` (787 lines), `schemas.py` (637), `routers/intel.py` (1,226) and
   `services/intel.py` (1,110) are touched by nearly every feature: the main parallel-work bottleneck.
10. **Docs drift.** Test counts, registry mode and the ARCHITECTURE data model and entry point are stale.
    The GitHub description still says "movie recommender".

## 3. Duplication and dead or legacy code

| Item | Finding | Recommendation |
|---|---|---|
| `jev_ml/intel/*` (19 files, 264 lines) | Pure re-export shims, but still the **primary import path** for the backend (`services/intel.py:70-73`, `schemas.py`), `scripts/evaluate_intelligence.py` and 12 test modules. Worse, movie-domain modules import *back* through the shim (`domains/movie/user_intel.py:82`, `user_scenario.py:51`, and a lazy import in `risks.py:155` to dodge a cycle). | Point the backend and domain modules at `jev_ml.core` / `jev_ml.domains.movie`. Keep the shim for tests and external callers for one release, then delete it. |
| Three `signals` modules | `jev_ml/signals.py` (interaction weights), `core/signals.py` and `domains/movie/signals.py` (intelligence signals) | Rename the first to `interactions.py` |
| Three `evaluation` modules | `jev_ml/evaluation/` (recommender), `core/evaluation.py` (platform), `domains/movie/evaluation.py` (719 lines, v1.1 intel eval), plus 4 scripts | One `jev_ml/eval/` facade with a common report schema (`{suite, version, metrics, thresholds, passed}`) |
| Two run entry points | `run_pipeline(PipelineInputs)` (movie) vs `run_domain(adapter)` (generic), branched in `services/intel.py:648-654` and again in `scenario()` | Build `MovieAdapter` in the service. One code path. |
| Enum literals | `models.py:287-312`; migration copies (`0002:41-43`, `0005:24-31`); `services/recommend.py:36` | A single `enums.py`, plus a CI test that the migrated DB CHECK text equals the model's |
| Pipeline version | Movie `intel-1.1.0` (kept for goldens) vs `core-1.0.0` vs the adapter's `pipeline_version` | Keep, but document the triplet in the run lineage |
| Version string | 4+ copies | One source |

No TODO, FIXME, NotImplemented or mock markers exist in production code (grep over `backend`, `ml`, `scripts`
and `frontend/src`). Mocks live only in tests (`IntelService.inputs_factory` injection). No placeholder values
were found: `core/types.py:12` forbids them, and the stages emit `null` plus a reason.

## 4. Traps (read before editing)

1. **Audit `ck_audit_action` CHECK.** The list of allowed actions is hard-coded in the constraint
   (`models.py:297-307` and `0005_domains.py:24-31`). Every new audit action requires a migration that
   drops and recreates the constraint. On SQLite that is a `batch_alter_table`, which **rebuilds**
   `audit_logs`. The downgrade must also delete rows with the new actions first (see
   `0005_domains.py:128`). If five workstreams each add actions, you get five migrations fighting over one
   constraint. **Mitigation:** one integrator migration registers every Phase 2 action up front
   (§7.3).
2. **`ck_intel_decision_confidence_kind` lacks `evidence`.** Core allows it (`core/decisions.py:30`).
   Persisting such a decision fails the **entire** run transaction, and the run then records itself as
   failed with "persisting the result failed".
3. **`ck_rec_confidence_kind` allows only `probability`** (`models.py:165`, mirrored at
   `services/recommend.py:36`). Any new confidence type on recommendations needs a migration.
4. **Alembic is linear, with hand-numbered revisions (`0001`–`0005`).** Parallel branches will each create
   a revision `0006` with `down_revision="0004"`/`"0005"`, which gives multiple heads. `auto_migrate` then
   fails at startup (`main.py:62-63`). No CI check catches this today.
5. **Migrations must not import models.** Constants are copied on purpose. Keep that rule, and test for
   parity instead.
6. **Partial unique indexes use `sqlite_where` / `postgresql_where`** (`models.py:214`). New ones must set
   both, or SQLite tests pass while PostgreSQL behaves differently. PostgreSQL is not in CI.
7. **Golden tests** (`tests/core/test_movie_golden.py`) pin movie outputs, including `state_hash`. Any
   core change that alters a hashed decision state breaks them by design. Regenerate the goldens only
   together with a documented contract change.
8. **`latest_run` semantics** are used by roughly 20 read endpoints and by the startup refresh. Changing it
   (the P0 replay fix) touches all of them at once, so do it in one PR.
9. **Cache keys encode the model version, decision id and profile version.** A per-process
   `EngineHolder` swap plus Redis-shared caches can serve a mix of versions across workers.
10. **Real-data tests skip silently in CI.** A green CI run says nothing about the MovieLens and FRED
    paths.

## 5. Implementation roadmap and dependency graph

Phases are ordered by dependency, not by calendar. Where the graph allows, parallel lanes run at the same
time (see §7).

```
P2.0 Enablers (integrator) ─────────────────────────────────────────────┐
  split models.py/schemas.py into packages · enums.py + CHECK parity test ·
  pre-register audit actions + confidence kinds (migration 0006) · CI: postgres service,
  single-head + roundtrip gate · run.mode (live|replay) column                       │
        │                    │                     │                    │           │
        ▼                    ▼                     ▼                    ▼           ▼
P2.1 Events            P2.3 Recommender       P2.4 Decision-intel   P2.9 Security  P2.10 Eval harness
(idempotent log,       quality (global split, quality + replay     (revocation,   (common report
 projections, replay)   leakage tests, cold     isolation fix        svc tokens,    schema, thresholds,
        │               start, calibration)     (uses run.mode)      admin boot)    CI job)
        │                    │                     │                                   │
        ▼                    │                     ▼                                   │
P2.2 Feedback → retrain ◄────┘               P2.5 Second serious domain                │
 → gated promote/rollback                     (ILINet + upgraded unemployment)         │
        │                                                                              │
        ▼                                                                              │
P2.6 Online experimentation (needs events + exposure log + governance)                 │
        │                                                                              │
        ▼                                                                              ▼
P2.7 Frontend productization (types codegen, boundaries, new admin/intel pages) ──► P2.11 Verification
                                                                                        │
                                                                                        ▼
                                                                                   P2.12 Docs
```

Hard dependencies:
- P2.1 → P2.2 → P2.6.
- P2.0 → everything that migrates.
- P2.3 → P2.2, because the promotion gate reuses its metrics.
- P2.4 → P2.5.
- P2.10 feeds the gates in P2.2, P2.3, P2.4 and P2.5.
- P2.7 can start on types codegen and boundaries right after P2.0 and add pages as the APIs land.

## 6. Acceptance criteria per phase

Every criterion is a command, a test or a measured number. "Report" means a committed JSON summary under
`docs/eval/` (or a release asset) in the common schema, plus a CI job that recomputes it.

### P2.0 Enablers
- `alembic heads` prints exactly one head. `upgrade head → downgrade base → upgrade head` passes on SQLite
  **and** PostgreSQL 17 in CI.
- A test compares every `CheckConstraint` in the migrated DB with the enum tuples in `enums.py`, and they
  are equal.
- `ck_audit_action` contains every Phase 2 action. `ck_intel_decision_confidence_kind` contains
  `evidence`.
- `IntelRun.mode` exists (default `live`; `replay` whenever `requested_as_of` is set).
- The 2 PostgreSQL-only tests run in CI.

### P2.1 Events with idempotency and replay
- Table `events` is append-only. The application role has no UPDATE or DELETE grant on PostgreSQL. It has a
  unique `idempotency_key`, `event_id`, `occurred_at`, `received_at` and a monotonically increasing `seq`.
- Posting the same event (same `Idempotency-Key`) 10 times gives 1 row and 10 identical responses (tested).
  This covers rate, unrate, favorite, watch, feedback and `POST /events:batch`.
- Projections (`ratings`, `watch_history`, `favorites`, `recommendation_feedback`) are written in the same
  transaction as the event. A test rebuilds every projection from `events` and compares it row for row.
- Replay: folding events with `occurred_at ≤ T` and `received_at ≤ T` reproduces the profile as served at T
  (tested with interleaved re-rates).
- Ingest throughput is at least 500 events/s single-process on a laptop with SQLite, and batch ingest
  handles 10k events in under 30 s. Measured and reported.

### P2.2 Feedback → retraining → gated promotion and rollback
- `scripts/export_training_snapshot.py` writes a versioned snapshot (MovieLens plus app events ≤ cutoff)
  with a dataset version hash. Two runs on the same data give the same hash.
- The scheduler (a compose service or cron) runs retraining only when the `retrain_model` decision answers
  `yes`. The decision id is stored on the candidate manifest.
- New models register as `candidate`. `promote()` passes only if all of these hold:
  - NDCG@10 non-inferior to the incumbent (paired bootstrap lower bound ≥ −0.005) on the global temporal
    split;
  - calibration AUC ≥ incumbent − 0.01;
  - ECE ≤ 0.01;
  - no cold-start bucket worse than popularity (after P2.3).

  A failing gate leaves the incumbent active and writes an audit `model.promote` row with
  `passed=false`.
- Rollback: `POST /models/rollback` restores `previous` in one request. A post-promotion monitor breach
  (negative-feedback rate up by more than 3σ over 24 h) triggers an automatic rollback in an integration
  test with synthetic traffic.
- Every API worker serves the new version within 60 s of promotion (tested with 2 workers).

### P2.3 Recommendation quality
- Both split protocols are reported. The headline uses `global_temporal`. Leakage tests assert split
  ordering and zero test rows in any fit (calibration included).
- Cold start: for profile sizes 0, 1–3 and 4–10, the served strategy's NDCG@10 is ≥ popularity's. The
  bootstrap CI lower bound of the difference must be ≥ −0.002. Otherwise the router serves popularity for
  that bucket, and that choice is recorded in the recommendation row.
- Calibration: test AUC ≥ 0.65 for the full profile and ≥ 0.60 for short profiles, with ECE ≤ 0.01. The
  `calibration_version` is stored on every `recommendations` row.
- Warm NDCG@10 on the user-temporal split does not regress by more than 0.005 against 0.1214.

### P2.4 Decision-intelligence quality
- Replay isolation (P0): a replay run never creates, updates or reopens a live warning. `/intel/*` without
  `run_id` returns the latest **live** run. The startup refresh ignores replays. App-event stages in a
  replay are filtered at `as_of` or marked `skipped: not replayable`. There is one test per rule.
- The offline evaluator and the API use the same `RunContext` builder. A test runs both on the same inputs,
  and the decisions and warnings are identical.
- Generic warnings: report precision, recall, FPR, base rate and **lift**, next to a baseline rule (the Sahm
  rule for unemployment). Pass: lift ≥ 1.5 and precision ≥ baseline on the held-out replay window. The
  tuning and evaluation windows are disjoint.
- Forecasts: in each domain, the selected model's median MASE ≤ naive's, with a fallback to naive when it
  is not. 80 % interval coverage falls within [0.72, 0.88] on scored origins.
- Change points: false-alarm rate ≤ 2 % at nominal 1 % on the synthetic AR(1) study.
- Auto-resolution: a warning absent for N=3 consecutive live runs is resolved with a `system` event and an
  audit row. High and critical warnings need an operator.
- Member strategy decisions are persisted and resolvable by `decision_id` from any recommendation row.
- The golden tests pass, or are regenerated inside a PR that documents the contract change.

### P2.5 A second serious domain
- CDC ILINet (weekly, national plus HHS regions) is fetched by a checksum-verified download with
  provenance. It runs through the generic adapter plus, at most, one hook for a seasonal baseline.
- Leak-free replays over at least 5 seasons. Warnings are confirmed against CDC's published epidemic
  baseline crossing. Report lead time (median weeks before the crossing), precision and recall.
- The forecast beats seasonal-naive on median MASE, with 80 % coverage in [0.72, 0.88].
- us-unemployment: ALFRED real-time vintages are used for replays (or the limitation is stated in the
  report), and the Sahm-rule baseline appears in the report.
- The domain works in Docker from a fresh clone (the trainer downloads it).

### P2.6 Online experimentation
- Assignment is deterministic: sha256(salt ∥ user_id), stable across workers and restarts (tested). The
  arm is stored on each exposure row.
- Every served response logs an exposure, including cache hits (tested: a cache hit adds a row or event).
- The analysis script produces, per arm: exposures, users, CTR, like rate, negative rate, a
  bootstrap CI of the difference, a sample-ratio-mismatch (SRM) χ² test, and guardrails (p95 latency,
  negative-feedback rate).
- Start, stop and ramp are admin-only and audited. The experiment config is immutable once started, so a
  change means a new experiment.
- An A/A test on synthetic traffic shows no significant difference at α = 0.05 in at least 95 % of 200
  simulated runs.

### P2.7 Frontend productization
- TS types are generated from `/openapi.json`. CI fails when the committed types differ from the
  generated ones.
- `error.tsx`, `loading.tsx` and `not-found.tsx` exist for the root, `/intel` and `/me`, and are tested.
- New pages: model candidates and promotion history (with rollback), online experiments, event-ingest
  health, and a replay badge and separate replay view.
- A Playwright smoke job runs in CI against compose: login, recommendations, feedback, an intel run and a
  replay. It asserts 0 console errors and 0 CSP violations.
- Vitest stays at or above the current 95 tests, adding a test for each new component.

### P2.8 Security
- Token revocation: logout, password change and admin demotion invalidate existing JWTs (a `token_version`
  column or a `jti` denylist), tested.
- `ensure_admin` never promotes a pre-existing account with a different password (tested).
- Service tokens are hashed, scoped (`events:write`, `scheduler`) and revocable. They are the only way to
  reach `/events:batch`.
- A per-account login throttle exists in addition to the per-IP one. Behind compose, the client IP is
  attributed correctly (edge proxy or proxy.ts), tested with spoofed XFF.
- CI runs `pip-audit` and `pnpm audit --prod`, with no high or critical findings. Dependabot is enabled.
- Settings default to `env=production`.

### P2.9 Evaluation
- One common report schema across the recommender, calibration, intel, domains, drift, ILINet and A/B
  suites. Each report carries the code commit, config hash, data version and thresholds.
- A nightly CI job downloads the data (checksums), runs `train --quick` and every suite, and fails when a
  threshold is broken. The summaries are committed or attached.

### P2.10 Verification
- The full gate list passes on a clean clone:
  - ruff, format and mypy;
  - pytest on SQLite and on PostgreSQL;
  - the real-data suite (nightly);
  - Vitest, tsc, eslint and build;
  - Playwright smoke;
  - an acceptance test against compose;
  - migration round trips.
- The 30 real-data tests run at least nightly.

### P2.11 Docs
- ARCHITECTURE.md is regenerated from §1: the event log, governance and experiments are shown, and the
  data model covers every table.
- README numbers are generated or dropped. No stale claims (registry mode, test counts).
- The GitHub description, topics and pyproject description say "decision and early-warning engine".
- Each Phase 2 capability has a one-page doc: what it does, how to run it, how it is evaluated, and its
  limitations.

## 7. Recommended file-ownership plan (6 workstreams + 1 integrator)

### 7.1 Enabling refactor first (integrator, 1–2 days, blocks everyone)
Turn the two monoliths into packages that re-export everything, so no import changes elsewhere:
- `backend/jev_api/models.py` → `models/__init__.py` (re-exports), `models/base.py`, `models/enums.py`,
  `models/catalog.py`, `models/recs.py`, `models/intel.py`, `models/audit.py`, plus empty-by-default
  `models/events.py`, `models/governance.py` and `models/experiments.py` created by their owners.
- `backend/jev_api/schemas.py` → the same split under `schemas/`.
- `routers/__init__.py` gets an `ALL_ROUTERS` tuple. `main.py` iterates it, so adding a router is a
  one-line change in one file (`main.py:177-188` today).
- `config.py`: group settings into commented sections, one per workstream. Workstreams only append inside
  their own section.
- Migration **0006** (integrator): `run.mode`, `evidence` confidence kind, and every Phase 2 audit action
  pre-registered (for example `event.ingest`, `event.replay`, `model.register`, `model.promote`,
  `model.rollback`, `model.retrain`, `experiment.create`, `experiment.start`, `experiment.stop`,
  `warning.auto_resolve`, `auth.logout`, `token.revoke`, `service_token.create`).

### 7.2 Workstreams

| WS | Owner of | Must not edit (request via integrator) |
|---|---|---|
| **WS1 Events & ingestion** | new `backend/jev_api/models/events.py`, `services/events.py`, `routers/events.py`, `schemas/events.py`; the write paths in `routers/movies.py` and `services/feedback.py` (emit events); `services/profile.py` (read from projections plus event time); `ml/jev_ml/domains/movie/ingest.py` app-event section; `tests/events/` | `services/recommend.py` (WS5 calls `events.emit`) |
| **WS2 Retraining & governance** | `ml/jev_ml/training.py`, `registry.py`, new `ml/jev_ml/governance/` (gates, promote, rollback), `scripts/train_models.py`, new `scripts/export_training_snapshot.py`, `scripts/scheduler.py`; `backend/jev_api/services/ml.py`, `services/sync.py`, `routers/admin.py` (models endpoints), `models/governance.py`; compose `trainer`/`scheduler` services; `tests/governance/` | `ml/jev_ml/models/*`, `calibration.py` (WS3) |
| **WS3 Recommender quality** | `ml/jev_ml/models/*`, `evaluation/*`, `calibration.py`, `features.py`, `explain.py`, `engine.py`, `signals.py` (rename), `configs/experiment.yaml`, `scripts/{evaluate_models,calibrate_recommendations,generate_recommendations}.py`, `tests/unit/*`, `tests/ml/*` | the serving code in the backend |
| **WS4 Decision intelligence & domains** | `ml/jev_ml/core/*`, `ml/jev_ml/domains/**` (except the WS1 ingest section and WS5's `user_intel.py` serving hooks), `ml/jev_ml/intel/` (shim retirement), `configs/domains/*`, `scripts/{run_intelligence,evaluate_intelligence,evaluate_domains,evaluate_drift,download_domain_data}.py`; `backend/jev_api/services/intel.py`, `routers/intel.py`, `models/intel.py`, `schemas/intel.py`; `tests/{core,intel,drift,domains}/`, `tests/integration/test_intel*.py`, `test_platform_api.py` | `services/recommend.py` |
| **WS5 Serving & experimentation** | `backend/jev_api/services/recommend.py`, `services/user_intel.py`, `routers/recommendations.py`, `routers/me.py`, new `services/experiments.py`, `routers/experiments_online.py`, `models/experiments.py`, `models/recs.py`, `scripts/analyze_experiment.py`; `ml/jev_ml/domains/movie/user_intel.py` serving helpers (`engine_with_overrides`, `build_strategy_profile`) jointly with WS4 (WS5 owns the functions below `serve`, WS4 the drift and policy code) | `ml/jev_ml/models/*` |
| **WS6 Frontend & docs** | `frontend/**`, `docs/**` (except the PHASE2 audit files), README.md, ARCHITECTURE.md, CHANGELOG.md, `scripts/capture_screenshots.py` | backend code |
| **Integrator** (tech lead; also owns security and CI) | `main.py`, `config.py`, `deps.py`, `security.py`, `routers/auth.py`, `metrics.py`, `db.py`, `cache.py`, `logging_setup.py`, `models/{base,enums,audit}.py`, `models/__init__.py`, `schemas/__init__.py`, **the Alembic chain** (merges), `services/audit.py`, `.github/workflows/*`, `docker/*`, `docker-compose.yml` (except the WS2 services), `pyproject.toml`, `uv.lock`, `SECURITY.md`, `tests/conftest.py`, `tests/integration/test_security*.py` | — |

With five engineers, merge WS6 into the integrator's scope, or split WS4 into core quality (WS4a) and the
second domain (WS4b). WS4b then owns `domains/generic/**`, the new ILINet config and adapter hook, and
`download_domain_data.py`.

### 7.3 Shared files: how to coordinate

| Shared file | Rule |
|---|---|
| `models.py` (becomes a package) | After §7.1 each WS owns its own submodule. `models/enums.py` and `models/__init__.py` are integrator-only. New enum values go through a PR to the integrator, bundled with the migration. |
| `schemas.py` | The same split. Shared response envelopes (`IntelPage`, error body) are integrator-only. |
| `main.py` | Frozen after §7.1. Routers register through `routers/__init__.ALL_ROUTERS` (one line per WS). Middleware changes are integrator-only. |
| `config.py` | Append-only inside a WS section. Settings names are prefixed per WS (`events_*`, `governance_*`, `experiments_*`). Do not reorder. |
| `metrics.py` | No edits needed: the registry takes arbitrary names. Namespaces: `events.*`, `governance.*`, `experiments.*`, `intel_*` (existing). Prometheus exposition is integrator work. |
| `services/recommend.py` | Owned by WS5. WS1 contributes `events.emit(...)`, WS2 the model-version resolver, WS3 the confidence and calibration version. All of these are **functions WS5 calls**, not edits to the file. The call sites are agreed in a short interface PR first. |
| Audit `ck_audit_action` | Pre-registered in 0006 (§7.1). A late new action is one integrator migration per sprint, batched, never one per WS. The downgrade deletes rows with the removed actions first. |
| Alembic chain | Only the integrator merges migrations. A WS opens its migration with a placeholder `down_revision` and a descriptive slug (`0007_events`). At merge, the integrator renumbers it to the next number and sets `down_revision` to the current head. CI enforces a single head, round trips on SQLite and PostgreSQL, and the CHECK parity test. Suggested order: 0006 enablers → 0007 events (WS1) → 0008 governance (WS2) → 0009 intel lineage and member decisions (WS4) → 0010 experiments and rec lineage columns (WS5). |
| `tests/conftest.py` | Integrator-only. A WS adds fixtures in `tests/<ws>/conftest.py`. |
| `routers/intel.py` / `services/intel.py` (1.1–1.2k lines) | WS4 only. The replay-isolation fix changes `latest_run` everywhere, in a single PR. |
| Frontend `lib/intel-types.ts` | Replaced by generated types (WS6). Backend WSs must keep `/openapi.json` accurate: `response_model` on every new route. |

## 8. Closing gaps without new infrastructure

| Gap | Minimal solution (chosen) | Rejected, and why |
|---|---|---|
| Event stream and ingestion | An append-only Postgres `events` table, unique idempotency key and `seq`, projections written in the same transaction. Consumers poll by `seq` (or use `LISTEN/NOTIFY` if latency matters). | Kafka or Redpanda: event volume is tiny (a few events/s), a single writer suffices, and it would add a broker to a "runs on any laptop" project. |
| Background and scheduled work | One `scheduler` compose service: a Python loop, a `pg_try_advisory_lock` per job, and an intel run plus retrain check on an interval. Host cron works too. | Celery, Airflow or Dagster: three jobs do not justify a workflow engine. |
| Run lock across workers | Postgres advisory lock, keeping the thread lock for SQLite | Redis Redlock: adds a failure mode for no gain |
| Multi-worker model swap | Workers compare the `registry.json` mtime (or a DB `active` row) per request and reload lazily | A pub/sub control plane |
| Online A/B | Hash assignment in code, a Postgres experiments table, arm on exposure rows, SQL plus a bootstrap script | A feature-flag SaaS, a stats service |
| Metrics and monitoring | Prometheus text exposition from the existing registry. Monitors computed by the scheduler from DB tables, which write warnings through the existing warning lifecycle (JEV monitors itself). | A new TSDB and Grafana stack as a requirement (optional compose profile at most) |
| Better cold start and ranking | Bucketed hybrid weights plus a popularity fallback, sklearn GBM or logistic calibrator/ranker | Deep sequence models (SASRec), which need data evidence first (roadmap already defers them) |
| Weak forecasts | Seasonal-naive, theta and a residual-quantile conformal band in numpy | Prophet or deep forecasters: heavy dependencies, and no evidence they beat the baselines on monthly or weekly aggregates |
| Drift recall | Omnibus combination and α tuning on the splice benchmark | Learned drift detectors |
| Decisions | Keep the typed, versioned rule policies | LLMs (explicitly out of scope, and they would break determinism and offline runs) |
| Architecture | Keep the modular monolith (one API, one ML package) | Microservices: the team is small, data is shared, and splitting would multiply the auth, migration and deploy surface |
| Token revocation | A `token_version` integer on `users` (the user is already loaded per request) | A session store or OAuth server |

## 9. Enabler — done

_Integrator, 2026-09-24. Structural and behaviour-preserving: `/openapi.json` and the route order are
byte-identical to v1.2.0, and the ORM DDL differs only by the three 0006 changes below._

### 9.1 Module layout: where each workstream adds code

`jev_api.models` and `jev_api.schemas` are packages. Their `__init__.py` re-exports every name the old
single modules had, so `from jev_api.models import X` and `from jev_api.schemas import X` still work.

| Module | Holds | Owner |
|---|---|---|
| `models/__init__.py`, `schemas/__init__.py` | re-exports, `__all__` | integrator |
| `models/base.py` | `Base`, `TimestampMixin`, `utcnow`, `in_`/`nullable_in` (CHECK text), `domain_column` | integrator |
| `models/enums.py` | **every CHECK vocabulary** (`AUDIT_ACTIONS`, `CONFIDENCE_KINDS`, `INTEL_RUN_MODES`, statuses, …) plus `CHECK_ENUMS`, the constraint → (table, column, values, nullable) registry the parity test reads | integrator |
| `models/users.py` | `User`, `UserGenrePreference` | integrator (security adds `token_version` here) |
| `models/catalog.py` | `Genre`, `Movie`, `MovieGenre` | shared, via integrator |
| `models/interactions.py` | `Rating`, `WatchHistory`, `Favorite` (future event projections) | WS1 |
| `models/recs.py` | `Recommendation`, `RecommendationFeedback`, `FEEDBACK_UNIQUE_INDEXES`, `UserIntelFeedback` | WS5 |
| `models/model_registry.py` | `ModelVersion`, offline `Experiment`, `EvaluationMetric` | WS2 |
| `models/intel.py` | every `Intel*` table, `run_mode()` | WS4 |
| `models/audit.py` | `AuditLog` | integrator |
| `models/events.py` | empty: the append-only event log | **WS1** |
| `models/governance.py` | empty: candidates, promotions, rollbacks, snapshots | **WS2** |
| `models/experiments.py` | empty: online experiments, arms | **WS5** |
| `models/security.py` | empty: service tokens, revocation | **integrator (security)** |
| `schemas/base.py` | `DbId`, `ORM` | integrator |
| `schemas/{users,movies,audit}.py` | auth/account, catalogue, audit schemas | integrator |
| `schemas/recs.py`, `schemas/me.py` | serving and member-intelligence schemas | WS5 |
| `schemas/model_registry.py` | model and offline experiment schemas | WS2 |
| `schemas/intel.py` | intelligence schemas (incl. `IntelPage`) | WS4 |
| `schemas/{events,governance,experiments,security}.py` | empty placeholders | WS1, WS2, WS5, security |

To add a table or schema, define it in your module, then append **one** import line in the package
`__init__.py` and its names to `__all__`. The package already imports the four placeholder modules, so
tables defined there are registered on `Base.metadata` (and seen by `alembic check`) with no other edit.
`schemas/**` has a ruff per-file ignore for RUF012: ruff cannot see that the cross-module `ORM` base is a
pydantic model.

### 9.2 Migrations and their owners

Single head, linear chain: `0001 → … → 0005 → 0006 → 0007 → 0008 → 0009 → 0010 (head)`.

| Revision | File | Owner | Content |
|---|---|---|---|
| 0006 | `0006_phase2_prep.py` | integrator | done (below) |
| 0007 | `0007_events.py` | WS1 | empty stub: events table, idempotency keys |
| 0008 | `0008_model_governance.py` | WS2 | empty stub: governance tables |
| 0009 | `0009_experiments.py` | WS5 | empty stub: online experiments, `recommendations.experiment_id/arm` |
| 0010 | `0010_security.py` | integrator (security) | empty stub: service tokens, `users.token_version` |

Each stub has `pass` bodies. Fill **your own stub in place**; never change `revision`/`down_revision` and
never add a new revision file yourself. WS4 has no stub. Its lineage columns and member decisions
(`config_hash`, `input_fingerprint`, `code_version`, persisted strategy decisions) get `0011` from the
integrator at merge time, as does any late migration. SQLite work goes through `op.batch_alter_table`,
and partial indexes set both `sqlite_where` and `postgresql_where`. Migrations never import the models:
copy constants in, and the parity test checks them.

**0006_phase2_prep:**
- `intel_runs.mode`: `live | replay`, NOT NULL, server default `live`, `ck_intel_run_mode`, and index
  `ix_intel_runs_domain_mode_started (domain, mode, status, started_at)`.
- **Backfill.** A run is a replay when it had an explicit `requested_as_of` **and**
  `COALESCE(as_of, requested_as_of)` is before the domain's data end. The data end is not stored, so it is
  inferred as the newest `as_of` of that domain's runs without an explicit as_of (those resolve to the data
  end). When no such run exists, the data end cannot be inferred and the run stays `live`. An explicit
  as_of at or after the data end also stays `live`.
- **New runs** record `mode = run_mode(requested_as_of)`: an explicit as_of means `replay`
  (`services/intel.py`, one line). Nothing reads the column yet.
- `ck_intel_decision_confidence_kind` gains `evidence` (trap 2). Every kind the core and domains emit
  (`probability`, `margin`, `rule`, `interval`, `evidence`) is now allowed.
- `ck_rec_confidence_kind` stays `probability`, the only kind the engine and calibration emit
  (`engine.py:177`, `calibration.py:513`).
- `ck_audit_action` gains every Phase 2 action (§9.3).
- The downgrade first deletes audit rows that use the new actions and decisions with `evidence`
  confidence, then narrows both constraints and drops `mode`. Runs are kept.
- Verified: upgrade → downgrade → upgrade on SQLite and on a throwaway `postgres:17`. `alembic check`
  shows no drift on either.

### 9.3 Audit actions (pre-registered, `enums.AUDIT_ACTIONS_PHASE2`)

| Owner | Actions |
|---|---|
| WS1 events | `events.ingest`, `events.replay` |
| WS2 governance | `model.register`\*, `model.retrain`, `model.promote` (log `detail.passed`; failing gates are audited too), `model.reject`, `model.rollback`, `dataset.snapshot` |
| WS4 intel | `warning.auto_resolve`\* |
| WS5 experiments | `experiment.create`, `experiment.start`, `experiment.ramp`\*, `experiment.pause`, `experiment.stop`, `experiment.conclude` |
| Security | `token.create`, `token.revoke`, `auth.logout`, `auth.revoke_all`, `auth.password_change`\*, `auth.login.throttled`\*, `user.role_change`\* |
| Operations | `retention.prune`\* |

\* Added beyond the brief because the roadmap needs them:
- candidate registration (P2.2);
- stale-warning auto-resolution (P2.4);
- experiment ramp (P2.6);
- revocation on password change or demotion, and the per-account throttle (P2.8);
- the retention job (§2).

`services/audit.record` already rejects unknown actions. A new action after this point is **one batched
integrator migration per sprint**, never one per workstream.

### 9.4 Router registry

`routers/__init__.py` defines `ALL_ROUTERS`, the single ordered tuple that `main.create_app` mounts.
`main.py` is frozen: to add a router, append one line to `ALL_ROUTERS` and its module to the import line in
`routers/__init__.py`. Order is match order, so append at the end.

### 9.5 Config sections

`config.Settings` is still one class, and every field name and env var is unchanged. Its fields are now
grouped under commented headers:
- core runtime (integrator);
- security;
- Phase 2 security hardening (`security_*`);
- recommendation serving (WS5);
- online experiments (WS5, `experiments_*`);
- intelligence (WS4, `intel_*`);
- events (WS1, `events_*`);
- governance (WS2, `governance_*`).

Append only inside your own section, and use your prefix. Never rename a field: the field name is the
environment variable (`JEV_<NAME>`).

### 9.6 Replay/live plumbing (for WS4)

- `IntelService.latest_run(db, status, domain, mode=None)` takes an optional `mode`. `None` keeps today's
  behaviour, and every caller still passes nothing.
- The P2.4 fix (latest means latest **live**, replays never touch warnings, the startup refresh ignores
  replays) is WS4's single PR. It switches the ~20 read paths and `_startup_refresh` to `mode="live"`.
- `run_out` does not expose `mode` yet. Adding it to `IntelRunOut` is WS4's call, together with WS6's
  replay badge.

### 9.7 CI guards

`tests/integration/test_migrations.py` checks:
- exactly one head and one base;
- a linear chain in which each file's name prefix equals its revision id;
- head → base → head on SQLite;
- that every CHECK constraint in a migrated database equals `enums.CHECK_ENUMS`, or is listed in
  `NON_ENUM_CHECKS`, and that the ORM matches too;
- that the core's `CONFIDENCE_KINDS` and `DECISION_KINDS` fit the CHECK;
- the 0006 backfill and downgrade.

Every test with `postgres` in its name skips unless `JEV_TEST_POSTGRES_URL` is set. The new CI job
`postgres` runs a `postgres:17` service and `pytest -k postgres`. It fails if any of those tests skips,
then runs `alembic check` on PostgreSQL. The `python` job also runs `alembic check` on SQLite. **Name every
new PostgreSQL-only test `*_postgres`.**

### 9.8 Rules for shared files

1. **Re-read before editing.** Other workstreams edit in parallel, so open the current file right before
   you change it, never from memory or an old copy.
2. **One-line appends only.** In shared files, add one line at the end of the relevant list or section:
   - `models/__init__.py`, `schemas/__init__.py`;
   - `routers/__init__.py`;
   - your section of `config.py`;
   - `enums.py`, through the integrator.

   Do not reorder, reformat or "tidy" what is already there.
3. **Never overwrite a shared file wholesale.** Do not use a Write or copy of a whole file, and do not
   regenerate one with a script. Use targeted edits only.
4. New enum values, audit actions, CHECK changes and migration numbers go through the integrator, bundled
   with the migration.
5. `main.py`, `models/enums.py`, `models/base.py`, the `__init__.py` files, `tests/conftest.py`,
   `.github/workflows/*` and the Alembic chain are integrator-owned.
