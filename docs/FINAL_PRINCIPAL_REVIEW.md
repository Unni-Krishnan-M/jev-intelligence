# Final principal review (v1.3.0)

_2026-09-25. A hostile read of the whole repository after all Phase-2 workstreams, the evaluation and validity review
and the release fixes. Scope: code under `ml/`, `backend/`, `scripts/`, `tests/`, the compose files and every doc.
The frontend was only read, not edited; frontend findings are handed to its owner (§4)._

Severity: **Critical** (wrong results or a security hole in the default path), **High** (a claim or behaviour a user
would rely on is wrong), **Medium** (a real gap with a workaround or disclosed), **Low** (hygiene).

## 1. Searches run

| Search | Command / method | Result |
|---|---|---|
| TODO, FIXME, XXX, HACK | `grep -rnE "TODO\|FIXME\|XXX\|HACK" backend ml scripts frontend/src` | none |
| Placeholders, mocks, fakes in production code | `grep -rn "placeholder\|mock\|dummy\|fake\|random\." backend/jev_api ml/jev_ml` | two stale comments ("WS placeholders", fixed); a padding "dummy" unit in `core/drift.py` (legitimate); seeded RNGs only in statistics code |
| Hard-coded metrics | the review's literal grep (§5 of the review), re-checked | none presented as computed; `EVALUATED_EFFECTS` (`domains/movie/user_intel.py`) is a cited copy, now pinned by a test |
| Dead or duplicate code | every module under `ml/jev_ml` and `backend/jev_api` checked for importers | six `jev_ml.intel` shim modules have no importer in the repo (L4) |
| Unsupported claims | every README and doc number traced to a stored run or the review | fixed in place (§3); three numbers without an artefact are now labelled |
| Reproducibility | clean-copy test run; commands in README | clean copy green; reports need regenerating (M8) |

## 2. Findings

| # | Sev. | Finding | Status |
|---|---|---|---|
| H1 | High | CTA warning skill was presented as "informative"; the lift CI includes 1, the untouched window gives 1.10, and the COVID result was worded as a forecast | **fixed** (docs: case study §4.2, §5.2, §5.4, §6; audit §4.2; progress; README) |
| H2 | High | The promotion gate was underpowered: an equivalent candidate passed the NDCG gate ≈ 36 % of the time; quick mode used B = 300 | **fixed** (`gates.py` gate-1.1.0: power-derived margins 0.010 / 0.011 / 0.006, B = 2000 in quick mode, recorded SE and power; tests; docs) |
| H3 | High | README headline (+21 %) came from the leaky per-user protocol; README counts and limitations were stale (174 tests; "never auto-resolved"; "feedback not fed back"; change points 7 %; precision 0.36) | **fixed** (README rewritten; counts from the final run) |
| H4 | High | The user's dev database `jev.db` was stamped at head but lacked the Phase-2 tables and columns (two from 0005, `users.token_version` from 0010), so Phase-2 endpoints would fail on it while `alembic upgrade` did nothing | **fixed** (backup, `scripts/repair_dev_db.py`, `alembic check` clean; completion report §8) |
| H5 | High (frontend) | `frontend/src/lib/api.ts` reads `{detail: {message, blockers}}` from a 409; the API now returns `{detail: "<string>", blockers: [...]}` | **handed to frontend** (§4) |
| M1 | Medium | Benchmark stars followed the unadjusted CI, not Holm (global "ALS retuned" was starred at p_holm 0.068) | **fixed** (`benchmark_report.significant`; stored reports regenerated; ML_IMPROVEMENT_REPORT) |
| M2 | Medium | `evaluate_domains.py` stored no per-unit warning rows, so no CI or re-analysis was possible | **fixed** (`warning_replay(keep_rows=True)`, `lift_uncertainty`; test). The stored official run predates it, so the unemployment lift has no CI yet |
| M3 | Medium | Drift precision 0.92 stated without its 1:1 synthetic prevalence; adaptation CI on 7 users quoted as an effect; two extra variants undisclosed | **fixed** (platform.md §7, progress, README) |
| M4 | Medium | "Well calibrated (ECE ≤ 0.0014)" at base rates near 1 %; AUC bars "cleared" by point estimates only | **fixed** (ML_IMPROVEMENT_REPORT §4, progress, README) |
| M5 | Medium | A/B replay: "+23 %" without CI, latency guardrail claimed on a replay, between-arm design | **fixed** wording (EXPERIMENTATION §7); paired replay mode **open** (roadmap 2) |
| M6 | Medium | The gate fitted content features on all MovieLens tags (test-period leakage, symmetric) | **fixed** (leak-free tag rebuild when the raw tags of the same dataset exist; recorded in `gate.json`) |
| M7 | Medium | The calibration gate's ECE margin (0.01) cannot fail at ~1 % base rates, and each model is calibrated on its own test set | **open**, documented (governance §8, roadmap 5) |
| M8 | Medium | `experiments/` and `models/` are gitignored, so every reported number must be regenerated after a clone (~25 min of CPU); only quick reproductions were re-run this phase | **open**, disclosed (README, matrix gate 7) |
| M9 | Medium | Docker acceptance and GitHub CI were not run on this tree; PostgreSQL-only tests skipped here | **open**, disclosed |
| M10 | Medium | API contract gaps: naive datetimes on SQLite for jobs, snapshots, governed models and experiments; rollback reason optional in the API and CLI; object-valued `detail` on 409; file-mirrored models looked like jobs | **fixed** (`UtcDatetime`, required reason, `GovernanceHTTPError`, `source`; tests) |
| M11 | Medium | Stale docs: auto-resolution "never" (deployment.md, intelligence.md), retrain via "activate" (deployment.md), tuning-window numbers without an artefact | **fixed** / labelled |
| L1 | Low | A movie replay reads the active model; determinism was claimed without that condition | **fixed** (run records `model_version`; Python pinning; docs); API pinning **open** |
| L2 | Low | `EVALUATED_EFFECTS` hand-copied from the drift report | **fixed** (pinned by `test_evaluated_effects_match_the_stored_drift_report`; skips without the report) |
| L3 | Low | Public `/health/ml` echoed the model-load exception (F6) | **fixed** (fixed message; test) |
| L4 | Low | `jev_ml.intel.{actions,common,lapse,modelstats,signals,warnings}` have no importer in the repo; the rest of `jev_ml.intel` is used by `scripts/evaluate_intelligence.py` and tests. All are re-export shims (233 lines in total) for v1.1 import paths | **open**: kept as the documented compatibility surface; deprecate in 2.0 |
| L5 | Low | Test-order dependence: `tests/integration/test_api.py::test_admin_endpoints` fails if the governance API tests run before it in one session (a gated candidate becomes promotable). The default collection order passes | **open** |
| L6 | Low | While regenerating benchmark reports, `experiments/rec-benchmark-20260924T171737Z-quick/REPORT.md` (an earlier results schema) was truncated and rebuilt; its calibration section now points to `results.json` | **fixed** (renderer tolerates the old schema); disclosed |
| L7 | Low | The only new candidate in the registry (`jev-20260924T174707Z-0298b516`) was rejected under gate-1.0.0 and was not re-gated under 1.1.0 | **open**, disclosed (governance §3) |
| L8 | Low | The PostgreSQL event watermark can miss in-flight commits; per-process run lock and refresher; unpruned `idempotency_keys` / `revoked_tokens` | **open**, disclosed (streaming §8, security §6) |
| L9 | Low | Stale "WS placeholders" comments in `models/__init__.py` and `schemas/__init__.py` | **fixed** |

No Critical finding. After this pass no High finding remains inside the backend, ML, scripts or docs; H5 needs a
frontend change.

## 3. Evaluation-review corrections applied

| Ref | Where applied |
|---|---|
| R1 | README evaluation table and protocol notes (the old single-protocol table was removed; every popularity comparator is labelled default or tuned); ML_IMPROVEMENT_REPORT header caveat |
| R2 | SECOND_DOMAIN_CASE_STUDY §5.2 and §6; progress (WS4b row); INTELLIGENCE_ENGINE_AUDIT §4.2 and stage table; README |
| R3 | SECOND_DOMAIN_CASE_STUDY §5.4 (two-change attribution, development window, 0.147 corrected, tuning runs labelled unsaved) |
| R4 | ML_IMPROVEMENT_REPORT §3 |
| R5 | platform.md §7(a), §7(b) (plus the two extra variants); progress; README |
| R6 | ML_IMPROVEMENT_REPORT §4; progress (three places); README |
| R7 | progress (WS2 row); RETRAINING_AND_MODEL_GOVERNANCE §3 and §8, and the code change (H2) |
| R8 | ML_IMPROVEMENT_REPORT §2 and §4; regenerated `REPORT.md` files |
| R9 | EXPERIMENTATION §7 |
| R10 | STREAMING_ARCHITECTURE opening paragraph, §4.1, §8; EARLY_WARNING_SYSTEM §3 |
| R11 | README (movie forecast coverage "over-covers") |
| R12 | README Limitations rewritten; unemployment and change-point numbers from the current official run (174851Z) with lift |
| R13 | SECOND_DOMAIN_CASE_STUDY §4.2, §5.3, §6; progress |
| Code issues in the review | gates.py tags (M6) and B (H2) and margins (H2); ingest.py model version (L1); benchmark_report stars (M1); evaluate_domains rows and CI (M2); `EVALUATED_EFFECTS` (L2) — all fixed; `simulate_ab_replay.py` paired mode — open |

## 4. Frontend items (owner: frontend engineer)

1. **409 shape.** Promote (`POST /governance/models/{v}/promote`) and legacy activate (`POST /models/{id}/activate`)
   now return `{"detail": "promotion blocked by the gate", "blockers": ["..."], "request_id": "..."}`. Read
   `body.blockers` (top level), not `body.detail.blockers` or `body.detail.message`.
2. `GET /governance/models` items carry `source: "job" | "artifact"`; link `job_id` only when `source == "job"`.
3. `POST /governance/models/rollback` requires `reason` (3–500 non-blank characters, 422 otherwise).
4. Datetimes on jobs, snapshots, governed models and experiments now always end in `+00:00`.
5. Version bump to 1.3.0 in `frontend/src/lib/version.ts` and `frontend/package.json`.
6. Not verified in this pass: `next typegen`, `tsc`, lint and build (only `pnpm test`: 129 passed).
