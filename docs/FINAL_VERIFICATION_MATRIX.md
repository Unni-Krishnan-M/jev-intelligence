# Final verification matrix (v1.3.0)

_2026-09-25. One row per Phase-2 quality gate. **PASS**: evidence gathered or re-checked in this final pass.
**PASS (prior)**: evidence from a workstream report that was not re-run here, cited. **PARTIAL**: holds with a stated
gap. Commands run from the repository root unless noted._

## Final runs (this pass)

| Run | Result |
|---|---|
| `uv run pytest -q -rs` (repository; real data and models present) | **466 passed, 8 skipped** in 78 s; all 8 skips are `JEV_TEST_POSTGRES_URL not set`; 474 collected |
| Clean copy: `git ls-files -co --exclude-standard -z \| tar --null -T - -cf - \| tar -xf - -C <tmp>`, then `uv sync --frozen && uv run pytest -q` | **452 passed, 22 skipped** in 73 s (the 8 PostgreSQL-only tests plus 14 real-data tests that skip without `data/`, `models/`, `experiments/`) |
| `uv run ruff check .` / `uv run ruff format --check .` / `uv run mypy` | all checks passed / 236 files formatted / no issues in 171 source files |
| `uv run alembic check` (against the repaired `jev.db`) / `uv run alembic heads` | no new upgrade operations / `0010 (head)`, single head |
| `cd frontend && pnpm test` | 129 tests in 10 files passed (not a frontend edit; run for the count) |

## Gates

| # | Gate | Status | Evidence |
|---|---|---|---|
| 1 | Forensic audit before changes | PASS (prior) | [PHASE2_GAP_MATRIX.md](PHASE2_GAP_MATRIX.md) (baseline 277 passed / 2 skipped at `997f2ad`), [PHASE2_ARCHITECTURE_AUDIT.md](PHASE2_ARCHITECTURE_AUDIT.md) |
| 2 | All tests green | PASS | final runs above (repository and clean copy); PostgreSQL-only tests were run by the workstreams (enabler: 289 on PostgreSQL 17; WS1 and WS2 reports) and not re-run here |
| 3 | No recommender regression | PASS | the active model is unchanged (`models/registry.json` → `jev-20260923T100141Z-bbb2e4c9`); `tests/ml/test_real_artifacts.py` passes on it; WS3 adopted neither candidate change ([ML_IMPROVEMENT_REPORT.md](ML_IMPROVEMENT_REPORT.md) §3, §4); the gate rejected the only new candidate |
| 4 | Tests for every new feature | PASS | events `tests/integration/test_events.py`, `test_events_ingest.py`; governance `tests/ml/test_governance.py`, `tests/integration/test_governance_api.py`, `test_governance_migration.py`; experiments `tests/integration/test_experiments.py`, `test_experiments_stats.py`; replay/lineage `tests/integration/test_intel_replay_lineage.py`; CTA `tests/domains/test_cta_domain.py`, `tests/core/test_forecast_seasonal.py`; leakage and statistics `tests/unit/test_leakage.py`, `test_stats_coldstart.py`; security `tests/integration/test_security_phase2.py`; this pass: `test_gate_margins_follow_the_power_rule`, `test_gate_records_its_power`, `test_warning_rows_and_month_cluster_lift_ci`, `tests/intel/test_replay_model_version.py`, `test_evaluated_effects_match_the_stored_drift_report`, `test_public_health_ml_hides_the_model_load_error`, governance API assertions for `source`, UTC offsets, rollback reason and the 409 shape |
| 5 | Migrations: chain, round trip, no drift | PASS | `alembic heads` single head 0010; `alembic check` clean; round trips and CHECK parity in `tests/integration/test_migrations.py` (SQLite; PostgreSQL variants skip here); the user's stale dev DB repaired with `scripts/repair_dev_db.py` and then `alembic check` clean (see [PROJECT_COMPLETION_REPORT.md](PROJECT_COMPLETION_REPORT.md) §8) |
| 6 | API validation and tests | PASS | strict request schemas (`extra="forbid"` in experiments), path patterns for versions and job ids, `RollbackRequest.reason` required, route walkers `test_every_intel_and_admin_route_requires_admin`, `test_admin_routes_reject_anonymous_and_members` |
| 7 | ML claims reproducible | PARTIAL | the reviewer re-ran the quick benchmark (identical metrics, config hash c97466e2b481) and the CTA warning evaluation (identical counts) ([EVALUATION_AND_VALIDITY_REVIEW.md](EVALUATION_AND_VALIDITY_REVIEW.md) §3). The full 11-minute benchmark was not re-run; `experiments/` is gitignored, so a clean clone must regenerate every report with the commands in the README. Three numbers have no stored artefact (CTA tuning-window FPRs, the calendar agreement figure, the calibrator µs) and are labelled as such in the docs |
| 8 | Evidence lineage | PASS | `test_every_decision_and_warning_has_a_lineage`, `test_replay_decision_lineage_and_auth`; every run records `pipeline_version`, `config_hash`, `input_fingerprint`, `event_watermark`, `model_version` |
| 9 | Warning lifecycle and audit | PASS | `test_auto_resolve_after_k_live_runs`, `test_auto_resolve_leaves_severe_warnings_to_an_operator`, `test_replay_never_touches_live_warnings`; audit rows in `tests/integration/test_intel_v11.py` |
| 10 | Streaming: replay and idempotency | PASS, with a scoped claim | `test_rate_with_idempotency_key_ten_times_is_one_event`, `test_replay_never_sees_late_arriving_events`, `test_same_as_of_gives_identical_decisions_via_api`; determinism is stated for a fixed code version, data files and active model (`tests/intel/test_replay_model_version.py`). No end-to-end test with events arriving between two replays |
| 11 | Retraining gates | PASS | `test_gate_passes_an_equivalent_model`, `test_gate_rejects_a_worse_model`, `test_gate_fails_a_broken_artifact`, `test_gate_is_stale_after_the_incumbent_changes`, power-rule tests; promotion, rollback and lock in `test_governance_api.py`; margins documented in [RETRAINING_AND_MODEL_GOVERNANCE.md](RETRAINING_AND_MODEL_GOVERNANCE.md) §3 |
| 12 | Experimentation: exposure and outcomes | PASS (offline only) | exposure logging including cache hits, attribution window, SRM, A/A false-positive rate in `test_experiments.py` and `test_experiments_stats.py`; demo `experiments/ab-replay-20260924T173730Z` (offline replay, not live traffic) |
| 13 | No unresolved critical or high security issue | PASS | [SECURITY_AUDIT_PHASE2.md](SECURITY_AUDIT_PHASE2.md): the one High (F1) fixed; F4 now fully wired (compose); F6 (Low) fixed in this pass; pip-audit and pnpm audit found nothing |
| 14 | Clean clone works | PASS | the clean-copy run above; `uv sync --frozen` succeeded with the updated `uv.lock` (1.3.0) |
| 15 | README matches behaviour | PASS | README rewritten in this pass; every number cites a stored run or the review's reproduction; test counts from the runs above; screenshots are files in `docs/screenshots/`. Docker acceptance (69 checks) is marked as last run for 1.2.0 |
| 16 | Independent review | PASS | [EVALUATION_AND_VALIDITY_REVIEW.md](EVALUATION_AND_VALIDITY_REVIEW.md) (R1–R13 applied, see [FINAL_PRINCIPAL_REVIEW.md](FINAL_PRINCIPAL_REVIEW.md) §3), [FINAL_PRINCIPAL_REVIEW.md](FINAL_PRINCIPAL_REVIEW.md) |

## Not verified in this pass

- Docker build and the compose acceptance test (user directive: no Docker builds). Last green: v1.2.0, 69/69.
- GitHub CI (nothing was pushed).
- PostgreSQL-only tests (no `JEV_TEST_POSTGRES_URL` here).
- Frontend typegen, tsc, lint and build (frontend owned by the frontend engineer; only `pnpm test` was run).
