# Project completion report: JEV Phase 2 (v1.3.0)

_2026-09-25. Final status is in §12. Every number cites a stored run, a test or the independent review
([EVALUATION_AND_VALIDITY_REVIEW.md](EVALUATION_AND_VALIDITY_REVIEW.md)); gates and evidence are in
[FINAL_VERIFICATION_MATRIX.md](FINAL_VERIFICATION_MATRIX.md), open findings in
[FINAL_PRINCIPAL_REVIEW.md](FINAL_PRINCIPAL_REVIEW.md)._

## 1. What existed before Phase 2 (v1.2.0, commit `997f2ad`)

- A domain-independent core (`ml/jev_ml/core`) with the movie adapter (hybrid recommender, raters, lapse, model
  governance decisions, preference drift, strategy decision) and a generic CSV + YAML adapter (US unemployment).
- Typed decisions with confidence kinds and abstention; early warnings gated by `early_warning_level` (`ewl-1.0.0`);
  a warning lifecycle without auto-resolution; scenarios, feedback, audit log.
- FastAPI + Alembic 0001–0005, Next.js console and member app, Docker Compose, CI.
- 277 pytest passed / 2 skipped, 95 Vitest ([PHASE2_GAP_MATRIX.md](PHASE2_GAP_MATRIX.md) §0).
- Known defects found by the audit: replays wrote live warnings and became the default "latest run"; replays mixed
  later app events into earlier analyses; member writes had no idempotency (a retry duplicated rows); training never
  saw app feedback; there was no promotion gate, online experiment or machine credential; JWTs were not revocable;
  the recommender was evaluated on one protocol with cross-user leakage and without CIs.

## 2. What was improved

| Area | Before | After | Evidence |
|---|---|---|---|
| Replay vs live | replays touched live warnings and reads | `intel_runs.mode`; replays isolated; reads default to live | `tests/integration/test_intel_replay_lineage.py` |
| Change points | 9.9 % false alarms at nominal 1 % | 1.3 % (φ from history) | INTELLIGENCE_ENGINE_AUDIT §4.4 |
| Unemployment warnings | precision 0.36, lift 1.22 | precision 0.415, lift 1.41 (development windows) | `platform-eval-20260924T174851Z` |
| Stale warnings | never closed | auto-resolved after K live runs (≤ medium by default) | `test_auto_resolve_after_k_live_runs` |
| Recommender evaluation | one leaky protocol, no CIs | two protocols, leakage checks, bootstrap CIs, Holm-adjusted paired tests | ML_EVALUATION, ML_IMPROVEMENT_REPORT |
| Security | irrevocable JWTs, admin-email squatting, shared login bucket | revocation, service tokens, per-account throttle, signed client address, supply-chain checks | SECURITY_AUDIT_PHASE2 |

## 3. What was newly implemented

- **Streaming** ([STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md)): append-only bitemporal `events` log;
  projections in the same transaction; `Idempotency-Key`; replay/diff of projections; pushed observations for
  generic domains; debounced refresher; event watermark per run. 730 events/s on SQLite.
- **Retraining and governance** ([RETRAINING_AND_MODEL_GOVERNANCE.md](RETRAINING_AND_MODEL_GOVERNANCE.md)): snapshots
  with app feedback, candidate training, the promotion gate, promote/rollback with hot swap and polling followers,
  a retrain lock, a decision-driven scheduler, lineage per version, `/governance/*` and `scripts/retrain.py`.
- **Experimentation** ([EXPERIMENTATION.md](EXPERIMENTATION.md)): `ab_*` tables, sticky salted-hash assignment,
  exposure logging including cache hits, attribution, SRM, Bonferroni, guardrails, power, persisted member decisions,
  an offline replay demo.
- **Second domain** ([SECOND_DOMAIN_CASE_STUDY.md](SECOND_DOMAIN_CASE_STUDY.md)): CTA daily ridership with daily and
  weekly series, seasonal forecasts, a holiday calendar and publication lag, configured in YAML.
- **Intelligence** ([INTELLIGENCE_ENGINE_AUDIT.md](INTELLIGENCE_ENGINE_AUDIT.md), [DECISION_ENGINE.md](DECISION_ENGINE.md),
  [EARLY_WARNING_SYSTEM.md](EARLY_WARNING_SYSTEM.md)): `ewl-1.1.0` (recovery-aware trends), decision and warning
  lineage, the `evidence` confidence kind.
- **This release pass**: power-derived gate margins (gate-1.1.0), leak-free tags in the gate, Holm-based stars,
  per-unit warning rows with month-cluster lift CIs, replay model-version recording and pinning, compose wiring of the
  proxy secret, API contract fixes (UTC offsets, required rollback reason, string `detail` with `blockers`, `source`
  label), the F6 health fix, the dev-DB repair script, and the documentation corrections R1–R13.

## 4. Architecture

See [../ARCHITECTURE.md](../ARCHITECTURE.md) and [INTELLIGENCE_PIPELINE.md](INTELLIGENCE_PIPELINE.md). One core, two
adapter kinds, one API process, one database; no broker, no external model service. Files remain the source of truth
for model artifacts (`models/registry.json`), the database for runs, lifecycle, events, jobs and experiments.
Migrations 0006–0010 add run modes, the event log, governance, experiments and security tables; single head 0010.

## 5. Results, with uncertainty

**Recommender** (`experiments/rec-benchmark-20260924T172521Z`):
- per-user temporal split (592 users; cross-user leakage): hybrid NDCG@10 0.1207 [0.1078, 0.1340], +0.020 over
  item-kNN (p_holm 0.001);
- leak-free global split (28 users): hybrid 0.1151 [0.059, 0.186]; vs item-kNN +0.024 [−0.025, +0.074] (n.s.); vs a
  tuned 90-day popularity baseline −0.0435 [−0.092, +0.0005] (n.s., negative);
- cold start (3 interactions): popularity 0.046 vs hybrid 0.035; tuned cold stages improve on the incumbent in 5 of 6
  cells (Holm-significant) but failed non-inferiority in one global bucket: not adopted;
- confidence: calibrated to the ~1 % base rate, Brier skill ≤ 0.008, AUC 0.55–0.67; the logistic calibrator (AUC
  0.673 [0.645, 0.696]) was not adopted.

**Early warnings** (monthly leak-free replays):
- us-unemployment (`platform-eval-20260924T174851Z`): precision 0.415 at base rate 0.295, lift 1.41, FPR 0.30, recall
  0.51; 2019–21 lift 1.04. No CI (rows not stored by that run; stored from v1.3.0 on).
- cta-ridership: domain-aware lift 1.44, month-cluster 95 % CI [0.52, 2.36], year-stratified 1.18 on the 2015–24
  development window; untouched 2003–09 window lift 1.10 [0.00, 2.33] (3 of 16 events); 2025–26: 0 of 3 events. FPR
  0.77 → 0.22 against the plain generic config. **No demonstrated warning skill.**
- movie: lapse precision uninformative (base rate 1.0); genre decline never fires.

**Forecasts:** movie MASE 0.93 vs naive 1.15 (21/21; intervals over-cover at 0.95); unemployment MASE 1.52 vs naive
1.41 (no better; coverage 0.61); CTA relative MAE 0.33 vs naive and 0.77 vs seasonal naive on 8/8 series.

**Drift:** detector precision 0.92 at a 1:1 synthetic prevalence (≈ 0.4–0.6 at 5–10 %), recall 0.29, FPR 0.026
[0.007, 0.089]; adaptation +0.006 on 7 users (sign test p = 0.25): no effect established, `standard` is served.

**Governance:** the real quick retrain was not shown non-inferior under gate-1.0.0 (Δ −0.0009, 90 % CI −0.0073..
+0.0055); the old margin gave an equivalent model a ≈ 36 % pass rate, the new power-derived margins ≈ 80 % per gate.

**Experimentation:** offline A/B replay on 610 members, 3 arms: inconclusive (recency +0.026 NDCG@10 [−0.009,
+0.061], p 0.092 vs adjusted α 0.025). Replay-only; no live traffic exists.

## 6. Security

One High (admin-email squatting at startup) and four Medium findings fixed by the security workstream; the last
Medium (shared login bucket) is now wired end to end (web signs, compose passes one secret); the Low F6 fixed in this
pass. pip-audit and pnpm audit: no findings. Residual risks: [SECURITY_AUDIT_PHASE2.md](SECURITY_AUDIT_PHASE2.md) §6.

## 7. Tests (final runs, 2026-09-25)

| Suite | Result |
|---|---|
| pytest, repository (real data present) | 466 passed, 8 skipped (PostgreSQL-only), 474 collected |
| pytest, clean copy of tracked + unignored files, `uv sync --frozen` | 452 passed, 22 skipped (8 PostgreSQL-only, 14 real-data) |
| ruff check / ruff format --check / mypy | clean / 236 files / no issues in 171 files |
| alembic check / heads | no drift / single head 0010 |
| Vitest | 129 passed in 10 files |

Growth since the Phase-2 baseline: 279 → 474 collected pytest tests, 95 → 129 Vitest.

## 8. Local dev database repair

`/home/unnikrishnan/Desktop/Jev/jev.db` was stamped `0010` but lacked 15 Phase-2 tables (events, governance,
experiments, tokens) and columns such as `intel_warnings.decision_id` / `early_warning_level` (0005) and `users.token_version`
(0010). Existing sessions of that database end once (tokens without `ver`), as for any 0010 upgrade. Its schema matched no single
revision (0006's `intel_runs.mode` existed), so re-stamping was unsafe. Steps:
1. backup `jev.db.bak-20260924T182659Z` (SQLite backup API; byte-identical to the original);
2. `uv run python scripts/repair_dev_db.py --dry-run`: rebuilt a head schema with `alembic upgrade head` and copied
   every shared table column by column; all row counts matched (users 3, ratings 12, favorites 4, genre preferences 4,
   recommendations 368, audit 2, intel rows, movies 9,742, …); the dry-run copy was deleted;
3. `uv run python scripts/repair_dev_db.py`: the same, then replaced `jev.db`;
4. `uv run alembic check` against it: no new upgrade operations.
No row was lost or deleted. A duplicate backup made by the dry run (identical SHA-256) was removed; `*.db.bak-*` and
`*.db.rebuild-*` are now gitignored.

## 9. Known limitations

- MovieLens-small is small and old; the leak-free recommender protocol is power-limited (28 users).
- Warning skill is modest (unemployment) or not demonstrated (CTA); movie warning quality is not measurable as built.
- No live traffic: experiment results are offline replays; live movie signals cannot be assessed.
- Replay determinism holds for a fixed code version, data files and active model; the API cannot pin the model.
- The calibration gate cannot fail at ~1 % base rates; the three accuracy gates jointly pass an equivalent model less
  than 80 % of the time.
- `experiments/` is gitignored: numbers must be regenerated after a clone.
- Not re-run in this phase: the full 11-minute benchmark, Docker build and acceptance (69 checks, last green at
  1.2.0), GitHub CI, PostgreSQL-only tests.

## 10. Reproducibility

The exact commands are in the README ([Reproducibility](../README.md#reproducibility)); the main ones:

```bash
uv sync --frozen
uv run python scripts/download_data.py && uv run python scripts/preprocess_data.py
uv run python scripts/download_domain_data.py us-unemployment && uv run python scripts/download_domain_data.py cta-ridership
uv run python scripts/train_models.py --calibrate
uv run python scripts/benchmark_recommenders.py          # --quick for ~100 s
uv run python scripts/evaluate_domains.py
uv run python scripts/evaluate_drift.py
uv run python scripts/simulate_ab_replay.py
uv run python scripts/retrain.py run --quick
uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run alembic check
```

## 11. Remaining roadmap

[roadmap.md](roadmap.md) "Next steps (from what 1.3 measured)": live traffic; a paired offline experiment replay;
warning skill (persistence rule, per-domain thresholds, a baseline rule, CIs everywhere); recommender power (larger
MovieLens, decayed popularity, learning-to-rank); governance statistics (shared calibration set); API replay pinning;
operations (Docker acceptance, retention jobs, a shared run lock).

## 12. Final status: **PARTIALLY_COMPLETE**

Every implementation workstream is done and every gate that can be checked on this machine passes (matrix: 13 PASS,
2 PASS with a stated scope, 1 PARTIAL). The status is not COMPLETE because:

1. **Gate 7 is PARTIAL**: the full benchmark was not re-run this phase and `experiments/` is not in the repository, so
   the ML claims are reproducible by command, not verified end to end here.
2. **Online experimentation is replay-only**: no live traffic exists, so no online effect has been measured.
3. **Docker acceptance was not re-run for 1.3.0** (user directive), and GitHub CI and the PostgreSQL-only tests were
   not run on this tree.
4. **One High finding is open on the frontend side (H5)**: the console still reads the old 409 body shape, so
   promotion blockers will not display until `frontend/src/lib/api.ts` reads the top-level `blockers`.
5. The CTA warnings show no demonstrated skill (reported, not hidden); this is a result, not a defect, but the
   Phase-2 bar (lift ≥ 1.5 on a held-out window) is not met.
