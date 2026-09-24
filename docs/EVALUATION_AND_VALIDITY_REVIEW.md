# Evaluation and validity review (Phase 2)

Reviewer: ML evaluation and scientific rigour. Date: 2026-09-24. Scope: every headline metric in
`README.md`, `docs/ML_IMPROVEMENT_REPORT.md`, `docs/ML_EVALUATION.md`, `docs/SECOND_DOMAIN_CASE_STUDY.md`,
`docs/EXPERIMENTATION.md`, `docs/RETRAINING_AND_MODEL_GOVERNANCE.md`, `docs/STREAMING_ARCHITECTURE.md`,
`docs/platform.md`, `docs/progress.md`, and the WS4a documents (`INTELLIGENCE_ENGINE_AUDIT.md`,
`DECISION_ENGINE.md`, `EARLY_WARNING_SYSTEM.md`, read in the state they were in at 23:30 IST).

The review is based mainly on the stored run artefacts under `experiments/` and `models/`. On the
coordinator's instruction, only cheap, key reproductions were re-run (§3). The full 11-minute
recommender benchmark was started and then stopped, and its partial output directory was deleted.
No source code was edited. The reviewer's scripts are in the session scratchpad
(`cta_untouched.py`, `cta_analyse.py`), and their method is described in §3.

Verdicts: **supported**; **overstated** (the number is right but the wording claims more than it
shows); **unsupported** (no artefact or analysis backs it); **not reproduced**.

---

## 1. Top issues by severity

| # | Severity | Issue | Where |
|---|---|---|---|
| 1 | **High** | The CTA warning lift of 1.44 is not statistically distinguishable from 1. A month-cluster bootstrap 95 % CI is **[0.52, 2.36]**. The year-stratified lift is **1.18**. On a genuinely untouched window (2003–2009, 664 units) the same config gives lift **1.10 [0.00, 2.33]** (3 TP of 113 warned). On 2025-01..2026-05 it gives 0 TP of 28 warned (3 positives). The domain-aware config has **no demonstrated warning skill**. | SECOND_DOMAIN_CASE_STUDY §5.2, §6; progress.md l.385; INTELLIGENCE_ENGINE_AUDIT §4.2 |
| 2 | **High** | Evaluation-window contamination for CTA is larger than §5.4 says. The 2015–24 window has been scored at least 4 times by two workstreams: WS4b v0 (lift 0.94), WS4a "before 1.0" with special days (1.27), WS4a "after 1.1" (1.44), and the WS4a ablations and K study. WS4b attributes 0.94 → 1.44 to the special-day class alone. According to WS4a's audit, the special days account for 0.94 → 1.27, and WS4a's reversal policy accounts for 1.27 → 1.44. The number 0.147 ("tuning FPR with special days") is also WS4a's post-1.1 tuning FPR. | SECOND_DOMAIN_CASE_STUDY §5.4 vs INTELLIGENCE_ENGINE_AUDIT §4.2 |
| 3 | **High** | The governance gate is underpowered by design. The quick candidate's 90 % CI half-width for ΔNDCG@10 is 0.0064 on 594 users, which is larger than the 0.005 margin. A candidate that is **exactly as good** as the incumbent passes the NDCG gate with probability of only about 0.36. All three non-inferiority gates together pass it with probability of roughly 0.2–0.3. "REJECTED by gate" therefore means "non-inferiority not shown", not "worse" (Δ = −0.0009, p = 0.79). | RETRAINING_AND_MODEL_GOVERNANCE §3; progress.md l.388; `models/jev-20260924T174707Z-0298b516/gate.json` |
| 4 | **High** | The README headline "Hybrid NDCG@10 0.1214: +21 % over the best single model" comes from the user-temporal protocol, which has cross-user leakage. The number was also measured with leaking tags; the leak-free value is 0.1207. On the leak-free global protocol, the hybrid is **0.0435 below** a tuned recently-popular baseline (n.s., 28 users), and it does not beat item-kNN (+0.024, n.s.). The README has not been updated for Phase 2. | README l.44, l.113–122 |
| 5 | Medium | The drift detector's "precision 0.92" is measured at an artificial 1:1 prevalence and rests on **2 false positives out of 78** (FPR CI 0.007–0.089). At a 10 % prevalence the implied precision is about 0.56, and at 5 % about 0.38. The label-1 class is a synthetic splice. The session-permutation design was chosen on the same 78 negatives. | README l.206, l.290; platform.md §7(a); progress.md l.335 |
| 6 | Medium | The drift adaptation "+0.006 [+0.001, +0.013]" rests on 7 users, of whom 3 improved and 4 tied. An exact sign test gives p = 0.25. A percentile bootstrap on n = 7 is anti-conservative. Two further variants were tested on the same 7 users (`adapt_strong_floor_0.1`, `explore_lambda_0.7`, in `report.json`) but are not disclosed in platform.md. The decision (keep `standard`) is right, but the CI should not be quoted as evidence of an effect. | README l.207; platform.md §7(b) |
| 7 | Medium | Calibration claims ("well calibrated, ECE ≤ 0.0014"; "ECE ≤ 0.002, within the 0.01 bar") are near-vacuous. Base rates are 0.8–1.8 %, so a constant predictor at the base rate would also have ECE ≈ 0. Relative miscalibration is up to 17 % (for example user-temporal N = 0: predicted 0.0072 vs observed 0.0087), and Brier skill is 0.000–0.008. "Clears the Phase-2 AUC bar of 0.65 / 0.60" is only a point estimate: the CI lower bounds are 0.645 and 0.589. | ML_IMPROVEMENT_REPORT §4; README l.307; progress.md l.83 |
| 8 | Medium | The cold-start gap-matrix claim "served strategy ≥ popularity in every bucket, 2 significantly" on the global protocol is measured against the **default** popularity. Against the tuned 90-day popularity, which the same report calls the strongest baseline, the stage is significantly better in **no** bucket and is 0.018 below it in bucket 1–3. | ML_IMPROVEMENT_REPORT §3 |
| 9 | Medium | Significance stars in the warm table follow the unadjusted CI, not Holm. The global "hybrid vs ALS (retuned) +0.0415 \*" has p_holm = **0.0675**, so it is not significant after the adjustment the doc says it applies. The global calibration "full" stratum (+0.171 AUC \*) uses C = 0.01, the setting the doc itself calls "almost constant, AUC not meaningful". | ML_IMPROVEMENT_REPORT §2, §4 |
| 10 | Medium | The A/B replay's "inconclusive" is largely a design artefact. It randomises 610 users into 3 arms (about 200 each), although offline every user can be scored under every variant, which would give a paired within-user test on 610 users. The "+23 % relative NDCG@10" for recency compares different user subsets, so it partly reflects between-arm baseline differences. The latency guardrail on a replay is not meaningful. | EXPERIMENTATION §7 |
| 11 | Low | "Replay is deterministic" is demonstrated only for (a) two back-to-back replays with no intervening change (`test_same_as_of_gives_identical_decisions_via_api`) and (b) the event visibility cut at the frame level. A movie replay reads the **currently active** model manifest (`ml/jev_ml/domains/movie/ingest.py:129-133`), so a promotion or rollback between two replays of the same `as_of` changes the model risks and decisions. The doc's "however many events arrived in between" is true for events, but not for registry changes. | STREAMING_ARCHITECTURE §4.1, l.9 |
| 12 | Low | Stale or contradicted README statements: "feedback is not yet fed back into retraining" (WS2 now does this), "Open warnings are never auto-resolved" (WS4a now auto-resolves, `test_auto_resolve_after_k_live_runs`), "change-point test over-alarms 7 % vs 1 %" (WS4a reports 1.3 %), and "unemployment warning precision 0.36" (the current official run is 0.415, lift 1.41). WS4a is still editing, so these should be refreshed at integration. | README l.283–311 |
| 13 | Low | Numbers in docs that no stored artefact produces: CTA §5.4 tuning-window FPRs (0.262 → 0.147) and the 5-variant tuning table; the CTA calendar agreement of 99.91 % (a run-time quality check, not stored in the eval report); and the calibrator cost of 2.1 / 15.4 µs per item (not in `results.json`, which only stores request latency). | SECOND_DOMAIN_CASE_STUDY §2, §5.4; ML_IMPROVEMENT_REPORT §6 |

No hard-coded metrics were found in the UI or backend (§5). The only match was a docstring example
in `frontend/src/lib/evaluation.ts:15`. `EVALUATED_EFFECTS` in
`ml/jev_ml/domains/movie/user_intel.py:113` is a hand-copied copy of the drift-eval numbers. It is
disclosed and cites its source, but it will not update itself.

---

## 2. Claims table

### 2.1 Recommender (WS3; `experiments/rec-benchmark-20260924T172521Z`)

| Claim | Source | Verdict | Evidence | Required correction |
|---|---|---|---|---|
| Hybrid NDCG@10 0.1214, +21 % over the best single model | README l.44, l.117; progress l.16 | **Overstated** | 0.1214 / 0.1006 = +20.7 % on user_temporal (cross-user leakage, all-tag content). Leak-free tags: 0.1207 [0.1078, 0.1340]. Global (leak-free, 28 users): 0.1151 vs item-kNN 0.0915 (Δ +0.024, CI [−0.025, +0.074]) and vs tuned popularity 0.1585 (Δ −0.0435, CI [−0.092, +0.0005]). | See §4, R1. |
| The hybrid beats every baseline significantly on user_temporal | IMPROVEMENT §2 | Supported | Every p_holm ≤ 0.016 (results.json `warm_paired_ndcg10`). | — |
| On global the hybrid "still significantly beats ALS" | IMPROVEMENT §2 | Supported for incumbent ALS (p_holm 0.021); **overstated** for retuned ALS | `hybrid_vs_als_retuned` p_holm = 0.0675, but the table marks it \* | Mark only Holm-significant rows with \*, or state that \* is unadjusted. |
| A time-decayed popularity baseline scores 0.043 above the hybrid on global (n.s.) | IMPROVEMENT §2, §7.3 | Supported | diff −0.0435, p 0.090, p_holm 0.72 | — (a correctly reported negative result) |
| Tag leakage found and fixed; effect −0.0007 | IMPROVEMENT §2 | Supported | `hybrid_vs_hybrid_all_tags` −0.0007 [−0.0019, +0.0005] | Note that the governance gate still uses all-tag content (§2.5). |
| Cold stages significantly better than the incumbent in 5 of 6 cells | IMPROVEMENT §3, §7.1 | Supported | 5 CIs exclude 0. The permutation p-values (0.0001, 0.0038, 0.0071, 0.0076, 0.0189) all survive a Holm adjustment over the 6 cells (the largest is 0.0189 × 2 = 0.038). | Report p_holm alongside. |
| Global: stages ≥ popularity in all 3 buckets, 2 significantly | IMPROVEMENT §3 | **Overstated** | Against the default popularity, yes. Against tuned popularity: +0.015 [−0.008, +0.035], −0.018 [−0.060, +0.022], +0.022 [−0.014, +0.060]. None is significant, and one is negative. | See R4. |
| Logistic calibrator clears the AUC bar of 0.65 (0.673) and the 0.60 bar for short profiles (0.624) | IMPROVEMENT §4 | **Overstated** | The CI lower bounds are 0.645 and 0.589, both below the bars. | "…point estimate 0.673 [0.645, 0.696]; the bar is met by the point estimate, not by the CI." |
| ECE ≤ 0.002, within the ≤ 0.01 bar; "well calibrated" | IMPROVEMENT §4; README l.307; progress l.83, l.126 | **Overstated** | Base rate 0.8–1.8 %. Mean predicted vs observed: 0.0072 / 0.0087 (N = 0), 0.0084 / 0.0100 (N = 10). Brier skill 0.000–0.008, that is, almost no information beyond the base rate. An ECE bar of 0.01 cannot fail at these base rates. | See R6. |
| Global full-stratum logistic AUC 0.780, +0.171 \* | IMPROVEMENT §4 | **Overstated** | C = 0.01 in this stratum as well, which the doc says makes the model near-constant. 25 users. | Treat all global calibration rows as uninformative. |
| Full run 636 s, quick about 100 s; seed-reproducible | EVALUATION §9 | Supported | The earlier quick runs 171737Z and 172239Z give identical metrics (config hash c97466e2b481). The reviewer's quick re-run is in §3. | — |
| Request latency 43.4 / 45.9 ms; calibrator 2.1 / 15.4 µs per item | IMPROVEMENT §6 | Request latency supported (`results.json latency_ms`); per-item µs **unsupported** by any artefact | — | Store the micro-benchmark or cite its command. |

### 2.2 CTA second domain (WS4b; `experiments/platform-eval-20260924T174628Z`)

| Claim | Source | Verdict | Evidence | Required correction |
|---|---|---|---|---|
| Domain-aware warnings: lift 1.44, precision 0.090, FPR 0.224, recall 0.333 (960 units) | CASE STUDY §5.2; progress l.385 | Numbers **reproduced exactly** (§3); interpretation **overstated** | Month-cluster bootstrap lift CI [0.52, 2.36]. Year-stratified lift 1.18 (expected TP under within-year random warning 17.0 vs 20 observed). The hypergeometric p = 0.041 assumes independent units, which 8 correlated entities × 120 months are not. Within-year lift is < 1 in 2018–2021. | See R2. |
| "The domain-aware config makes the warnings informative but still weak" | CASE STUDY §5.2 | **Unsupported** ("informative") | Lift CI includes 1. Untouched 2003–09: lift 1.10 [0.00, 2.33]. 2025–26: 0/28. | See R2. |
| The special-day change was chosen on tuning replays; the final numbers are "not a pristine hold-out" | CASE STUDY §5.4 | Disclosure supported; **attribution wrong** | WS4a's audit §4.2 shows special days alone give 302 warned / lift 1.27 / FPR 0.309 on eval. The 1.44 needs WS4a's 1.1 reversal policy, which was also compared on this window. The 0.147 tuning FPR matches WS4a's post-1.1 tuning FPR. | See R3. |
| Tuning window FPR 0.262 → 0.147; 5 variant rows | CASE STUDY §5.4 | **Unsupported** (no stored artefact) | No `experiments/` file contains these numbers. | Store the tuning replays under `experiments/`, or label them "from unsaved tuning runs". |
| Forecasts beat naive and seasonal naive on 8 of 8 series (median rel. MAE 0.334 / 0.766), coverage 0.775 | CASE STUDY §5.1; progress l.385 | Supported (stored) | report.json `forecast_evaluation_window`. No significance test (e.g. Diebold–Mariano), but the margins are large (23 % vs seasonal naive) over 522 issue times. The candidates were fixed before replay. The tuning-window median of 0.736 is close. | Optionally add a DM test with HAC standard errors. |
| COVID: "warned system-wide 2–3 days before the collapse" | CASE STUDY §4.2; progress l.385 | **Overstated** | The warning fires on drops that were already observed (03-12 and 03-13, z −3.7). This is detection of the onset of one known event in a replay chosen with hindsight, not a forecast. | "JEV flagged the onset of the COVID drop from the first two days of falling ridership (data through Fri 13 March), before the steepest decline." |
| Plain generic has no skill (lift 0.98) | CASE STUDY §5.2 | Supported | 738/960 warned | — |

### 2.3 A/B replay (WS5; `experiments/ab-replay-20260924T173730Z`)

| Claim | Source | Verdict | Evidence | Required correction |
|---|---|---|---|---|
| Arms 198 / 195 / 217, SRM p = 0.50, inconclusive | EXPERIMENTATION §7; progress l.389 | Supported | report.json | — |
| Recency +0.0259 [−0.0094, +0.0613], p 0.092, "+23 % relative" | EXPERIMENTATION §7 | Numbers supported; "+23 %" **overstated** | Between-subject comparison on about 200 users per arm. Control 0.1105 vs the benchmark's full-population hybrid 0.1207 shows that arm baselines move by about 0.01 through assignment alone. | Report the relative figure with its CI (−8 % to +55 %), and add a paired within-user replay (every user × every variant). |
| Needs 13,414 members per arm | EXPERIMENTATION §7 | Supported for the unpaired design | report.json `sample_size` | Note that a paired offline design needs far fewer. |
| Guardrails not breached (latency) | EXPERIMENTATION §7 | **Overstated** for latency | Replay latency on a shared laptop is not a serving measurement. | "latency not assessed in replay". |
| A/A false-positive rate 5.9 % (z) and 6.0 % (bootstrap) is within ≤ 5 % | EXPERIMENTATION §5 | Supported with a caveat | With 2,000 z-tests the 95 % band around 5 % is ±1 %, so 5.9 % is in the upper part. The bootstrap figure on 400 runs is ±2.1 %. | — |

### 2.4 Drift (`experiments/drift-eval-20260924T045528Z`)

| Claim | Source | Verdict | Evidence | Required correction |
|---|---|---|---|---|
| Detector precision 0.92, recall 0.29, FPR 0.026 | README l.206, l.290; platform §7; progress l.335 | **Overstated** (precision) | TP 23 of 78 spliced, FP 2 of 78 untouched. Precision is at 1:1 prevalence only. FPR CI [0.007, 0.089]. At 5–10 % prevalence the precision would be about 0.38–0.56. | See R5. |
| Adaptation +0.006 [+0.001, +0.013] on 7 users | README l.207; platform §7(b) | **Overstated** as a CI | 3 improved, 0 worse, 4 tied: sign test p = 0.25. Four variants were tried on the same 7 users. | See R5. |
| 7 users show preference drift (adaptation section) vs "9 have drift; 7 only activity" (real users section) | platform §7 | Consistent but confusing | Different populations: history = train + val vs all data. | Say which data each count uses. |

### 2.5 Governance gate (WS2)

| Claim | Source | Verdict | Evidence | Required correction |
|---|---|---|---|---|
| Real quick retrain REJECTED by the gate (NDCG −0.0009, CI below the −0.005 margin) | progress l.388 | Numbers supported; wording **overstated** | gate.json: Δ −0.0009, 90 % CI [−0.0073, +0.0055], p 0.79. Cold-start Δ −0.0021 [−0.0055, +0.0016]. The candidate uses untuned default weights. | "…not shown non-inferior (Δ −0.0009, 90 % CI −0.0073..+0.0055); the gate cannot distinguish it from the incumbent." |
| The gate is leak-free because both recipes are refit on the candidate snapshot's split | GOVERNANCE §3 | Supported, with 3 caveats | (a) user_temporal keeps cross-user leakage, symmetric for both recipes. (b) `gates.py:299` loads `movies.csv` with all tags, so content features leak test-period tags. This is symmetric, and WS3 measured the effect at −0.0007. (c) The incumbent's hyper-parameters were tuned on validation of the same MovieLens split, so test is not reused for tuning. The fair-comparison concern is different: a `--quick` candidate (untuned) is compared with a tuned recipe. | Document (b), or reuse `evaluation/leakage.py` tag rebuilding in the gate. |
| Margins and statistics: −0.005 NDCG, −0.01 recall, 90 % two-sided CI (one-sided 95 %), B = 2000 (300 quick) | GOVERNANCE §3 | **Unsound power** | With a paired SE of about 0.0039 on 594 users, P(pass | true Δ = 0) ≈ 0.36 for NDCG, ≈ 0.75 for cold start and ≈ 0.74 for recall. The three gates are conjunctive, so the combined pass rate is about 0.2–0.3. B = 300 also makes the 5th-percentile bound noisy. | See R7. |
| The calibration gate compares ECE and AUC | GOVERNANCE §3, §8 | Supported as disclosed | Each model's own test set, not a shared one. At base rates near 1 %, an ECE margin of 0.01 cannot fail (see §2.1). | Use relative calibration (predicted / observed) or Brier skill on a shared set. |

### 2.6 Streaming (WS1)

| Claim | Source | Verdict | Evidence | Required correction |
|---|---|---|---|---|
| "Intelligence runs read the log bitemporally, so a replay is deterministic" | STREAMING l.9, §4.1 | **Partially supported** | The frame-level cut is tested (`test_replay_never_sees_late_arriving_events`), and so are two back-to-back replays (`test_same_as_of_gives_identical_decisions_via_api`). There is no end-to-end test with events arriving **between** two replays that compares full outputs. The replay reads the currently active model (`ingest.py:129-133`). On PostgreSQL, the watermark can miss in-flight commits (disclosed). | "…deterministic for a fixed code version, data files and active model; events that arrive after `as_of` are excluded." Add an end-to-end test. |
| 10 identical rates give 1 event | STREAMING §3 | Supported (test) | `test_rate_with_idempotency_key_ten_times_is_one_event` | — |

### 2.7 README and older intelligence numbers

| Claim | Source | Verdict | Evidence | Required correction |
|---|---|---|---|---|
| Movie forecasts: median MASE 0.93, 21/21 beat naive, 80 % coverage 0.95 | README l.181, l.202–203 | Supported numerically; coverage **misread** | A coverage of 0.95 for a nominal 80 % interval means the intervals are too wide (over-coverage), not good. It is based on 12 scored origins per series. | "80 % intervals over-cover (0.95): too wide." |
| Unemployment warnings: precision 0.36 / FPR 0.39 at base rate 0.30, recall 0.52 | README l.204, l.284 | Supported for platform-eval-051637Z; **stale** | Lift is 1.22 (not stated). The current official run (174851Z, WS4a 1.1) gives 0.415 / 0.300 / lift 1.41. | State the lift, and update at integration. |
| Lapse warnings 18/18, base rate 1.0, uninformative | README l.204 | Supported (per kind) | The overall movie base rate is 0.108 (lift 9.26), dominated by lapse. | — |
| Lapse AUC 0.886, Brier 0.115, ECE 0.061 | README l.182 | Not re-run (stored `intel-eval-20260923T134257Z`) | The base rate is 74 % (README l.311), so Brier should be compared with the base-rate Brier of 0.19. The README gives the recency-rule Brier (0.155), which is fine. | — |
| Change points 7 % false alarms at nominal 1 % | README l.185, l.306 | **Stale** | WS4a now reports about 1.3 % (INTELLIGENCE_ENGINE_AUDIT l.19). | Update at integration. |
| "App users … feedback is not yet fed back into retraining"; "Open warnings are never auto-resolved" | README l.299, l.308 | **Contradicted** by WS2 and WS4a | — | Remove or update. |
| Cold start: popularity 0.046 vs hybrid 0.035 at 3 interactions | README l.124 | Supported (0.0460 vs 0.0346) | IMPROVEMENT §3 | — |

---

## 3. Reproduction log

All commands were run from `/home/unnikrishnan/Desktop/Jev` on 2026-09-24 between 23:26 and 23:50
IST. The load average was 3–5, and WS4a was editing `ml/jev_ml/core` during the runs.

| # | Command | Result | vs reported |
|---|---|---|---|
| 1 | `uv run python scripts/benchmark_recommenders.py` (full) | Stopped after about 5 min on the coordinator's instruction. The partial output directory `experiments/rec-benchmark-20260924T175634Z` was deleted. Its log showed the same user_temporal cold-stage choices (popularity / pop_blend / pop_blend) as the reference run. | — |
| 2 | `uv run python scripts/benchmark_recommenders.py --quick` | run `experiments/rec-benchmark-20260924T180527Z-quick` (99.7 s, git 6d29cca, config hash c97466e2b481): global_temporal hybrid 0.1151, item-kNN 0.0915, ALS 0.0657, popularity 0.0944, tuned popularity 0.1585, hybrid − tuned popularity −0.0435; user_temporal (120-user subsample) hybrid 0.1292 | **Reproduced exactly**: identical to the earlier quick runs 171737Z and 172239Z, and the global_temporal values equal the full reference run 172521Z (§2 of the improvement report). Adoption decision unchanged (cold stages and logistic calibrator not adopted). |
| 3 | `scratchpad/cta_untouched.py final eval` (the CTA domain-aware config, monthly replays 2015-01..2024-12, scored with the official `entity_units` + `seasonal_outcome_fn`) | 960 units, 222 warned, TP 20, FP 202, positives 60, precision 0.0901, lift 1.44, FPR 0.224, recall 0.333 (52 s) | **Reproduced exactly** (report.json 174628Z), even with WS4a's concurrent edits |
| 4 | same, `final early` (2003-01..2009-12, never used by WS4b or WS4a) | 664 observable units, 113 warned, TP 3, positives 16, base rate 0.024, precision 0.027, **lift 1.10, month-cluster 95 % CI [0.00, 2.33]**, FPR 0.170, recall 0.19, hypergeometric p 0.53 | New: an untouched window with no skill |
| 5 | same, `final late` (2025-01..2026-05) | 136 units, 28 warned, TP 0, positives 3, FPR 0.211 | New: 0 of 3 events caught |
| 6 | same, `v0 early,late` (final config without `anomaly_calendar`) | early: 160 warned, TP 4, FPR 0.241, lift 1.04 [0.27, 1.98]. late: 34 warned, TP 0, FPR 0.256 | The special-day class **does** generalise as an FPR reduction (0.241 → 0.170 on the untouched window), but neither version has skill (lift 1.04 vs 1.10) |
| 7 | `scratchpad/cta_analyse.py` on the eval rows: 2,000 month-cluster bootstrap resamples of the lift; year-stratified expected TP | Eval lift CI **[0.52, 2.36]**; year-stratified lift **1.18** (expected TP 17.0) | New |
| 8 | Re-read stored artefacts: `results.json` (warm paired and Holm; cold buckets vs `popularity_tuned`; calibration base rates), `gate.json`, `drift-eval report.json`, `ab-replay report.json`, `platform-eval-051637Z / 174851Z` | All numbers quoted in §2 match the docs except where the table says otherwise | — |
| 9 | Grep of `frontend/` and `backend/jev_api/` for doc metric literals (0.1214, 0.1207, 0.1006, 0.92, 0.886, 1.44, 0.673, 0.632, 0.0014, 0.1585, 0.748, 0.334, 13414, 0.0259, 0.29, 0.36, 0.57, 0.63, 0.93, 1.15, 0.061, "21 %") | Only matches: CSS `leading-[0.92]` and one docstring example (`frontend/src/lib/evaluation.ts:15`) | No hard-coded metrics |

Governance gate power (analytic, from gate.json):

- SE = (ci_hi − ci_lo) / (2 × 1.645);
- P(pass | Δ = 0) = P(Z ≥ (1.645 × SE − margin) / SE).

Results:

- NDCG: SE 0.0039, pass probability 0.36;
- cold start: SE 0.0022, pass probability 0.75;
- recall: SE 0.0044, pass probability 0.74.

---

## 4. Required corrections

### Docs (exact wording)

- **R1. README l.44:** replace "Hybrid NDCG@10 **0.1214**: +21 % over the best single model" with:
  "Hybrid NDCG@10 **0.121** on the per-user temporal split (+20 % over item-kNN; this split lets other
  users' later ratings into training). On the leak-free global split (28 users) the hybrid is not
  better than item-kNN or a recently-popular baseline (see docs/ML_IMPROVEMENT_REPORT.md)."
  Add the same caveat under the table at l.113, and note that the Popularity row is untuned.
- **R2. SECOND_DOMAIN_CASE_STUDY §5.2 and §6, progress l.385, INTELLIGENCE_ENGINE_AUDIT §4.2:**
  replace "makes the warnings informative but still weak … lift 1.44" with:
  "reduces false positives 3.4× (FPR 0.77 → 0.22), but its warning skill is not statistically
  established. Lift 1.44 has a month-cluster bootstrap 95 % CI of 0.52–2.36, the year-stratified
  lift is 1.18, and on the untouched 2003–09 window the same config gives lift 1.10 (CI 0.00–2.33;
  3 of 16 events)."
  In progress l.385, replace "warning lift 1.44 (below 1.5 bar)" with "warning lift 1.44, CI
  includes 1; no skill on an untouched window".
- **R3. SECOND_DOMAIN_CASE_STUDY §5.4:** add:
  "The 0.94 → 1.44 change combines two changes scored on this window: the special-day class
  (0.94 → 1.27, WS4b) and WS4a's early-warning policy 1.1 (1.27 → 1.44). The 2015–24 window has been
  scored at least four times, so it is a development window, not a hold-out. The 2003–09 window is
  the untouched check."
  Also correct the 0.147 attribution, or store the tuning runs.
- **R4. ML_IMPROVEMENT_REPORT §3:** replace "is met by the stages on the global protocol (all 3
  buckets ≥ popularity, 2 significantly)" with:
  "is met against the default popularity baseline (all 3 buckets ≥, 2 significantly), but not
  against the tuned 90-day popularity baseline (Δ +0.015, −0.018, +0.022; none significant)."
- **R5. README l.206–207 and l.290, platform.md §7, progress l.335:**
  - Replace "precision 0.92" with "precision 0.92 at a 1:1 synthetic prevalence (2 false positives
    in 78; FPR 0.026, CI 0.007–0.089); at a 5–10 % prevalence this implies a precision of about
    0.4–0.6".
  - Replace "+0.006 [+0.001, +0.013] on only 7 drifting users" with "+0.006 on 7 drifting users
    (3 improved, 4 tied; sign test p = 0.25); no effect established".
  - Disclose the two extra variants.
- **R6. README l.307, progress l.83/126/139, ML_IMPROVEMENT_REPORT §4:** replace "well calibrated
  (ECE ≤ 0.0014)" with:
  "calibrated to the base rate (about 1 %; ECE ≤ 0.002, relative error up to 17 %) with almost no
  skill beyond it (Brier skill ≤ 0.008) and weak discrimination (AUC 0.55–0.67)."
  Replace "clears the Phase-2 bar of 0.65" with "reaches 0.673 [0.645, 0.696]: the point estimate
  clears the 0.65 bar, the CI does not".
- **R7. progress l.388, RETRAINING_AND_MODEL_GOVERNANCE §3:** replace "REJECTED by gate (NDCG
  −0.0009, CI below −0.005 margin)" with:
  "not shown non-inferior (Δ −0.0009, 90 % CI −0.0073..+0.0055), so the gate rejected it".
  Add to §8:
  "At 594 users the NDCG CI half-width (≈ 0.0064) exceeds the 0.005 margin, so an equally good
  candidate passes the NDCG gate only about 36 % of the time and all three accuracy gates about
  20–30 % of the time. Either widen the margin (e.g. 0.01 ≈ 8 % relative), use B = 2000 in quick
  mode, or evaluate on more users."
- **R8. ML_IMPROVEMENT_REPORT §2:** mark with \* only the rows with p_holm < 0.05; the global row
  "ALS retuned +0.0415" becomes n.s. (p_holm 0.068). In §4, drop the \* on the global full stratum
  (C = 0.01, 25 users).
- **R9. EXPERIMENTATION §7:**
  - Replace "(+23 % relative NDCG@10)" with "(+23 % relative, 95 % CI −8 % to +55 %; different
    members in each arm)".
  - Replace "p95 latency is not higher" with "latency is not assessed in a replay".
  - Add: "An offline replay can score every member under every variant (a paired design); this
    between-arm replay demonstrates the pipeline, not the effect."
- **R10. STREAMING_ARCHITECTURE l.9 and §4.1:** add "for a fixed code version, data files and
  active model: a movie replay reads the model that is active at replay time".
- **R11. README l.203 and l.181:** "80 % interval coverage 0.95" → "0.95 (over-covers: intervals
  too wide)".
- **R12. README Limitations:** remove "feedback is not yet fed back into retraining" and "Open
  warnings are never auto-resolved". Update the change-point and unemployment-warning numbers to
  the current official run once WS4a is final, and state the lift next to every precision.
- **R13. SECOND_DOMAIN_CASE_STUDY §4.2 and progress l.385:** replace "COVID warning 2–3 days before
  the collapse" with "COVID onset flagged from the first two days of falling ridership (data through
  13 March 2020), in a replay of a known event".

### Code issues (file:line)

- `ml/jev_ml/governance/gates.py:299` (with `_evaluate_recipe` at l.154–178): the gate loads
  `movies.csv` with all MovieLens tags, so content features see test-period tags. This is the
  leakage WS3 fixed in the benchmark (`evaluation/leakage.py`). It is symmetric between recipes,
  and small (−0.0007), but the gate should reuse the leak-free tag rebuild.
- `ml/jev_ml/governance/gates.py:77-81`: quick mode caps the bootstrap at B = 300, which makes a
  5th-percentile non-inferiority bound noisy. It should keep B ≥ 2000; the bootstrap is cheap
  compared with the refits.
- `ml/jev_ml/governance/gates.py:72-73` and `config.py` governance defaults: with the margin
  smaller than the CI half-width, the gate rejects most equivalent candidates (§2.5).
- `ml/jev_ml/domains/movie/ingest.py:129-133`: a replay reads the currently active model. Either
  record the manifest version in the replay lineage and allow pinning it, or document it (R10).
- `ml/jev_ml/evaluation/benchmark_report.py` (the \* rule): significance is taken from the
  unadjusted CI while p_holm is computed. Use p_holm for the warm family.
- `scripts/evaluate_domains.py:259-270`: the CTA warning evaluation stores no per-unit rows (only
  9 examples), so no CI or re-analysis is possible from the artefact. Store the rows, and add a
  month-cluster bootstrap CI for lift (the reviewer's `cta_analyse.py` shows the method).
- `scripts/simulate_ab_replay.py`: add a paired mode that serves every member under every variant.
- `ml/jev_ml/domains/movie/user_intel.py:113`: `EVALUATED_EFFECTS` is a hand-copied constant. Load
  it from the drift-eval report, or pin it with a test against the report.

---

## 5. Hard-coded metrics check

A grep of `frontend/src` and `backend/jev_api` (`.ts`, `.tsx`, `.py`) for every metric value quoted
in the docs found none presented as computed values. The UI reads metrics from the API.

---

## 6. What holds up well

- WS3's two-protocol design, paired per-user tests, leakage pins and pre-declared adoption rules,
  including two honest non-adoptions.
- WS4b's disclosure of the v0 run, and its use of a seasonal-naive benchmark for forecasts. The
  forecast gains are large and stable across the tuning and evaluation windows.
- WS5's statistics module (unit = member, SRM, Bonferroni, power warnings) and its explicit "not
  live traffic" labelling.
- WS2's decision to compare refit recipes rather than production artefacts, which is the correct
  way to avoid scoring a model on data it was trained on.
- The drift policy's refusal to adopt adaptation on 7 users.
