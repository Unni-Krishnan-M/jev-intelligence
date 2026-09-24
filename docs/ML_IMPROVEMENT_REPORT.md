# Recommender ML improvement report (Phase 2, WS3)

Source run: `experiments/rec-benchmark-20260924T172521Z/` (`results.json`, `REPORT.md`,
`per_user.json`), produced by `uv run python scripts/benchmark_recommenders.py`.
The run used dataset `ml-latest-small-31a303aa-wd-2bb80720a955`, git `6d29cca`, config hash
`3dfb9f004789` and seed 42, and took 636 s on a laptop CPU. The incumbent is the active model
`jev-20260923T100141Z-bbb2e4c9`. The methodology, including the leakage controls and the decision
rules, is in [ML_EVALUATION.md](ML_EVALUATION.md). All intervals are 95 % per-user bootstrap CIs,
and Δ means a paired difference. In the warm comparisons (§2), `*` marks a Holm-adjusted p < 0.05
(the family the doc adjusts); in the cold-start and calibration tables it marks a CI that excludes 0.
`experiments/rec-benchmark-*/REPORT.md` were regenerated from their `results.json` with the same rule
(`benchmark_report.significant`); no metric changed.

**Protocol caveat for the historical headline.** The per-user temporal split lets other users' later
ratings into training (cross-user leakage), so its hybrid NDCG@10 of 0.121 (+20 % over item-kNN) is
an optimistic, high-power comparison. On the leak-free global split (28 users) the hybrid is not
better than item-kNN (+0.024, CI −0.025..+0.074) or the tuned recently-popular baseline (−0.0435,
CI −0.092..+0.0005).

**The active model was not changed.** The serving default is unchanged, and so is its
`calibration.json`. Both candidate improvements failed their pre-declared adoption rule on the
leak-free global protocol, which is small, so they are available but off by default (§7).

## 1. What changed

| area | change | files |
|---|---|---|
| evaluation rigour | Per-user metric vectors on every `EvalResult`. Bootstrap CIs, paired bootstrap plus sign-flip permutation tests, Holm adjustment and user-cluster bootstrap (`evaluation/stats.py`). Both protocols in one benchmark. | `evaluation/evaluator.py`, `evaluation/stats.py`, `evaluation/benchmark.py`, `evaluation/benchmark_report.py` |
| leakage | Split-order checks. **Tag leakage found and fixed** in the evaluation path: content features are rebuilt with only the tags written before each stage's cut. Films released after a global cut are excluded. Test pins cover popularity and IDF (train only), calibration (validation only), and catalogue statistics that are never used by the ranking. | `evaluation/leakage.py`, `tests/unit/test_leakage.py` |
| cold start | `HybridConfig.cold_stages` holds profile-size conditioned weights (free weights or a popularity blend), with a new-user protocol for the global split, simulated onboarding genres, and validation-only stage tuning over 5 strategy families | `models/hybrid.py`, `evaluation/cold_start.py` |
| calibration | A per-stratum logistic calibrator on score, rank, profile size, item popularity and signal agreement. The method is chosen per stratum by validation cross-fit log-loss, and isotonic stays as the fallback. `Calibrator.predict(..., signals=None)` is backward compatible. | `calibration.py`, `engine.py`, `scripts/calibrate_recommendations.py --method` |
| baselines | Popularity gains a reach weight and an optional time decay (defaults unchanged). Popularity, item-kNN and ALS are each tuned with 12 validation configurations. | `models/popularity.py`, `configs/experiment.yaml` (`benchmark:`) |
| reproducibility | `scripts/benchmark_recommenders.py [--quick] [--protocol P]` writes `experiments/rec-benchmark-<ts>/` with the dataset version, git commit and config hash | `scripts/benchmark_recommenders.py` |
| serving surface | `Recommendation.weight_strategy` (`"adaptive"` or the name of a cold stage), so the choice can be recorded per row | `engine.py` |

## 2. Headline: warm test, both protocols (incumbent hyper-parameters)

| protocol | users | model | NDCG@10 | Recall@10 |
|---|---|---|---|---|
| user_temporal (before: reported without CI, tags leaking) | 592 | hybrid | 0.1214 | 0.0990 |
| user_temporal (after: leak-free tags) | 592 | hybrid | **0.1207 [0.1078, 0.1340]** | 0.0983 [0.0857, 0.1124] |
| | | item-kNN | 0.1006 [0.0887, 0.1128] | 0.0751 [0.0647, 0.0859] |
| | | ALS | 0.0956 [0.0838, 0.1078] | 0.0840 [0.0715, 0.0984] |
| | | popularity (tuned) | 0.0761 [0.0641, 0.0877] | 0.0487 [0.0401, 0.0572] |
| **global_temporal** (headline, leak-free) | 28 | hybrid | **0.1151 [0.0590, 0.1858]** | 0.0535 [0.0215, 0.0948] |
| | | item-kNN | 0.0915 [0.0467, 0.1444] | 0.0526 [0.0195, 0.0953] |
| | | ALS | 0.0657 [0.0214, 0.1227] | 0.0349 [0.0079, 0.0756] |
| | | popularity (default) | 0.0944 [0.0316, 0.1721] | 0.0377 [0.0091, 0.0816] |
| | | **popularity (tuned: 90-day decay)** | **0.1585 [0.0913, 0.2400]** | 0.0626 [0.0331, 0.1000] |

Paired NDCG@10 differences for the hybrid (Holm-adjusted p):

| vs | user_temporal Δ | global_temporal Δ |
|---|---|---|
| popularity (default) | +0.0432 [+0.0303, +0.0560] * | +0.0207 [−0.0393, +0.0766] |
| popularity (tuned) | +0.0447 [+0.0323, +0.0575] * | **−0.0435 [−0.0917, +0.0005]** (p_holm 0.72) |
| item-kNN (incumbent / retuned) | +0.0202 * / +0.0136 * | +0.0235 / +0.0060 (n.s.) |
| ALS (incumbent / retuned) | +0.0251 * / +0.0251 * | +0.0493 * (p_holm 0.021) / +0.0415 (n.s., p_holm 0.068) |
| hybrid fully retuned on the protocol | −0.0012 [−0.0041, +0.0018] | −0.0001 [−0.0243, +0.0253] |
| same hybrid with all-tag content (leaky) | −0.0007 [−0.0019, +0.0005] | +0.0001 [+0.0000, +0.0003] |

How to read it:

- On the high-power protocol, the hybrid beats every baseline significantly on NDCG@10 and
  Recall@10.
- On the leak-free global protocol only 28 users are evaluable, so the CIs are about ±0.06. The
  hybrid still significantly beats the incumbent ALS (p_holm 0.021). Against the retuned ALS the
  +0.0415 is not significant after the Holm adjustment (p_holm 0.068), although its unadjusted CI
  excludes 0.
- **Negative result.** A time-decayed ("recently popular") baseline, tuned on the same 12-config
  budget, scores 0.043 NDCG@10 above the hybrid on the global test. That is not significant, but
  the CI barely reaches 0. Recency of popularity matters when the future is truly unseen, and the
  hybrid's popularity signal is all-time. This is the most important open question for the
  recommender. The next step is a decayed popularity signal inside the hybrid, tuned on the global
  protocol, with a larger dataset to get the power.
- Retuning the whole pipeline on each protocol's validation gives no significant gain in either
  protocol. Neither does a 12-config item-kNN grid, although the retuned item-kNN is itself
  +0.0065 better as a standalone model. The incumbent hyper-parameters are therefore not the
  bottleneck.
- **Leakage found.** Most MovieLens tags are written in the users' own held-out period: under
  user-temporal only 265 of 1,572 tagged films keep a tag before the cut. Fixing it lowers the
  hybrid by 0.0007 (CI includes 0), so the historical 0.1214 was essentially sound, but it is now
  measured leak-free.

## 3. Cold start (goal 2)

Validation chose one strategy per bucket, and test reports it as the user-level mean NDCG@10 over
the bucket's sizes (0 | 1, 3 | 5, 10), with and without simulated onboarding genres.

| protocol | bucket | users | incumbent hybrid | chosen (validation) | hybrid + stage | popularity | Δ stage − incumbent | Δ stage − popularity |
|---|---|---|---|---|---|---|---|---|
| user_temporal | 0 | 592 | 0.0392 | popularity | 0.0447 | 0.0447 | **+0.0056 [+0.0017, +0.0097] \*** | 0 |
| user_temporal | 1–3 | 592 | 0.0338 | popularity blend 0.75 | 0.0425 | 0.0457 | **+0.0087 [+0.0028, +0.0146] \*** | −0.0032 [−0.0063, −0.0002] * |
| user_temporal | 4–10 | 592 | 0.0379 | popularity blend 0.5 | 0.0461 | 0.0477 | **+0.0082 [+0.0024, +0.0144] \*** | −0.0016 [−0.0072, +0.0038] |
| global (new users) | 0 | 88 | 0.1585 | free weights | 0.2409 | 0.1846 | **+0.0824 [+0.0521, +0.1132] \*** | +0.0564 [+0.0277, +0.0868] * |
| global (new users) | 1–3 | 88 | 0.2079 | content/profile | 0.2283 | 0.2055 | +0.0204 [−0.0212, +0.0623] | +0.0227 [−0.0208, +0.0681] |
| global (new users) | 4–10 | 88 | 0.3019 | free weights | 0.3157 | 0.2438 | **+0.0138 [+0.0024, +0.0247] \*** | +0.0719 [+0.0384, +0.1064] * |

Multiplicity: the five starred "stage − incumbent" cells have permutation p-values 0.0001, 0.0038,
0.0071, 0.0076 and 0.0189, and all survive a Holm adjustment over the 6 cells (largest adjusted
p 0.038).

At the historical 3-interaction point (user-temporal, no onboarding), the incumbent scores 0.0346,
the stage 0.0436 and popularity 0.0460. The gap to popularity shrinks from −0.011 to −0.002.

**Decision: the serving default is unchanged.** The pre-declared rule required non-inferiority
(Δ CI lower bound ≥ −0.005) in *every* bucket of *both* protocols. Global 1–3 fails it (lower
bound −0.021, on 88 users). There is a second reason: the two protocols choose different
strategies. user_temporal validation picks popularity blends, while global validation picks free
weights with more recency and content. So there is no single stage set that is proven on both. The
mechanism, the tuner and the evidence are in place. `HybridConfig.cold_stages` can be switched on
per model version once a larger dataset settles the 1–3 bucket.

Further findings:

- **Simulated onboarding genres hurt the incumbent at N = 0** (user-temporal 0.0447 → 0.0336,
  global 0.1875 → 0.1296). With `cold_start_boost`, the genre-preference signal dominates popularity
  for brand-new users, and genre match alone is a weak predictor. Every tuned stage removes this
  loss. It is the clearest single defect found.
- The gap-matrix bar "served strategy ≥ popularity in every bucket" is met against the default
  popularity baseline on the global protocol (all 3 buckets ≥, 2 significantly), but not against the
  tuned 90-day popularity baseline (Δ +0.015, −0.018, +0.022; none significant). On user-temporal the 1–3 bucket
  stays 0.003 below plain popularity, a significant difference. Popularity itself remains the
  strongest cold strategy there.
- Explanations are unaffected: a stage changes only the weights. Each item's score equals the sum
  of its six real contributions, and anchor films come from the user's own profile
  (`test_cold_stage_keeps_explanations_grounded`).

## 4. Confidence calibration and discrimination (goal 3)

Both calibrators are fitted on validation only (models fitted on train) and applied unchanged on
test. The strata are profiles truncated to 0, 3 or 10, plus the full profile. The AUC CIs are
user-cluster bootstrap intervals.

| protocol | stratum | test rows | isotonic AUC | logistic AUC | Δ AUC | Δ Brier (×10⁻⁶) | logistic ECE |
|---|---|---|---|---|---|---|---|
| user_temporal | 0 | 28,350 | 0.553 [0.524, 0.583] | 0.582 [0.547, 0.617] | **+0.029 [+0.010, +0.048] \*** | −1 [−3, +1] | 0.0015 |
| user_temporal | 3 | 28,350 | 0.566 [0.534, 0.600] | 0.624 [0.589, 0.659] | **+0.058 [+0.037, +0.080] \*** | **−5 [−8, −2] \*** | 0.0006 |
| user_temporal | 10 | 28,350 | 0.570 [0.540, 0.599] | 0.584 [0.546, 0.618] | +0.014 [−0.022, +0.049] | −9 [−18, +1] | 0.0013 |
| user_temporal | full | 28,350 | 0.632 [0.605, 0.656] | **0.673 [0.645, 0.696]** | **+0.041 [+0.020, +0.061] \*** | **−59 [−104, −13] \*** | 0.0020 |
| global | 0 | 1,250 | 0.619 | 0.301 | −0.318 [−0.640, +0.181] | +51 [−21, +174] | 0.0041 |
| global | 3 | 1,250 | 0.452 | 0.614 | +0.162 [−0.001, +0.237] | **−156 [−229, −91] \*** | 0.0032 |
| global | 10 | 1,250 | 0.551 | 0.442 | −0.109 [−0.218, +0.082] | **+500 [+132, +1029] \* (worse)** | 0.0132 |
| global | full | 1,250 | 0.610 [0.451, 0.734] | 0.780 [0.653, 0.905] | +0.171 [+0.088, +0.257] (C = 0.01, 25 users: not meaningful) | +46 [−76, +191] | 0.0038 |

Brier skill against the base rate for the full user-temporal profile goes from 0.0045 to 0.0078,
and ECE stays at or below 0.002 in every user-temporal stratum.

- **user_temporal (the only protocol with power here):**
  - the logistic calibrator lifts full-profile AUC from 0.632 to **0.673 [0.645, 0.696]**: the
    point estimate clears the Phase-2 bar of 0.65, the CI does not;
  - it lifts the 3-interaction AUC from 0.566 to **0.624 [0.589, 0.659]**: the point estimate clears
    the 0.60 bar for short profiles, the CI does not;
  - Brier improves significantly in 2 strata.
  - Calibration: the confidence is calibrated to the base rate (about 1 %; ECE ≤ 0.002, relative
    error up to 17 %, e.g. predicted 0.0072 vs observed 0.0087 at N = 0) with almost no skill
    beyond it (Brier skill ≤ 0.008) and weak discrimination (AUC 0.55–0.67). At base rates of
    0.8–1.8 % an ECE bar of 0.01 cannot fail (a constant predictor at the base rate passes it), so
    the ≤ 0.01 ECE bar is not evidence of useful calibration.

  Its largest coefficients are rank and signal agreement, then popularity (per stratum, in
  `results.json`).
- **global_temporal:** only 25 users have at least 5 validation ratings, so every estimate is very
  noisy. In 3 of the 4 strata the cross-fit chose C = 0.01, which makes the logistic model almost
  constant, so its AUC is not meaningful there. In the 10-interaction stratum it chose C = 10 and
  is significantly worse on Brier.
- **Decision: the default stays isotonic.** The pre-declared rule (no stratum of any protocol
  significantly worse) fails on global stratum 10. The logistic calibrator is shipped and tested,
  and it can be enabled per model with
  `uv run python scripts/calibrate_recommendations.py --method logistic`. That writes
  `cal-1.1.0-*`, which the engine serves through `RankedItem.signals`. Recommendation: enable it
  after re-running the benchmark on a dataset where the global protocol has power, or with the C
  grid capped (a design choice that has to be made before looking at test).

## 5. Baselines on an equal budget (goal 4)

| baseline | configs | chosen on user_temporal validation | chosen on global validation |
|---|---|---|---|
| popularity | 12 | reach 1.0, no decay | reach 0.25, **90-day half-life** |
| item-kNN | 12 | k 25, shrinkage 0 | k 200, shrinkage 50 |
| ALS | 12 | 32 factors, λ 0.01, α 5 (same as the incumbent) | 16 factors, λ 0.1, α 5 |
| hybrid (ensemble) | 55 (on top of its components) | ramp 5, boost 5, λ 0.8 | ramp 15, boost 5, λ 0.9 |

The warm comparisons in §2 use these tuned baselines as well as the incumbent's, and the hybrid's
margin over them holds on user-temporal. On global, the tuned popularity baseline is the strongest
single model (see the negative result above).

## 6. Performance

| | measurement |
|---|---|
| benchmark (both protocols, tuning, calibration) | 636 s full; about 100 s `--quick` |
| request latency, active model, 20 items | served n=3: 43.4 ms; with cold stages: 45.9 ms; n=40: 41.5 vs 40.8 ms. The machine was shared with other workstreams (load average 8–11). [recommendation-algorithms.md](recommendation-algorithms.md) records 18.7–19.0 ms for the same path on an idle machine. The stages add no cost beyond noise. |
| calibrator per item | isotonic 2.1 µs, logistic 15.4 µs (0.3 ms per page), from an ad-hoc micro-benchmark whose output was not stored (`results.json` stores only request latency); treat as indicative |

## 7. Negative results and open items

1. Cold stages are significantly better than the incumbent in 5 of 6 protocol-bucket cells, but
   they are not proven non-inferior in global 1–3, and the chosen strategies differ by protocol.
   **Not adopted.**
2. The logistic calibrator is significantly better on user-temporal (AUC 0.673 vs 0.632), but
   significantly worse on Brier in one global stratum with 25 users. **Not adopted.**
3. A recently-popular baseline beats the hybrid on the global protocol by 0.043 NDCG@10 (not
   significant, 28 users). The hybrid's popularity signal should become time-aware.
4. Onboarding genres degrade the incumbent for brand-new users (the boost is too strong on genre
   preference).
5. Full retuning on each protocol yields no gain, so hyper-parameters are not the bottleneck.
6. `ml-latest-small` gives the global protocol 28 warm, 88 new and 25 calibration users. Every
   leak-free conclusion is power-limited. Re-running the benchmark on ML-1M or ML-25M (same code)
   is the single most useful next step.
7. `scripts/train_models.py` still fits its evaluation content model on all tags. The effect is
   measured at −0.0007 NDCG@10 (n.s.), and the benchmark is the leak-free reference.
   The promotion gate (`governance/gates.py`) refits both recipes with the leak-free tag rebuild
   since v1.3.0 (`gate.json` → `split.tags`).

## 8. Reproduce

```bash
uv run python scripts/benchmark_recommenders.py            # regenerates every number above
uv run python scripts/benchmark_recommenders.py --quick    # CI smoke
uv run python scripts/calibrate_recommendations.py --method logistic --dry-run   # logistic calibration of the active model, not written
```
