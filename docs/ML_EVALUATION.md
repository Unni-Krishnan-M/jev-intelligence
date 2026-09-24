# Recommender evaluation methodology

This document describes how JEV's recommender is evaluated offline. It covers the split
protocols, relevance, metrics, uncertainty, leakage controls, cold start and confidence
calibration, plus the exact commands. The results are in
[ML_IMPROVEMENT_REPORT.md](ML_IMPROVEMENT_REPORT.md). The older single-protocol description
([evaluation.md](evaluation.md)) is still valid for `scripts/train_models.py` runs.

Every number in the improvement report is produced by one command:

```bash
uv run python scripts/benchmark_recommenders.py          # full run, both protocols
uv run python scripts/benchmark_recommenders.py --quick  # CI smoke: subsampled users, no retuning
uv run python scripts/benchmark_recommenders.py --protocol global_temporal   # one protocol
```

A run writes `experiments/rec-benchmark-<UTC timestamp>[-quick]/`:

| file | content |
|---|---|
| `results.json` | every metric, CI, paired test, tuning trial and decision, plus the dataset version, git commit, `config_hash`, seed, command and Python version |
| `per_user.json` | per-user NDCG@10 of every model and cell (to recompute any statistic) |
| `config.yaml` | the full experiment configuration used |
| `REPORT.md` | the tables, generated from `results.json` |

The configuration lives in the `benchmark:` section of `configs/experiment.yaml`. `config_hash`
is the SHA-256 of that section, the split and evaluation settings, and the incumbent
hyper-parameters. A different hash means a different experiment.

## 1. Data

MovieLens `ml-latest-small` (610 users, 9,742 films, 100,836 ratings) plus the Wikidata
enrichment. The dataset version is recorded in every result, currently
`ml-latest-small-31a303aa-wd-2bb80720a955`.

## 2. Split protocols

Both protocols are always reported. Neither is sufficient alone.

| | `user_temporal` | `global_temporal` (headline) |
|---|---|---|
| cut | each user's own time line: the last 20 % of their ratings are test, the 10 % before that validation | two global timestamps at the 70 % and 80 % quantiles of all ratings |
| cross-user leakage | **yes**: other users' later ratings are in training | **none**: nothing after the cut is seen by any fit |
| warm test users | 592 | 28 (users with history before the cut and a relevant rating after it) |
| strength | statistical power | realism: this is how a retrained model meets the future |

`global_temporal` is the headline because it is leak-free. On `ml-latest-small` it is also small:
most users rate in one burst, so few of them straddle a global cut. Its CIs are therefore wide,
and the user-temporal protocol supplies the power. A claim is only made when both protocols agree,
or at least do not contradict each other.

## 3. Stages, fitting and tuning

For each protocol:

- **validation stage**: components are fitted on `train` and scored against `validation`.
  All tuning and every choice (hyper-parameters, cold-start strategy, calibrator method and its
  regularisation) happen here.
- **test stage**: components are refitted on `train + validation` with the chosen parameters and
  scored against `test`. The test split is used for reporting only.

**Incumbent** means the hyper-parameters recorded in the active model's manifest (currently
`jev-20260923T100141Z-bbb2e4c9`), that is, the model being served. **Retuned** means the full
`training.tune` procedure re-run on the protocol's own validation split.

**Equal tuning budget for baselines.** Popularity, item-kNN and ALS each get 12 validation
configurations (`benchmark.tuning`):

- popularity: reach weight {0, 0.25, 1} × score half-life {none, 90, 365, 1095 days}. The time
  decay turns the baseline into "recently popular", which is a strong baseline under a global cut;
- item-kNN: k {25, 50, 100, 200} × shrinkage {0, 10, 50};
- ALS: factors {16, 32, 64} × λ {0.01, 0.1} × α {5, 20}.

The hybrid is an ensemble of these components, so its own search (24 Dirichlet weight samples, a
27-point cold sweep and 3 MMR settings) comes on top of theirs. It is reported as such.

## 4. Relevance and candidates

- Relevant means a held-out rating ≥ 4.0.
- A user is evaluated when they have training history and at least one relevant held-out item.
- Candidates are the whole catalogue minus the user's profile. Under the global cut, films
  released after the cut and never rated before it are also excluded (see §6).
- The same users, profiles and exclusions are used for every model. The evaluator raises an
  error if a model recommends a consumed item.

## 5. Metrics and uncertainty

- **Accuracy**: Precision, Recall, F1, NDCG, MAP and HitRate at K ∈ {5, 10, 20}. The key metrics
  are NDCG@10 and Recall@10. Beyond accuracy: coverage, intra-list diversity and novelty.
- **CIs**: percentile bootstrap over users (B = 2,000; each metric is a mean over users, so the
  user is the resampling unit). `jev_ml/evaluation/stats.py:bootstrap_ci`.
- **Paired comparisons**: models are compared on the same users. Each comparison reports:
  - the mean per-user difference with its bootstrap CI;
  - a two-sided sign-flip permutation p-value (10,000 permutations);
  - win, tie and loss shares.

  p-values within the warm family of a protocol are Holm-adjusted. "Significant" means the 95 % CI
  excludes 0.
- **Non-inferiority**: a change is "not worse" when the lower bound of the paired CI is at least
  −0.005 NDCG@10 (the margin the governance gate also uses).
- **Calibration statistics** (AUC, Brier) are row-level. Their CIs use a user-cluster bootstrap
  (B = 500), which resamples whole users with all their candidate rows.

## 6. Leakage controls

`jev_ml/evaluation/leakage.py` implements the controls below, and `tests/unit/test_leakage.py`
pins them.

| risk | control | test |
|---|---|---|
| test rows before training rows | `check_split` is asserted on every benchmark split: under the global protocol `max(train.ts) < min(val.ts)` and `max(val.ts) < min(test.ts)`; under the user-temporal protocol the same ordering per user; and no (user, item) pair appears in two parts | `test_global_split_no_test_row_precedes_train_max`, `test_check_split_detects_violations` |
| **user tags in content features** (found and fixed) | `movies.csv` aggregates *all* MovieLens tags, including tags written during the validation and test periods. The benchmark rebuilds the tags field from `tags.csv` with only the tags written before the stage's cut (global: one timestamp; user-temporal: each user's own cut). The TF-IDF vocabulary and IDF are then fitted on that leak-free metadata. | `test_tags_and_idf_use_only_tags_before_the_cut`, `test_tag_cutoffs_follow_the_protocol` |
| popularity from the future | popularity, trending and the Bayesian average come from `TrainContext.interactions` (the fit rows) only | `test_popularity_is_fitted_on_training_rows_only` |
| films that did not exist yet | under the global cut, films released after the cut year and never rated before the cut are excluded from all candidate sets | `test_future_items_are_unreleased_and_unseen` |
| catalogue statistics | `movies.n_ratings` and `mean_rating` span all ratings, but they are only used by request filters, never by the unfiltered ranking that is evaluated | `test_ranking_ignores_catalogue_statistics` |
| calibration fitted on test | both calibrators are fitted, and their method and regularisation chosen, on validation candidates only (models fitted on train). Inverting every test rating leaves every fitted parameter unchanged. | `test_fit_uses_no_test_data` (isotonic), `test_logistic_calibration_fit_uses_no_test_data` |
| new users seeing the future | in the global new-user protocol the models are fitted before the cut. The profile holds only the user's own earliest window ratings, and the targets come strictly after them. | `test_new_user_protocol_is_leak_free` |

Known residual limitations:

- The user-temporal protocol keeps its documented cross-user leakage (it is a protocol property,
  which is why the global protocol is the headline).
- Static metadata (Wikidata people, keywords, descriptions) is taken as known at every cut.
- `scripts/train_models.py` still fits its content model on all tags when it computes the test
  numbers in its experiment report. The measured effect of this is in the improvement report
  (hybrid NDCG@10 with all tags vs cut tags), and the benchmark is the leak-free reference.

## 7. Cold-start protocol

`jev_ml/evaluation/cold_start.py`. Profile sizes N ∈ {0, 1, 3, 5, 10}, grouped into buckets
0, 1–3 and 4–10. Warm (> 10) users keep the adaptive weights.

- **user_temporal**: each evaluable user's profile is truncated to their first N training
  interactions, and the user is folded in as unseen (the ALS fold-in path). The targets are the
  user's test rows, the same for every N.
- **global_temporal (new users)**: the users are those with no rating before the cut who are
  active in the held-out window. Their first 10 window ratings are the history they build in the
  app; the profile is the first N of those, and the targets are their window ratings after those
  10. The models never see any row after the cut. This is exactly how the service folds in a new
  app user.
- **Simulated onboarding**: the app asks new users for genres, but MovieLens has no such answers.
  The *onboarding* variant sets the 3 genres most frequent among the films the user liked within
  their first 10 interactions, which always precede the targets. This is an optimistic stand-in
  for a self-declared taste, so both variants (with and without onboarding) are reported.

**Strategies** (`HybridConfig.cold_stages`, a weight set per bucket):

| family | weights for the bucket |
|---|---|
| `incumbent` | the adaptive ramp and boost weights (no stage) |
| `popularity` | popularity only (the fallback) |
| `pop_blend` | (1 − a) · adaptive + a · popularity, a ∈ {0.25, 0.5, 0.75} |
| `stage_hybrid` | free weights over the six signals: the incumbent's base weights, then 20 seeded Dirichlet samples |
| `profile_content` | content and profile only: weights over content, genre preference, popularity and recency (10 Dirichlet samples) |

Per bucket, the candidate with the best mean validation NDCG@10 over the bucket's representative
sizes (0 / 1 and 3 / 5 and 10) and both onboarding variants is kept. The incumbent wins ties. The
test split then reports each bucket as a user-level mean over the bucket's cells (the users are
the same in every cell), with bootstrap CIs and paired tests against the incumbent and against
both popularity baselines.

A stage only changes the weights. Every served item still carries `{raw, normalized, weight,
contribution}` for all six signals, its score is exactly the sum of those contributions, and its
explanation is built from them (`test_cold_stage_keeps_explanations_grounded`).

**Serving decision rule** (`benchmark.decide`, test data only): the stages are adopted when, in
every protocol and every bucket, the paired NDCG@10 difference against the incumbent has a CI lower
bound of at least −0.005, and at least one bucket is significantly better.

## 8. Confidence calibration

The target is P(the user rates the recommended film ≥ 4 among their next 5 ratings), on top-50
lists. The strata are profile truncations 0, 3 and 10 plus the full profile (see
[recommendation-algorithms.md](recommendation-algorithms.md#recommendation-confidence-calibrationpy)).
Two calibrators are compared on exactly the same validation fit and test candidates:

- **isotonic** (current, `cal-1.0.0`): one 1-D isotonic map per stratum, on the score or the rank,
  chosen by a 2-fold user-parity cross-fit;
- **logistic** (`cal-1.1.0`): per stratum, a standardised L2 logistic regression on features every
  served item already carries: score, log(rank), rank, log1p(profile size), the popularity
  signal's raw value, and signal agreement (the share of active signals with a normalised value
  ≥ 0.5). C ∈ {0.01, 0.1, 1, 10} is chosen by cross-fitted log-loss. The stratum then **serves**
  whichever of its isotonic and logistic maps has the lower cross-fitted validation log-loss.

The test report covers AUC (with a cluster-bootstrap CI), Brier, Brier skill against the
validation base rate, ECE (equal-width and equal-mass), reliability bins, and the paired AUC and
Brier differences, all per stratum.

At serving time `Calibrator.predict(score, rank, n_profile, signals=None)` keeps its old
signature. With `signals` (the engine passes `RankedItem.signals`) a logistic stratum uses the
logistic model; without them it falls back to the isotonic map. This matters because
`domains/movie/user_intel.py` calls it without signals. The cost is a 6-feature dot product.

**Adoption rule**: the selected calibrator is adopted when no stratum of any protocol is
significantly worse than isotonic on test AUC or Brier, and at least one stratum is significantly
better on AUC.

## 9. Reproducing and extending

- Seeds: `seed: 42` (config) drives the splits' tie order, ALS initialisation, Dirichlet samples,
  bootstrap and permutations. Re-running the same commit, data and config gives the same
  `results.json` apart from timings.
- The full run took 636 s (about 11 min) on a laptop CPU and `--quick` about 100 s (users subsampled to
  120, 200 bootstrap samples, sizes {0, 3}, no retuning).
  Calibration is not subsampled in quick mode.
- New models plug in through `fns` in `jev_ml/evaluation/benchmark.py:run_protocol`, and they are
  then evaluated under both protocols, cold and warm, with the same statistics.
