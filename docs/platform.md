# JEV platform: core engine and domain adapters

JEV is an Intelligent Decision & Early-Warning Engine. It turns changing data into signals, trends, anomalies,
forecasts, risk assessments, structured decisions, explanations, recommendations and actions. Everything
domain-specific lives behind a **domain adapter**. The movie recommender is the first adapter, and the reference
implementation. A second adapter runs the same engine on any structured time-stamped dataset.

This document is the binding design contract for the platform migration (v1.2). [intelligence.md](intelligence.md)
stays authoritative for the result contract of each stage. This document adds the domain dimension and the new
capabilities, and it never renames an existing field.

## 1. Layers

```
                 ┌──────────────────────── JEV core (jev_ml/core) ─────────────────────────┐
 DataSource ──►  │ validate → series → signals → trends/change points → anomalies → forecast │
 (adapter)       │ → risk → JEV decisions (typed, versioned) → early warnings → actions      │
                 │ → explanations/evidence · scenarios · feedback · evaluation               │
                 └───────────────▲──────────────────────────────────────────▲───────────────┘
                                 │ generic observations + SeriesSpecs         │ domain extras (hooks)
          ┌──────────────────────┴───────────┐                ┌──────────────┴───────────────────────┐
          │ domains/generic (any CSV + config)│                │ domains/movie (MovieLens + app DB)    │
          │ e.g. US state unemployment (BLS)  │                │ raters, lapse, model governance,      │
          └───────────────────────────────────┘                │ preference drift, hybrid recommender  │
                                                               └───────────────────────────────────────┘
```

- **Core** (`ml/jev_ml/core/`): domain-independent types, the generic pipeline, and the engines that operate on
  generic `Series`. These are series building, trends, change points, series anomalies, forecasting, scenarios,
  risk scoring, the decision framework (specs, confidence kinds, abstention, batches), the early-warning decision,
  warnings, actions, signals and the drift detectors. Core never imports `jev_ml.models`, `jev_ml.engine` or
  anything movie-specific.
- **Domain adapters** (`ml/jev_ml/domains/<name>/`): an adapter loads its data as generic observations, declares
  which series to build, and may contribute extra stages through hooks.
  - The movie adapter wraps the existing recommender code (`jev_ml.models`, `jev_ml.engine`, `jev_ml.training`,
    `jev_ml.evaluation`, `jev_ml.calibration`), which stays where it is. Artifacts and imports keep working;
    nothing is deleted.
- **Compatibility:** `jev_ml.intel.run_pipeline(PipelineInputs)` keeps working and returns exactly what it returned
  before for the movie domain, plus additive fields. It becomes a thin shim over the core pipeline plus the movie
  adapter. A golden regression test compares the movie output before and after the migration (ids, counts, keys,
  values).

## 2. Core types (`jev_ml/core/types.py`)

These are dataclasses whose `to_dict()` returns the wire format of [intelligence.md](intelligence.md) §4, plus the
additive fields below. Every emitted object gains `"domain": "<adapter key>"`.

| Type | Key fields (existing, then **added**) |
|---|---|
| `DataSource` | source, kind (static_snapshot/live/artefact/dataset), rows, first/last event, age/lag, expected_update, fresh, **license, url, checksum** |
| `Entity` | **entity_id** (`"<type>:<name>"`), entity_type, name, attributes |
| `Observation` (frame schema) | **timestamp (s), entity_id, entity_type, event_type, value (float/NaN), source, group columns** (e.g. genre) |
| `SeriesSpec` | **id template, metric (count/share/mean/nunique/level), group_by, unit, adverse_direction (up/down/both/none), is_count** |
| `Signal` | id, kind, entity_type, entity, value, unit, strength, direction, source, observed_at, evidence, **entity_id, baseline, change, confidence, confidence_kind** |
| `Trend` | existing fields, **magnitude (recent − prior), velocity (= slope per period), baseline (= prior_mean), supporting_observations (= n_points), confidence (= 1 − q), confidence_kind "evidence"** |
| `Anomaly` | existing fields, **observed_value, expected_value, anomaly_score (0..1), confidence, confidence_kind** |
| `Forecast` | existing fields, **domain, horizon unit** |
| `Risk` | existing fields, **severity (= level), contributing_factors (= factors), domain** |
| `Decision` | existing fields |
| `Recommendation` | **item_id, score, rank, reason, evidence, confidence, decision_id** (the decision it is downstream of) |
| `Evidence` | kind, label, value, detail, ref |
| `Scenario`, `Action`, `Feedback` | existing contracts |

The added fields are derived from values the stage already computes. None may be filled with constants or
placeholders. When a value does not exist, emit `null` together with the reason in `detail` (or leave it out).

## 3. Domain adapter protocol (`jev_ml/core/adapter.py`)

```python
class DomainAdapter(Protocol):
    key: str                                  # "movie", "generic:us-unemployment"
    info: DomainInfo                          # name, description, entity types, frequency, sources (+licence),
                                              # capabilities: {"recommendation": bool, "user_intelligence": bool}
    def load(self, as_of, now) -> DomainData  # observations frame (≤ as_of after core filtering), entities,
                                              # series specs, sources, adapter-private context
    def quality_checks(self, data) -> list[Check]            # adapter checks, added to the core checks
    def extra(self, ctx: CoreContext) -> DomainExtras       # optional: anomalies, predictions, risks, decisions,
                                                             # signals, diagnostics (movie: raters, lapse, models)
```

`run_domain(adapter, as_of=None, now=None, suppressed_keys=None, config=None) -> PipelineResult` is the generic
entry point. A registry `jev_ml.domains.available()` lists the adapters that can load on this machine: the movie
adapter needs `data/processed`, and a generic adapter needs its dataset file. Every other adapter is listed with
`available: false` and a reason.

## 4. JEV decision layer additions

**Early-warning decision** (`early_warning_level`, core, `choice`), one per situation (an entity/series with
evidence). It answers "Should this situation trigger an early warning?" with one of
`NO_ACTION | MONITOR | WARNING | URGENT_ACTION`.
- **State** (structured evidence, all computed upstream): signal strength, trend direction plus q-value, anomaly
  score and severity, forecast direction relative to `adverse_direction`, and risk score and level.
- **Policy** `ewl-1.0.0`: documented, monotone point scores. More adverse evidence can never lower the level.
  Confidence is a `margin` between the top two levels. It abstains when the only evidence is a stage that was
  skipped.
- **Warnings are downstream of this decision.** A warning is raised only when the answer is `WARNING` or
  `URGENT_ACTION`. It carries `decision_id`, and its severity maps from the level (plus the risk/anomaly severity).
- **Movie compatibility:** for the movie domain the thresholds reproduce the current warning set (same keys and
  severities), so the golden test still passes.

**Recommendation strategy** (movie, per user, `choice`). It answers "How should this user's recommendations be
produced?" with one of `standard | adapt_to_recent | explore`.
- **State:** the drift report (§5), profile size, and recommendation-confidence statistics.
- **Hybrid configuration:** `adapt_to_recent` shortens the profile recency half-life, and `explore` raises the MMR
  diversity λ. The exact parameters are declared in config and their effect is evaluated (§7).
- **Downstream:** recommendations are served under the chosen strategy and carry `decision_id`, the strategy and
  its evidence.

## 5. Preference drift (core detector + movie adapter)

Core `jev_ml/core/drift.py` provides generic drift tests between an entity's *historical* and *recent* windows
(split by time; the recent window is the last N events or days, configurable):

| Aspect | Test | Output |
|---|---|---|
| categorical distribution (movie: genres) | Jensen–Shannon distance + permutation test (seeded) | historical/recent top categories, JS, p |
| numeric level (movie: rating) | Mann–Whitney U / mean shift with bootstrap CI | shift, CI, p |
| event rate (movie: viewing frequency) | Poisson rate ratio test | ratio, CI, p |
| numeric attribute (movie: release year) | median shift with bootstrap CI | shift, CI, p |
| content similarity (movie: TF-IDF centroid) | cosine between window centroids vs a within-history permutation null | cosine, p |
| acceptance (movie: recommendation feedback) | two-proportion test, when there is enough app feedback | rates, p |

- Every aspect runs with an explicit minimum sample size. Below it, the aspect reports `insufficient_data`.
- Holm correction across aspects. `drift_detected` means at least one aspect is significant after correction.
- Confidence is 1 − (smallest adjusted p), labelled `confidence_kind: "evidence"`. It is not a probability.

## 6. Generic structured dataset adapter

- **Config:** `configs/domains/<name>.yaml`: source file, timestamp column, entity column, value column, optional
  group columns, frequency, metrics (`level` / `count` / `mean` / `share` / `nunique`), adverse direction, and an
  optional declared impact weight per entity (labelled as declared).
- **Second reference domain:** `cta-ridership`, daily Chicago Transit Authority boardings (City of Chicago Data
  Portal): see [SECOND_DOMAIN_CASE_STUDY.md](SECOND_DOMAIN_CASE_STUDY.md).
- **Demo dataset:** `us-unemployment`, monthly unemployment rates for the US and a set of states, from the Bureau
  of Labor Statistics via FRED (public domain).
  - `scripts/download_domain_data.py us-unemployment` fetches it, records the SHA-256 of each file, and writes a
    long-format CSV to `data/raw/domains/us-unemployment/`.
  - Like MovieLens, the data is never committed.
- **Replay** works exactly as for movies. For example, as of 2008-06-01 the engine should see unemployment rising
  before the recession peak, and as of 2020-05-01 the April-2020 spike. What it actually reports is whatever the
  data and the documented methods produce.

## 7. Evaluation additions (all generated by real experiments)

| Metric | Protocol |
|---|---|
| Warning precision / false-positive rate | monthly replays (leak-free); a warning is *confirmed* when its adverse condition is observed in the following `h` periods (movie lapse warnings: realised lapse among flagged users; generic: adverse move beyond noise) |
| Decision consistency | flip rate of `early_warning_level` between consecutive monthly replays; monotonicity property tests |
| Forecast MAE / RMSE / calibration | existing rolling-origin backtest, now per domain (+ RMSE), with interval coverage |
| Drift detector precision / recall | labelled synthetic drift created by splicing real user histories (genuine user A's history followed by user B's recent events) vs untouched histories |
| Drift adaptation effect | NDCG@10 on the test split for users with detected drift: `standard` vs `adapt_to_recent`, with a paired bootstrap CI |
| Recommendation usefulness from feedback | accepted / rejected rates from app feedback, reported as "insufficient data" until feedback exists |

## 8. API additions (all backward compatible)

- `GET /intel/domains`: adapters with info, availability, latest run and open-warning counts.
- `POST /intel/runs` accepts `{"domain": "<key>", "as_of"?}`. The default domain is `movie`.
- Every `/intel/*` read accepts `?domain=`, default `movie`. Rows gain a `domain` column (migration `0005`), and
  "at most one open warning per key" becomes per (domain, key).
- `GET /me/intelligence` (any signed-in user):
  - preference history (genre shares over time windows);
  - the drift report;
  - the current recommendation-strategy decision with its evidence;
  - the user's signals.
- `POST /me/intelligence/scenarios` projects the user's preference trend (continue / accelerate / reverse) and
  ranks recommendations under each projected preference with the real hybrid model. It returns assumptions,
  uncertainty (from resampling the user's events) and evidence.
- `GET /recommendations` responses gain an `intelligence` block: strategy decision id, answer, confidence,
  confidence kind and evidence summary. Existing fields are unchanged.
- `POST /me/intelligence/feedback` accepts `accepted | rejected` on a strategy decision or a recommendation.
  Operator feedback stays under `/intel/feedback`.

## 9. Frontend information architecture

- The landing page and README present JEV as the decision and early-warning engine. "Movies" is one domain.
- The top bar shows **Intelligence** (the platform console) and **Movies** (the existing member app: Tonight, For
  you, Discover, Library).
- **Console** (`/intel`):
  - A domain switcher (Movies / US unemployment / …) kept in `?domain=`.
  - Navigation: Overview, Signals, Trends, Anomalies, Predictions, Early warnings, Decisions, Recommendations,
    Evidence, Scenarios, Feedback, Model evaluation, then a System group (health, audit).
  - Pages hide sections that the chosen domain does not produce. They say why rather than render empty movie
    widgets.
- **"My intelligence"** (`/me/intelligence`, any user): preference history, drift, the strategy decision with its
  evidence, recommendations downstream of that decision, preference what-if scenarios, and accept/reject feedback.

## 10. Response shapes for the new endpoints

```jsonc
// GET /intel/domains
{"items": [{
  "key": "movie", "name": "Movies", "description": "…", "entity_types": ["platform", "genre", "user", "model"],
  "frequency": "month", "sources": [DataSource],
  "capabilities": {"recommendation": true, "user_intelligence": true, "lapse": true, "raters": true,
                   "model_governance": true, "scenarios": true},
  "available": true, "reason": null,               // reason when available == false
  "latest_run": Run | null, "warnings_open": 0
}]}

// GET /me/intelligence
{"user_id": 1, "as_of": "ISO", "profile": {"n_events": 0, "first_event": "ISO|null", "last_event": "ISO|null"},
 "preference_history": {"categories": ["Drama", "…"],
                        "windows": [{"label": "historical|recent|<period>", "start": "ISO", "end": "ISO", "n": 0,
                                     "shares": {"Drama": 0.31}}]},
 "drift": {"status": "ok|insufficient_data", "drift_detected": false, "confidence": 0.0, "confidence_kind": "evidence",
           "historical_window": {"start": "ISO", "end": "ISO", "n": 0}, "recent_window": {"start": "ISO", "end": "ISO", "n": 0},
           "aspects": [{"aspect": "genre_distribution|rating_level|activity_rate|release_year|content_similarity|acceptance",
                        "status": "ok|insufficient_data", "test": "…", "statistic": 0.0, "p_value": 0.0,
                        "p_adjusted": 0.0, "significant": false, "effect": {}, "detail": "…"}],
           "summary": "…", "evidence": [Evidence]},
 "strategy": Decision,                 // spec_id "recommendation_strategy"; options standard|adapt_to_recent|explore
 "signals": [Signal],
 "recommendations": [{"item_id": 1, "title": "…", "rank": 1, "score": 0.0, "reason": "…", "confidence": 0.0,
                      "decision_id": "dec-…", "evidence": [Evidence]}]}

// POST /me/intelligence/scenarios  body: {"k": 10, "scenarios": [{"name": "…", "kind": "continue|accelerate|reverse", "factor": 1.0}]}
{"as_of": "ISO", "assumptions": ["…"], "uncertainty_note": "…",
 "baseline": {"shares": {"Drama": 0.3}, "recommendations": [Recommendation]},
 "scenarios": [{"name": "…", "kind": "…", "assumptions": ["…"],
                "projected_shares": {"Drama": {"mean": 0.35, "lo80": 0.28, "hi80": 0.41}},
                "recommendations": [Recommendation], "overlap_with_baseline": 0.6}],
 "evidence": [Evidence]}

// POST /me/intelligence/feedback  body: {"target_type": "strategy|recommendation", "target_id": "dec-…|<item_id>", "verdict": "accepted|rejected", "note": null}
// → {"id", "target_type", "target_id", "verdict", "created_at"}

// GET /recommendations → existing response + "intelligence": {"decision_id", "strategy", "confidence",
//   "confidence_kind", "drift_detected", "summary", "evidence": [Evidence]}
```

## Implementation notes (preference drift & user intelligence)

Code: `ml/jev_ml/core/drift.py` (generic tests), `ml/jev_ml/domains/movie/user_intel.py` (drift report,
strategy decision, serving under the strategy), `ml/jev_ml/domains/movie/user_scenario.py` (what-if
scenarios), `scripts/evaluate_drift.py` (§7 numbers), tests in `tests/drift/`. `engine.py` is unchanged.

**Public functions**

- `user_intelligence(engine, interactions, *, as_of=None, feedback=None, k=10, config=None, user_id=None,
  genre_prefs=(), excluded_movie_ids=()) -> dict`: the `GET /me/intelligence` payload.
- `strategy_for(engine, interactions, feedback=None, *, user_id=None, as_of=None, genre_prefs=(),
  excluded_movie_ids=(), config=None, k=10, profile=None) -> (decision, profile_kwargs, config_overrides)`.
  Serve with `build_strategy_profile(engine, interactions, genre_prefs, excluded, **profile_kwargs)` and
  `engine_with_overrides(engine, config_overrides).recommend(...)`. Pass the request's standard `profile` so it is
  not built twice. The decision id is the same as in `user_intelligence` for the same inputs.
- `user_preference_scenarios(engine, interactions, spec, *, as_of=None, genre_prefs=(), excluded_movie_ids=(),
  config=None) -> dict`: the `POST /me/intelligence/scenarios` payload. It raises `ValueError` for k outside
  1..50, more than 4 scenarios, an unknown kind or an out-of-range factor.
- `feedback` rows are mappings or objects with `timestamp` (or `created_at`) and either `accepted: bool` or
  `verdict`/`feedback` (`accepted|like|clicked` count as accepted, `rejected|dislike|not_interested` as rejected).
- `as_of` defaults to the latest event. Events after `as_of` are ignored.

**Shapes.** These follow §10. Additive fields only: the decision carries `domain: "movie"`, signals have
`kind: "drift"` plus `dedup_key`, `title`, `window`, `freshness_days` and `domain`, and recommendations carry
`confidence_kind` and `strategy`. Scenario recommendations carry `decision_id: null`, because they are not
downstream of a decision. Each scenario also echoes its `factor`. An abstained strategy decision has `answer: null`
and `confidence: null` (intelligence.md convention). The strategy actually served (`standard`) is in
`state.served_strategy` and in every recommendation's `strategy`.

**Events and windows.** Each movie is one event: its first engagement time, with its latest rating. The recent window
is the last `recent_n = 20` events, or the last `recent_days` days when that is set. It is extended to whole UTC days,
so no session is split. The historical window is everything before it. The report is `insufficient_data` when:
- there are fewer than 10 recent or 20 historical events, or
- all events fall within 14 days, or
- every aspect is below its own guard.

**Tests** (Holm across the aspects that ran, alpha 0.05, 499 permutations or bootstraps, seeded per aspect):

| Aspect | Test |
|---|---|
| genre_distribution | JS distance of per-event genre shares |
| rating_level | Mann–Whitney U. Mean shift with a bootstrap CI. p from the session permutation of \|rank-biserial r\| |
| activity_rate | Rate of *active days*, quasi-Poisson rate ratio. Overdispersion comes from 28-day blocks of the history. The window-boundary days are excluded: both windows start on an active day by construction, which inflated short windows |
| release_year | Median shift of the *film's age when watched* (event year − release year). Raw release years drift for every long-lived user because the calendar moves. Raw medians are kept in `effect` |
| content_similarity | Cosine of the TF-IDF centroids of the content model. The release-decade tokens are removed for the same reason. Null: within-history permutation |
| acceptance | Pooled two-proportion z |

`statistic` is the aspect's headline effect (JS distance, mean shift, rate ratio, median shift, cosine, rate
difference). The raw test statistics are in `effect`. `confidence` is 1 − the smallest adjusted p, with kind
`evidence`.

**Session-cluster permutation.** This is the most important design choice. MovieLens ratings arrive in bursts
(median user: all ratings within about an hour). An event-level permutation test therefore flags "drift" whenever the
recent window is one session that differs from the average. The permutation tests instead permute whole active days:
the recent window's k clusters are compared with k clusters drawn at random. This is exact under exchangeable
sessions, whatever the session sizes. A test needs at least 8 historical sessions. A size-matched variant (drawing
days until the event count matches) was tried and rejected: its false-positive rate on untouched histories rose to
0.18, against 0.03 for the exact version.

**Strategy policy `strategy-1.0.0`** (choice, margin confidence):
- Evidence in bans, e(p) = min(4, −log10 p).
- `standard` scores c = −log10(0.05). The preference evidence e(min adjusted p over genre, content, film age and
  rating) goes to `adapt_to_recent` if its offline effect is proven, otherwise to `explore` if its cost is proven
  small, otherwise to neither. "Proven" means the paired-bootstrap CI lower end of ΔNDCG@10 is > 0 for adapt, or
  ≥ −0.01 for explore, measured on at least 30 drifting users.
- A significant *drop* in acceptance adds its evidence to `explore`. This is a rule; MovieLens has no feedback, so it
  could not be validated offline.
- The answer is the argmax. Confidence = (best − second) / (best + second).
- The policy is monotone: more drift evidence never lowers an alternative's score.
- The decision abstains when the drift report is insufficient.

**Strategy parameters.**
- `adapt_to_recent`: profile weights are multiplied by 0.5 + 0.5·0.5^(age / h). The half-life h is the span of the
  recent window, clamped to [14, 365] days. The decay has the same form as `UserProfile.from_events`, applied after
  `engine.build_profile`, so no engine change is needed.
- `explore`: MMR λ 0.6 instead of the tuned 0.8.

### Evaluation (`experiments/drift-eval-20260924T045528Z`, model `jev-20260923T100141Z-bbb2e4c9`)

**(a) Detector on labelled synthetic drift.** 164 real users have at least 40 events over at least 14 days.
- Label 1: user A's full history plus the last m unseen events of a random other user B. The added events are laid
  out in A's own session structure (A's gaps between active days and A's events per day), so only *what* is watched
  and *how it is rated* changes.
- Label 0: A untouched. Real histories can contain genuine drift, so the FPR is an upper bound.
- Precision is at 1:1 prevalence. The same 78 untouched histories have enough sessions to test.
- **Reading the precision.** "Precision 0.92" (m = 20) is precision at a 1:1 synthetic prevalence (23 true positives
  of 78 spliced, 2 false positives in 78 untouched; FPR 0.026, 95 % CI 0.007–0.089). At a 5–10 % prevalence this
  implies a precision of about 0.4–0.6. The label-1 class is a synthetic splice, and the session-permutation mode was
  chosen on the same 78 negatives.

| mode | m | preference aspects P / R (95% CI) / F1 | FPR (pref.) | any aspect P / R | FPR (any) |
|---|---|---|---|---|---|
| session permutation (served) | 10 | 0.85 / 0.13 (0.08–0.22) / 0.23 | 0.026 | 0.67 / 0.22 | 0.115 |
| session permutation (served) | 20 | 0.92 / 0.29 (0.21–0.40) / 0.45 | 0.026 | 0.72 / 0.29 | 0.115 |
| session permutation (served) | 40 | 0.83 / 0.17 (0.10–0.29) / 0.29 | 0.026 | 0.53 / 0.17 | 0.115 |
| event permutation (rejected) | 20 | 0.58 / 0.95 / 0.72 | 0.585 | 0.57 / 0.95 | 0.610 |

- Per aspect (served mode, m = 20), recall is: genre 0.16, rating 0.12, film age 0.19, content 0.10. The FPR of each
  is ≤ 0.016.
- activity_rate is not changed by the splice. Its rate of 0.10 on untouched histories is a false-alarm rate. It is
  the reason "any aspect" has a higher FPR than the preference aspects.
- m = 40 has lower recall than m = 20 because half of the spliced events then fall into the historical window.
- The detector is therefore precise but conservative: only strong, multi-session changes are flagged.

**(b) Adaptation effect.**
- Protocol: the manifest's per-user temporal split. History = train + validation, relevance = test rating ≥ 4,
  profiles are folded in, K = 10.
- Primary setting (`refit`): the components are refitted on train + validation with the active model's recorded
  hyper-parameters, which is leak-free for the test window.
- `active`: the served artifacts, which were trained on all data including the test window. It is leaky and shown
  for comparison only.
- Of 592 users, 7 show preference drift in their history. 545 are one- or two-session histories (insufficient).

| group (n) | setting | NDCG@10 standard | Δ adapt_to_recent (95% CI) | Δ explore (95% CI) |
|---|---|---|---|---|
| preference drift (7) | refit | 0.121 | +0.0063 (+0.0014..+0.0134) | +0.0028 (−0.0014..+0.0077) |
| preference drift (7) | active (leaky) | 0.220 | +0.0039 (+0.0006..+0.0086) | +0.0087 (−0.0085..+0.0309) |
| event-mode drift (59) | refit | 0.165 | +0.0007 (−0.0083..+0.0092) | −0.0020 (−0.0090..+0.0052) |
| all users (592) | refit | 0.121 | +0.0011 (−0.0003..+0.0027) | −0.0016 (−0.0039..+0.0006) |

Recall@10 differences are 0.000 for the 7 drifting users (refit).

The +0.0063 for `adapt_to_recent` on 7 drifting users rests on 3 improved, 4 tied and 0 worse (exact sign test
p = 0.25); a percentile bootstrap on n = 7 is anti-conservative, so **no effect is established**. Two further
variants were evaluated on the same 7 users and are stored in the same `report.json` (not adopted):
`adapt_strong_floor_0.1` (+0.0298, 5 improved, 0 worse; sign test p = 0.06; +0.0016, n.s., on all 592 users) and
`explore_lambda_0.7` (+0.0015, CI −0.0014..+0.0048). Four variants tried on 7 users is itself a multiple-comparison
problem. Counts: the 7 users here are drift in the *train + validation* history (the adaptation protocol); the 9
under "Real users" below are drift over *all* data.

**Decision from the data.** The only CIs that exclude zero rest on 7 users (3 improved for the served variant). That is too few to
change what drifting users are served. The larger event-mode group of 59 shows no effect, and neither does the full
population. The policy therefore requires at least 30 evaluated drifting users. Neither alternative qualifies, so
drifting users keep `standard`. The decision still reports the drift, and the rationale and evidence show the
measured effects. Rerunning `scripts/evaluate_drift.py` on a larger dataset and updating `EVALUATED_EFFECTS` in
`user_intel.py` is the intended way to enable adaptation.

**(c) Latency.** All 610 real users, with the standard profile passed in:
- `strategy_for`: p50 4.1 ms, p95 18.0 ms, max 29.8 ms. For typical users (interquartile history of 35–168 events)
  the p95 is 5.5 ms.
- Full `user_intelligence`: p50 19 ms, p95 27 ms.

**Real users (active model).** Across all 610 users:
- 532 are `insufficient_data`, mostly single-session histories. User 1, for example, has 232 ratings within 9 days.
- 69 have no drift.
- 9 have drift; in 7 of them only activity drifts.
- User 18 (502 events, 2016–2018): genre mix (JS 0.42, p_adj 0.032) and content profile (cosine 0.55 vs 0.75
  typical) drifted.
- User 573: ratings rose from 4.17 to 4.79 (p_adj 0.01).
- User 318 (879 events, 2009–2018): the recent window is 47.5 % Documentary against 8 % historically, but this is not
  significant after Holm correction (p_adj 0.11): 20 events in few sessions.
- With the calendar confounds (raw release year, decade tokens) in place, every long-lived user had "release-year
  drift". Removing them was necessary for plausible reports.

**Scenarios.**
- Bands are 80 % bootstrap intervals (200 resamples of the user's events within each window). Mean = the projection of
  the observed shares; the band is widened to include it when the bootstrap distribution is degenerate.
- Ranking: each profile event is reweighted by the projected / baseline share of its genres (clipped to [0.1, 10])
  and ranked by the served hybrid model. A scenario with no projected change reproduces the baseline ranking exactly.
- Example: for user 318, "continue" projects Documentary 0.72 (80 % band 0.50–0.83) and "reverse" projects
  Documentary 0.08. Overlap with the baseline top 8 is 0.75 and 0.63.

## Implementation notes (platform core)

Covers sections 1–4, 6 and the warning-precision / decision-consistency / forecast rows of section 7.

### Code and public API
- **Core** `ml/jev_ml/core/`: `types.py` (§2 dataclasses, `typed(result)`), `adapter.py` (`DomainAdapter`,
  `DomainInfo`, `DomainData`, `CoreContext`, `DomainExtras`), `config.py` (`CoreConfig`), `series.py`
  (`SeriesSpec`, observation → series builder), `quality.py`, `trends.py`, `anomalies.py`, `forecast.py`,
  `scenario.py`, `risk.py`, `decisions.py` (`DecisionSpec`, confidence kinds), `batches.py`,
  `early_warning.py`, `warnings.py`, `actions.py`, `signals.py`, `pipeline.py` (`run_domain`), `evaluation.py`.
  A test parses every core module and fails on an import of `jev_ml.models|engine|training|calibration|data|
  features|signals|evaluation|intel|domains`.
- **Movie adapter** `ml/jev_ml/domains/movie/`: `config.py` (`IntelConfig(CoreConfig)`), `ingest.py`, `series.py`,
  `raters.py`, `lapse.py`, `modelstats.py`, `risks.py`, `decisions.py`, `actions.py`, `signals.py`,
  `evaluation.py`, `adapter.py` (`MovieAdapter`). These are the former `jev_ml/intel` modules, moved (not copied);
  the recommender packages are untouched.
- **Generic adapter** `ml/jev_ml/domains/generic/` (`GenericAdapter`, YAML schema in `config.py`) and
  `configs/domains/us-unemployment.yaml`. Registry: `jev_ml.domains.available()` → one item per adapter with
  `DomainInfo` fields + `available` + `reason`; `jev_ml.domains.get_adapter(key)`.
- **Compatibility** `jev_ml/intel/*`: re-exports of the moved names; `run_pipeline(PipelineInputs, config)` =
  `run_domain(MovieAdapter(inputs), inputs.as_of, inputs.now, inputs.suppressed_keys, config)`;
  `run_scenario` accepts a `PipelineResult` or `PipelineInputs`; new `run_default(as_of, now)`.
- **How to call a domain** (backend): `run_domain(get_adapter("movie", ...) or MovieAdapter(inputs), as_of, now,
  suppressed_keys)` — the existing `run_pipeline(inputs)` path is unchanged — and for generic domains
  `run_domain(get_adapter("generic:us-unemployment"), as_of, now, suppressed_keys)`. Scenarios on any result:
  `jev_ml.core.scenario.run_scenario(result, spec)`.
- CLIs: `scripts/run_intelligence.py [--domain KEY] [--as-of D] [--list]`,
  `scripts/download_domain_data.py us-unemployment`, `scripts/evaluate_domains.py`.

### Pipeline (`run_domain`)
`adapter.load(as_of, now, config)` → core validation (skipped when the adapter validated its own raw data:
`DomainData.core_checks = False`, recorded in `run.validation`) + `adapter.quality_checks` → series → trends →
series anomalies → forecasts (+ share forecasts) → `adapter.extra(ctx)` (movie: raters, live feedback, lapse,
model governance, movie risks, the three movie batches, movie signals, action mapping) → risks (adapter risks +
quality + staleness + generic families) → early-warning decisions (batch `early_warning`) → warnings → actions →
signals → summary. Stage timings keep the v1.1 keys. Every object gains `domain`; `run` gains `domain`,
`core_version` (`core-1.0.0`), `frequency`, `validation`. `summary.counts.early_warning` counts levels.

### Series builder
Metrics `count | share | mean | nunique | level | sum`, monthly, weekly (Mon..Sun) or daily, list-valued group columns (a film
counts once per genre), `grid_end = as_of` (partial month excluded from every baseline) or
`last_observation` (published statistics). Consecutive specs with the same `group_by` are emitted entity by entity,
which reproduces the v1.1 series order. Spec flags route series to trend / anomaly scan / forecast / share
forecast; `volume_guard` + `min_volume`, `resolution` (published precision guard) and `anomaly_basis`
(`level` | `change`: score period-over-period changes, for smooth level series) are per spec.

### Daily/weekly frequency and seasonality (WS4b; opt-in, monthly output byte-identical)
- **Frequency** `frequency: day | week | month` (config); `run.frequency`, `series[].frequency` and
  `forecast.horizon_unit` report `day`/`week`/`month`. Partial-period logic is unchanged (a period is complete once the
  next one has started at as_of; at midnight the previous day is the last complete one). Gaps stay NaN; duplicates are
  dropped by the core validator (CTA station data has 10). `sum` = sum of values with a `min_count` guard (a week with
  a missing day is NaN, not a low week). Derived values: `rolling: k` (trailing mean, all k periods required) and
  `ratio_lag: L` (ratio to L periods earlier), e.g. the 28-day total against the weekday-aligned 28 days a year earlier.
- **Season classes** (`core.series.season_classes`): day of week for days, week of year for weeks, month for months.
  `calendar: weekday_us_holidays` maps New Year's, Memorial, Independence, Labor, Thanksgiving and Christmas Day
  (observed on the nearest weekday) to the Sunday class. On CTA's own `day_type` column it agrees on 99.91 % of 9,312
  days (adapter check `<source>_calendar`, config `calendar_check`). `anomaly_calendar: weekday_us_special_days` adds
  class 7 (those six plus MLK Day, Presidents' Day, the Friday after Thanksgiving and 24–31 Dec) for anomaly baselines
  only.
- **Seasonal forecasting** (`core.forecast`, taken when a spec sets `forecast_models` or `seasonal_period`):
  - Candidates: `naive, moving_average, holt_damped, drift, theta, seasonal_naive, seasonal_naive_yearly`
    (364 d / 52 w / 12 m), `calendar_naive` (the latest period of the same class), `holt_winters` (additive damped
    ETS(A,Ad,A) on log1p for counts, grid `HW_*`, one causal pass over all origins) and `holt_winters_calendar`
    (season index = calendar class).
  - Gaps are filled for fitting (same class) and never scored.
  - MASE uses the in-sample lag-m naive scale. `backtest.naive_mase` is the **seasonal-naive** benchmark
    (`benchmark: seasonal_naive`), so the risk/signal/early-warning "beats naive" skill is measured against seasonal
    naive. `random_walk_mase` and `seasonal_naive_mase` are reported too, plus `rmse`.
  - Selection is unchanged: lowest backtest MASE on the origins before the issue time. Intervals are the same conformal
    order statistics.
  - `rolling_evaluation(series, cfg, ...)` replays that whole served procedure at many issue times.
  - Scenarios rerun the selected model through `forecast.rerun` and word the horizon in the series' own unit.
- **Seasonal anomaly baseline** `anomaly_basis: seasonal`: a point is scored against the previous
  `anomaly_baseline_months` points *of its class* (Mondays with Mondays). This is the only hook in
  `core/anomalies.py`: one additive branch.
- **Other hooks outside WS4b ownership** (additive, identical for monthly/weekly):
  - `risk.recent_from` parses the period with the series' own frequency (it was hard-coded to weeks for any non-month).
  - `pipeline` reports `run.frequency = "day"`.
- **Known cosmetic gaps (WS4a modules):**
  - `signals.period_age_days` treats a daily date ending in `-01` as a month and other daily dates as weeks, so
    `freshness_days` of daily signals can be off by up to 6 days.
  - Risk titles print `detected_at[:7]` (a month) for daily anomalies.
- **Evaluation** (`scripts/evaluate_domains.py`, `domains/generic/evaluation.py`):
  - `plain_config` is the generic baseline: same data and thresholds, every seasonal option removed.
  - The seasonal confirmation rule is a sustained (two consecutive weeks) fall of the weekday-aligned year-over-year
    change of the 7-day total beyond 2σ√k within 4 weeks.
  - Units are (replay, entity), with a warned-status flip rate.
  - Results: [SECOND_DOMAIN_CASE_STUDY.md](SECOND_DOMAIN_CASE_STUDY.md).
- **us-unemployment** keeps its v1.2 config. Its synthetic golden (`tests/domains/test_generic_golden.py`, written
  before this work) is byte-identical.
  - A rolling replay of the served forecast over 2005–2026 (monthly issue times) gives:
    - v1.2 trio: median relative MAE vs naive 0.993, below 1 on 10 of 17 series, 80 % coverage 0.69.
    - With drift and theta added: 0.986, again 10 of 17, coverage 0.69.
  - That gain is too small to change what is served. The monthly forecasts are still essentially naive.

### Early-warning decision (`early_warning_level`, policy `ewl-1.0.0`)
One decision per situation: `series:<id>` (trend, change point, anomalies, forecast and risks naming the series) or
`entity:<type>:<name>` (risks/anomalies not about one series). Decisions are emitted for situations with > 0 points
or a skipped stage; all-zero situations are NO_ACTION by construction and listed in the batch record
(`situations_without_points`). Policy table (defaults; per-domain overrides in brackets):

| evidence | points |
|---|---|
| risk (score s; bands low 5 / medium 15 / high 30 / critical 50) | piecewise-linear: 0 at 0, 1 / 3 / 5 / 7 at the band edges, extrapolated, cap 8 |
| anomaly (its severity measure: \|robust z\| 3.5 / 5 / 7 / 10; IsolationForest s 0.5 / 0.66 / 0.72 / 0.80; feedback lift 1.0 / 1.2 / 1.5 / 2.0) | same map |
| anomaly outside the last 3 complete periods, or favourable when `ewl_anomaly_direction = adverse` [movie: `any`] | capped at 2.5 |
| suppressed item (min-volume, resolution, inactive rater, operator dismissal) | 0 |
| adverse trend (not flat) | 1 + 1.5 (1 − q) |
| adverse change point | 1 + 1.5 (1 − p) |
| adverse forecast (beats naive, \|rel. change\| ≥ 10 %) | 1 + 1.5 · min(1, \|change\|) · skill |
| aggregation | P = min(8, max(points) + c · (#stages ≥ 1 point among trend/anomaly/forecast − 1)), c = 1 [movie: 0] |
| levels | NO_ACTION < 1 ≤ MONITOR < 3 ≤ WARNING < 5 ≤ URGENT_ACTION |
| option scores / confidence | cumulative ramps ±1 point around each threshold; `margin` = (s1 − s2)/(s1 + s2) |
| abstention | every evidence item is a skipped stage (trend/forecast lacked history) |

State (nested by stage, as the console reads it): `signal{strength,n_evidence}`, `trend{direction,q,points}`,
`anomaly{score,severity,suppressed,points}`, `forecast{direction,relative_change,adverse_direction,points}`,
`risk{score,level,points}`, plus `components[]`, `skipped_stages`, `points`, `corroborating_stages`, `policy`.
Monotonicity is by construction (every map is non-decreasing; max and the stage count only grow) and is tested
with randomised property tests.

**Warnings are downstream**: raised only for WARNING/URGENT_ACTION, carrying `decision_id` and
`early_warning_level`. Movie (`warning_mode = components`): one warning per qualifying component with its v1.1 key
and severity; since a component with severity ≥ medium is exactly one with ≥ 3 points, this reproduces the v1.1
warning set. Generic (`warning_mode = situation`): one warning per situation, key `warning:<situation>`, severity =
max(level floor medium/high, qualifying component severities), confidence = the decision's margin.

### Generic adapter
Any long CSV + YAML (timestamp, entity, value, optional entity type / event type / group columns, series specs,
adverse direction, thresholds overriding `CoreConfig`, declared `impact_weights` labelled "declared"; unknown keys are
rejected). Leakage: rows with `timestamp + availability_lag_days > as_of` are unpublished at as_of and excluded
(counted, info check); the core drops rows after as_of. Adapter checks: value range, gaps per entity. Freshness:
`fresh = as_of − last usable observation ≤ max_lag_days`; default as_of = wall clock, so an old file becomes a
stale source. Generic risks: `adverse_trend` (q ≤ 0.10), `adverse_forecast` (skill > 0, ≥ 10 % move; likelihood =
share of steps whose 80 % interval lies beyond the last value), `adverse_anomaly` (recent, adverse); impact
declared or |relative change| / 0.25.

**us-unemployment**: FRED `UNRATE` + `<ST>UR` for CA TX FL NY PA IL OH GA NC MI NJ WA (BLS, public domain),
fetched without a key; raw files, long CSV and `provenance.json` (URL, licence note, SHA-256 per file and of the
long CSV) under `data/raw/domains/us-unemployment/` (gitignored by `data/raw/*`). Series: rate per area (level,
`anomaly_basis: change`, resolution 0.1) and the unweighted mean per Census region; adverse direction up.
Availability lag 38 days for the US (first-Friday release) and 55 days for states. FRED has no October-2025 value
(the 2025 federal shutdown), which the gap check reports on current data (quality 0.94).

### Real runs (2026-09-24, `now` fixed)
| run | status | warnings | ewl decisions | notes |
|---|---|---|---|---|
| movie, default (as of 2018-09-24) | watch | 1: `risk:audience_lapse:all` medium | 13 (12 MONITOR, 1 WARNING, margin 0.98) | 55 decisions (42 v1.1 + 13); 1.05 s |
| movie, 2017-07-01 | alert | 1: Horror share spike 2017-05, high | 18 (17 MONITOR, 1 URGENT_ACTION, 5.57 pts) | 62 decisions; 0.97 s |
| us-unemployment, default (data to 2026-08) | alert | 6 (FL critical: 3.0 % → 4.1 % over two years; WA, West, South, NY high; TX medium) | 9 | 0.21 s |
| us-unemployment, 2008-06-01 (sees Apr 2008) | alert | 10 (IL, NY critical; FL forecast +21 %; US high) | 11 (6 URGENT, 4 WARNING) | broad adverse rise before the peak |
| us-unemployment, 2020-05-01 (sees Mar 2020) | alert | 10 (8 critical: March jumps of +1.0…+2.2 pp in CA FL IL TX WA, US +0.9 pp) | 13 (9 URGENT) | the April spike is not yet published at 2020-05-01 |

### Evaluation (`experiments/platform-eval-20260924T051637Z`, h = 6 periods)
| metric | movie | us-unemployment |
|---|---|---|
| forecast backtest (selected on origins 1–12, scored 13–24) | 21 series; median MAE 47.8, RMSE 59.1 ratings; MASE 0.93 vs naive 1.15; beats naive 21/21; coverage80 0.95 | 17 series; median MAE 0.145, RMSE 0.177 pp; MASE 1.52 vs naive 1.41; beats naive 3/17 (naive selected on 8); coverage80 0.61 |
| warning precision / FPR | lapse: 18/18 warned replays confirmed (precision 1.0), but the base rate is 1.0 (every observable replay's flagged group lapsed ≥ 70 %), so FPR is undefined and precision is uninformative; genre decline: never warned in 24 replays, 20 of 342 share-series replays had an adverse move (recall 0) | 96 replays × 17 series: precision 0.36 at base rate 0.30 (lift 1.2), FPR 0.39, recall 0.52; 2006–10: precision 0.44, FPR 0.44, recall 0.65; 2019–21: precision 0.11, FPR 0.32, recall 0.15 |
| ewl flip rate (consecutive replays) | 0.23 of 455 situation pairs (49 escalations, 57 de-escalations), mostly MONITOR ↔ NO_ACTION; WARNING-boundary flips 0.026 | 2006–10: 0.13 (boundary 0.053); 2019–21: 0.22 (boundary 0.106) |

Confirmation rules. Generic and movie genre decline: the series moves in its adverse direction by more than
2 σ √k within k ≤ 6 periods of the replay's last complete period (σ = 1.4826 · MAD of the 24 prior changes,
floored at the resolution). Movie lapse: the realised 180-day lapse share of the users flagged at the replay
(P ≥ 0.7) is ≥ 0.7. Excluded (outcome not observable): movie series anomalies (direction-agnostic, no adverse
condition), rater/manipulation (no labels), live feedback/rejection (not replayable), model risks (skipped in
replays), data quality/staleness.

Weak results, reported as they are:
- Unemployment warnings lag turning points: the 24-month trend stays "up" long after a peak, so 2010 and
  mid-2020 (recoveries) are full of unconfirmed warnings (17 of 17 series warned through 2009–10). Recall in
  2019–21 is 0.15 because nothing trended adversely before COVID: every replay from 2019-11 to 2020-03 had at most
  one warning while the April-2020 jump (within 6 months of each) confirmed an adverse move on all 17 series.
- Unemployment forecasts: the selected model beats naive on 3 of 17 series, *is* naive on 8 (selected on the
  first half of the origins) and is worse on 6; the 80 % intervals under-cover (0.61) on the scored origins.
- Replays use the current FRED vintage, not real-time vintages (ALFRED); revisions are not modelled.

### Contract deviations and intentional behaviour changes (movie)
- `DomainAdapter.load(as_of, now, config)` takes the config; adapters provide `default_config()`.
- Movie validation stays in the adapter (`core_checks = False`): adding core checks would change the v1.1 quality
  score.
- `run.pipeline_version` stays `intel-1.1.0` for the movie domain (tests and stored runs key on it); the core
  version is in `run.core_version`.
- `decisions` gains the `early_warning_level` decisions and `decision_batches` the `early_warning` batch, so
  `summary.counts.decisions` / `decisions_abstained` grow; `POLICY_VERSIONS` gains `early_warning_level`.
- Every batch's `state_hash` changes: the hashed snapshot now carries the additive fields.
- Warnings add `decision_id`, `early_warning_level`, `domain`; anomalies add `observed_value`, `expected_value`,
  `anomaly_score`, `confidence(_kind)`, `entity_id`, `source`, `adverse`, `severity_basis`, `baseline_points`,
  `anomaly_basis`, `level`, `relative_deviation`; trends, risks, signals, forecasts, series and sources gain the
  §2 fields (plus `series_id`/`severity_basis` on risks, `license/url/checksum` on the MovieLens source and
  `sla_days/staleness_days/stale_response` on the app source).
- The types in `types.py` are the typed view of the dict records (round-trip tested); the engines still build dicts.
- The golden test (`tests/core/test_movie_golden.py`) compares four synthetic variants (CI) and the two real-data
  runs (local) against goldens written by the pre-migration code; it filters exactly the changes listed above.
