# JEV Intelligence — decision & early-warning layer

JEV began as a hybrid movie recommender. The intelligence layer sits on top of it and turns the
platform's changing data (the MovieLens rating stream, live app activity, and the recommender's own
evaluation results) into evidence-backed signals, early warnings, bounded decisions and action plans
for the people who run the platform.

The recommender is unchanged and still serves members. The intelligence layer serves operators
(admin accounts) through `/intel/*` endpoints and the Intelligence console in the web app.

> Status: this document is the design contract the implementation follows. Sections marked
> *implementation notes* are filled in as each stage lands.

## 1. What "JEV" decides, and what it does not

"JEV" is the decision layer of this project, not an external LLM. Decisions are **bounded and typed**:
each one is a fixed question with a declared answer type (`boolean`, `choice`, `score`), a fixed
option set and a versioned policy. Normal code does all arithmetic, aggregation, statistics and
validation. The decision policy only combines the evidence those computations produce.

Every decision records:
- the structured input state it saw (a snapshot of the numbers it used),
- the answer and a score for every option,
- a confidence **with its kind**:
  - `probability`: comes from a probabilistic computation, such as a bootstrap or a calibrated classifier,
  - `margin`: a normalised evidence margin between the best and second-best option. This is *not* a probability,
  - `rule`: a deterministic threshold rule, with confidence 1 when the rule fires. It is reported so it is never dressed up as a probability,
- a rationale (human-readable reasons), the evidence it relied on, and the policy version,
- `abstained: true` plus a `fallback_reason` when the data is insufficient. JEV abstains rather than guesses.

No confidence value anywhere in the system is random or hard-coded.

## 2. Data sources and the analysis clock

| Source | Kind | Contents | Freshness expectation |
|---|---|---|---|
| `movielens` | static snapshot | 100,836 ratings, 1996-03 → 2018-09 | none (archival). Its age is reported, not alarmed |
| `app` | live | ratings, watches, favourites, recommendation feedback and served recommendations from the app DB | events expected daily *if* the app has active users |
| `model` | artefact | active model manifest, latest experiment metrics, per-user NDCG | model age and evaluation results |

The pipeline runs **as of** a timestamp. The default is the last MovieLens event. Every stage uses only
data at or before `as_of`, so a run is leak-free and can be *replayed* at any past date. For example,
running as of 2017-07-01 lets you watch the system react to the Q2-2017 volume spike. Live app
signals use the wall clock (`now`). The final month is excluded from baselines when it is incomplete
at `as_of`, and is flagged `partial` in series output.

## 3. Pipeline

```
INGEST → VALIDATE → UNDERSTAND → DETECT → PREDICT → ASSESS → DECIDE → RECOMMEND/ACT → EXPLAIN
  │         │           │            │         │         │        │          │
 sources  quality     series,     trends,  forecasts,  risks  decisions  warnings,
          checks,     signals     change-  lapse model               action plan
          freshness               points,
                                  anomalies
FEEDBACK (operator verdicts on decisions / warnings / actions) → evaluation of the layer itself
```

Code lives in `ml/jev_ml/intel/` as pure functions over pandas/numpy with no web or DB imports.
`run_pipeline(inputs, config) -> PipelineResult` is the single entry point. The API gathers inputs
from the DB and files, calls it, and persists the result.

### Series (UNDERSTAND)
Monthly series (period start dates, `YYYY-MM-01`) are built from ratings ≤ `as_of`:
- `volume:all`: ratings per month.
- `volume:genre:<Genre>`: ratings per month on films of that genre. A film counts once per genre.
- `share:genre:<Genre>`: genre volume ÷ total volume.
- `rating:genre:<Genre>`: mean rating per month (only months with ≥ `min_count` ratings).
- `active_users:all`: distinct raters per month.

Genres come from the processed catalogue. `(no genres listed)` is excluded.

## 4. Result contract (`PipelineResult.to_dict()`)

All timestamps are ISO-8601 UTC strings. All ids are stable strings, so re-running the same `as_of`
on the same data yields the same ids (deterministic seeds everywhere).

```jsonc
{
  "run": {
    "pipeline_version": "intel-1.0.0",
    "as_of": "2018-09-24T14:27:30Z",
    "now": "2026-09-23T12:00:00Z",
    "data_version": "ml-latest-small-31a303aa-wd-2bb80720a955",
    "model_version": "jev-20260923T100141Z-bbb2e4c9",   // null when no model
    "stage_ms": {"ingest": 12.3, "validate": 4.1, "...": 0},
    "config": { /* IntelConfig as dict */ }
  },
  "data": {
    "sources": [{
      "source": "movielens", "kind": "static_snapshot", "rows": 100836,
      "first_event": "...", "last_event": "...",
      "age_days": 2921.4,                 // now - last_event
      "lag_days": 0.0,                    // as_of - last_event
      "expected_update": null,            // or "daily"
      "fresh": null                       // null = no SLA, else bool
    }],
    "quality": {
      "score": 0.97,                      // share of weighted checks passed (0..1)
      "checks": [{"name": "rating_range", "passed": true, "value": 0, "threshold": 0,
                  "severity": "error", "detail": "0 ratings outside [0.5, 5.0]"}]
    }
  },
  "series": [{"id": "volume:genre:Drama", "metric": "volume", "entity": "Drama",
              "entity_type": "genre", "unit": "ratings/month",
              "points": [{"t": "2018-08-01", "v": 812, "partial": false}]}],   // last 120 months
  "signals": [{
    "id": "sig-…", "dedup_key": "trend:share:genre:Drama", "kind": "trend|anomaly|change_point|forecast|quality|live|model",
    "entity_type": "genre|platform|user|model|source", "entity": "Drama",
    "title": "Drama share rising", "value": 0.012, "unit": "share/month",
    "strength": 0.63,                     // 0..1, defined per kind (documented in code)
    "direction": "up|down|flat|null",
    "source": "movielens", "observed_at": "...", "window": "2016-10..2018-09",
    "freshness_days": 0.0,
    "evidence": [Evidence]
  }],
  "trends": [{
    "id": "trend-…", "series_id": "share:genre:Drama", "entity": "Drama", "metric": "share",
    "window": {"start": "2016-10-01", "end": "2018-08-01", "months": 24},
    "direction": "up|down|flat",
    "slope": 0.0011, "slope_ci": [0.0004, 0.0019],      // Theil–Sen, 95 %
    "change_rate_pct_per_month": 0.9,                   // slope / window mean
    "kendall_tau": 0.41, "p_value": 0.004,              // Mann–Kendall
    "evidence_strength": 0.996,                         // 1 - p (NOT a probability of the trend)
    "recent_mean": 0.21, "prior_mean": 0.18, "ratio": 1.17,   // vs the preceding window of equal length
    "change_point": {"date": "2017-04-01", "before_mean": 0.17, "after_mean": 0.22, "p_value": 0.01} // or null
  }],
  "anomalies": [{
    "id": "anom-…", "kind": "series_spike|series_drop|rater_behaviour|live_feedback",
    "entity_type": "genre|platform|user", "entity": "Drama", "series_id": "volume:genre:Drama",
    "detected_at": "2017-05-01", "value": 1900, "baseline": 420, "deviation": 1480,
    "score": 6.1,                         // robust z (MAD) for series; isolation-forest percentile for raters
    "method": "robust_z|isolation_forest|rate_test",
    "severity": "low|medium|high|critical",
    "suppressed": false, "suppression_reason": null,   // min-volume guard, prior dismissal, …
    "features": {},                        // rater anomalies: the behavioural features
    "evidence": [Evidence]
  }],
  "predictions": {
    "forecasts": [{
      "id": "fc-…", "series_id": "volume:all", "entity": "all", "metric": "volume",
      "model": "holt_damped|moving_average|naive", "model_version": "fc-1.0.0-<hash>",
      "horizon_months": 6, "issued_at": "<as_of>", "features_used": ["log1p(volume) history, 36 m"],
      "points": [{"t": "2018-10-01", "mean": 1100, "lo80": 600, "hi80": 2100}],
      "backtest": {"origins": 24, "mase": 0.91, "smape": 0.42, "mae": 310,
                   "coverage80": 0.79, "naive_mase": 1.0}     // rolling-origin, ≤ as_of only
    }],
    "lapse": {                            // P(user rates nothing in the next horizon_days)
      "model_version": "lapse-1.0.0-<hash>", "horizon_days": 180,
      "features": ["days_since_last", "..."],
      "metrics": {"auc": 0.0, "brier": 0.0, "ece": 0.0, "base_rate": 0.0,
                  "baseline_auc": 0.0, "n_train": 0, "n_test": 0, "train_cutoffs": [], "test_cutoffs": []},
      "calibration": [{"bin": "0.0-0.1", "predicted": 0.05, "observed": 0.04, "n": 120}],
      "population": {"n_scored": 0, "expected_lapses": 0.0, "high_risk": 0, "threshold": 0.7},
      "top": [{"user_id": 1, "p": 0.93, "features": {}}],
      "status": "ok|insufficient_data", "detail": null
    }
  },
  "risks": [{
    "id": "risk-…", "kind": "genre_demand_decline|audience_lapse|rating_manipulation|data_quality|data_staleness|model_staleness|model_quality|recommendation_rejection",
    "title": "…", "entity_type": "…", "entity": "…",
    "likelihood": 0.7, "impact": 0.4, "exposure": 0.12, "confidence": 0.8, "data_quality": 0.97,
    "score": 28.0,                        // 0..100 = 100 * likelihood * impact, shrunk by (confidence * data_quality)
    "level": "low|medium|high|critical",
    "factors": [{"name": "…", "value": 0.0, "weight": 0.0, "contribution": 0.0, "detail": "…"}],
    "evidence": [Evidence], "recommended_response": "…"
  }],
  "decisions": [{
    "id": "dec-…", "key": "retrain_model", "spec_id": "retrain_model", "policy_version": "retrain-1.0.0",
    "question": "Should the recommendation model be retrained now?",
    "kind": "boolean|choice|score", "options": ["yes", "no"],
    "answer": "no", "option_scores": {"yes": 0.31, "no": 0.69},
    "confidence": 0.38, "confidence_kind": "probability|margin|rule",
    "state": { /* exact inputs the policy used */ },
    "rationale": ["…"], "evidence": [Evidence],
    "abstained": false, "fallback_reason": null,
    "entity_type": "model", "entity": "jev-…"
  }],
  "warnings": [{
    "key": "risk:genre_demand_decline:Horror",   // dedup key: one open warning per key
    "title": "…", "description": "…",
    "severity": "low|medium|high|critical", "confidence": 0.8,
    "trigger": {"rule": "risk.level>=medium", "condition": "…", "observed": 0.0, "threshold": 0.0},
    "evidence": [Evidence], "recommended_action": "…",
    "source": {"type": "risk|anomaly|decision", "id": "…"}
  }],
  "actions": [{
    "id": "act-…", "title": "…", "priority": "P1|P2|P3", "priority_score": 0.0,
    "reason": "…", "expected_impact": "…", "effort": "low|medium|high", "risk": "…",
    "evidence": [Evidence], "next_step": "…", "source": {"type": "decision|warning|risk", "id": "…"}
  }],
  "summary": {"status": "nominal|watch|alert", "headline": "…",
              "counts": {"signals": 0, "trends_up": 0, "trends_down": 0, "anomalies": 0,
                         "risks_high": 0, "warnings": 0, "decisions": 0, "actions": 0}}
}
```

`Evidence = {"kind": "metric|series|record|test|model", "label": str, "value": number|string|null, "detail": str|null, "ref": str|null}`
`ref` points at another object id or a `series_id`, so the UI can link evidence to the chart it came from.

### Scenario contract (`run_scenario`)
Input:
```jsonc
{"series_id": "volume:genre:Drama", "horizon_months": 12, "as_of": null,
 "scenarios": [{"name": "Trend continues", "kind": "continue"},
               {"name": "Trend slows", "kind": "slow", "trend_multiplier": 0.5},
               {"name": "Trend reverses", "kind": "reverse", "trend_multiplier": -1.0},
               {"name": "Shock −20 %", "kind": "shock", "level_shift_pct": -20, "shock_month": 1}]}
```
Output:
```jsonc
{"series_id": "…", "as_of": "…", "model": "…", "baseline": {"points": [{"t","mean","lo80","hi80"}], "total": 0},
 "history": [{"t","v"}],
 "scenarios": [{"name": "…", "kind": "…", "assumptions": ["…"],
                "points": [{"t","mean","lo80","hi80"}], "total": 0,
                "delta_vs_baseline": 0, "delta_pct": 0.0, "end_level": 0}],
 "comparison": [{"name": "…", "total": 0, "delta_pct": 0.0, "end_level": 0, "rank": 1}],
 "uncertainty_note": "80 % intervals from rolling-origin residuals; scenarios shift the trend, not the noise."}
```

## 5. Persistence

Everything added lives in Alembic migration `0002_intelligence`:

| table | purpose |
|---|---|
| `intel_runs` | one row per pipeline run: as_of, status, timings, versions, summary and the full result JSON |
| `intel_warnings` | early warnings with lifecycle: `new → acknowledged → investigating → resolved` or `dismissed`. Unique open warning per `key`. Tracks `first_seen_run_id`, `last_seen_run_id` and `occurrences` |
| `intel_warning_events` | audit trail of every status change (who, when, from → to, note) |
| `intel_decisions` | every decision per run (flattened columns + state/evidence JSON) |
| `intel_scenarios` | saved what-if analyses (input + output JSON) |
| `intel_feedback` | operator verdicts on decisions, warnings, actions and predictions |

Warning lifecycle rules:
- A candidate whose key already has an open warning updates `last_seen`, `occurrences`, `severity` and `evidence`. It never duplicates.
- A key *dismissed* as a false positive stays suppressed for `suppress_days` unless severity escalates.
- A resolved key that fires again opens a new warning, with `reopened_from` pointing at the old one.

## 6. API (`/intel/*`, admin only)

| Method | Path | Notes |
|---|---|---|
| GET | `/intel/status` | overview: latest run summary, freshness, quality, model status, open warning counts, health |
| GET/POST | `/intel/runs` | list runs / trigger a run `{as_of?: ISO date}`. The run executes synchronously and is recorded |
| GET | `/intel/runs/{id}` | one run (without the full result) |
| GET | `/intel/signals` `/intel/trends` `/intel/anomalies` `/intel/risks` `/intel/actions` | from the latest (or `?run_id=`) run; filters + `limit/offset`; response `{items, total, run_id, as_of}` |
| GET | `/intel/predictions` | forecasts + lapse model |
| GET | `/intel/series/{series_id}` | a series with its trend, anomalies and forecast (for charts) |
| GET | `/intel/decisions` `/intel/decisions/{id}` | decision log across runs, with feedback |
| GET | `/intel/warnings` `/intel/warnings/{id}` | filter by status/severity; detail includes history |
| PATCH | `/intel/warnings/{id}` | `{status, note?}`; transitions validated |
| POST/GET | `/intel/scenarios` | run (and optionally save) a what-if / list saved |
| POST/GET | `/intel/feedback` | record a verdict / list with summary (decision precision, warning precision) |
| GET | `/intel/evaluation` | latest offline evaluation report of the intelligence layer |
| GET | `/admin/metrics` | in-process counters: requests, errors, pipeline runs/failures and duration, decision counts, warning counts |

### Response shapes

```jsonc
// Page<T>, used by list endpoints built on a run
{"items": [T], "total": 0, "limit": 50, "offset": 0, "run_id": "uuid|null", "as_of": "ISO|null"}

// Run
{"id": 1, "run_id": "uuid", "trigger": "startup|manual|script|schedule", "status": "running|succeeded|failed",
 "as_of": "ISO", "started_at": "ISO", "finished_at": "ISO|null", "duration_ms": 0.0,
 "pipeline_version": "…", "data_version": "…", "model_version": "…|null",
 "summary": {/* result.summary */}, "stage_ms": {}, "error": null}

// GET /intel/status
{"latest_run": Run|null, "summary": {}|null, "data": {/* result.data */}|null,
 "model": {"version": "…", "trained_at": "ISO", "age_days": 0.0, "dataset_version": "…"}|null,
 "warnings_open": {"total": 0, "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0}},
 "recent_decisions": [Decision], "top_signals": [Signal], "top_risks": [Risk],
 "confidence_histogram": [{"bin": "0.0-0.2", "n": 0}],          // over the latest run's decisions
 "health": {"database": "ok|error", "cache": "ok|error", "model": "ok|unavailable", "pipeline": "ok|failed|never_run"}}

// Warning (DB-backed)
{"id": 1, "key": "…", "title": "…", "description": "…", "severity": "…", "confidence": 0.8,
 "status": "new|acknowledged|investigating|resolved|dismissed",
 "trigger": {}, "evidence": [Evidence], "recommended_action": "…", "source": {},
 "detected_at": "ISO", "last_seen_at": "ISO", "updated_at": "ISO", "occurrences": 1,
 "first_seen_run_id": "uuid", "last_seen_run_id": "uuid", "reopened_from": null,
 "history": [{"from_status": null, "to_status": "new", "note": "…", "actor": "system|<email>", "at": "ISO"}]}  // detail only
// PATCH body: {"status": "acknowledged", "note": "…"}. Allowed: new→{acknowledged,investigating,resolved,dismissed},
// acknowledged→{investigating,resolved,dismissed}, investigating→{resolved,dismissed}; resolved/dismissed are terminal.

// Decision (DB-backed) = contract Decision + {"db_id": 1, "run_id": "uuid", "as_of": "ISO", "created_at": "ISO",
//   "feedback": {"correct": 0, "incorrect": 0}}

// POST /intel/feedback
{"target_type": "decision|warning|action|prediction", "target_id": "string id", "verdict":
 "correct|incorrect|useful|not_useful|false_positive", "note": "≤1000 chars|null", "outcome": "≤500 chars|null"}
// → {"id", "target_type", "target_id", "verdict", "note", "outcome", "actor", "created_at"}
// GET /intel/feedback → {"items": [...], "total": 0, "summary": {
//   "decision": {"correct": 0, "incorrect": 0, "accuracy": null|0.0},
//   "warning": {"useful": 0, "not_useful": 0, "false_positive": 0, "precision": null|0.0},
//   "action": {"useful": 0, "not_useful": 0}, "prediction": {"correct": 0, "incorrect": 0}}}

// GET /intel/predictions → {"run_id", "as_of", "forecasts": [Forecast], "lapse": Lapse}
// GET /intel/series/{series_id} → {"run_id", "as_of", "series": Series, "trend": Trend|null,
//                                  "anomalies": [Anomaly], "forecast": Forecast|null}
// POST /intel/scenarios body = scenario input + {"save": bool, "title": "…"} → scenario output + {"id": int|null}
// GET /intel/scenarios → {"items": [{"id", "title", "series_id", "created_at", "input", "output"}], "total"}

// GET /intel/evaluation → {"available": bool, "run_dir": "…", "report": EvaluationReport|null}
EvaluationReport = {
  "created_at": "ISO", "pipeline_version": "…", "data_version": "…", "as_of": "ISO",
  "forecast": {"per_series": [{"series_id", "model", "mase", "smape", "mae", "coverage80", "naive_mase", "origins"}],
               "summary": {"n_series": 0, "median_mase": 0.0, "share_beating_naive": 0.0, "mean_coverage80": 0.0}},
  "lapse": {"metrics": {/* as in predictions.lapse.metrics */}, "calibration": [/* bins */],
            "baselines": [{"name": "recency_rule", "auc": 0.0, "brier": 0.0}]},
  "anomaly": {"injection": {"protocol": "…", "n_genuine": 0, "n_injected": 0,
                            "attack_types": [{"type": "random|average|bandwagon", "n": 0, "precision": 0.0, "recall": 0.0,
                                              "f1": 0.0, "auc": 0.0, "baseline_precision": 0.0, "baseline_recall": 0.0}]},
              "series": {"protocol": "…", "detection_rate": 0.0, "false_alarm_rate": 0.0, "n_trials": 0}},
  "change_point": {"protocol": "…", "detection_rate": 0.0, "false_alarm_rate": 0.0, "mean_abs_location_error_months": 0.0, "n_trials": 0},
  "latency": {"pipeline_ms_mean": 0.0, "pipeline_ms_p95": 0.0, "n_runs": 0, "stage_ms": {}},
  "notes": ["…"]
}
```

## 7. Implementation notes (ML)

Code: `ml/jev_ml/intel/` (`config, common, ingest, series, trends, anomalies, forecast, lapse,
modelstats, risk, decisions, warnings, actions, signals, scenario, pipeline, evaluation`). Every
threshold lives in `IntelConfig` (embedded in `run.config`); each default has a one-line reason in
`config.py`. CLIs: `uv run python scripts/run_intelligence.py [--as-of YYYY-MM-DD] [--out f.json]` and
`uv run python scripts/evaluate_intelligence.py`. Tests: `tests/intel/`.

### Public API
- `PipelineInputs(interactions, movies, app_ratings=None, app_feedback=None, app_served=None,
  model_manifest=None, experiment=None, per_user_ndcg=None, dataset_meta=None, as_of=None, now=None,
  suppressed_keys={}, data_version=None)`. Frame timestamps are UNIX seconds (datetimes are accepted in
  the app frames). `suppressed_keys` maps a dismissed warning/anomaly key to either the severity it had
  when it was dismissed (a later *higher* severity is an escalation and fires again) or a free-text
  reason (suppressed at every severity).
- `load_default_inputs(as_of=None, now=None, processed_dir=None, models_dir=None, experiments_dir=None)`
  reads the processed data, the active model manifest and its experiment run (`metrics.json`,
  `per_user_ndcg10.json`).
- `run_pipeline(inputs, config=None) -> PipelineResult`. Use `.to_dict()` (strict JSON: no NaN, no numpy
  types) and `.summary`. The result also keeps `series_objects`, `forecast_states` and `prepared` in
  memory for scenarios.
- `run_scenario(result_or_inputs, spec, config=None) -> dict`. It raises `ValueError` on invalid specs.
- `jev_ml.intel.evaluation.run_evaluation(inputs=None, config=None, experiments_dir=None,
  n_latency_runs=5) -> (report, run_dir)`. `evaluate(...)` builds the report without writing it.

### Additions to the contract (fields added, none renamed)
- `run.last_complete_month`. `data.excluded_after_as_of`. `data.live` (live-feedback test status).
  `data.sources[].detail`. Top-level `diagnostics` (list of skipped stages with reasons).
- `data.sources` also lists a `model` source (`kind: "artefact"`). `quality.checks[].source`.
- `run.stage_ms` keys: `ingest_validate, series, trends, anomalies, forecast, lapse, risk, decide,
  recommend, explain, total`. Frames arrive already loaded, so ingest and validation are timed as a
  single stage.
- `trends[]`: `q_value` (Benjamini–Hochberg), `n_points`, `entity_type`. `change_point.null`
  (`permutation|ar1`) and `change_point.phi`.
- `anomalies[]`: `dedup_key` (the warning key) and `metric`. For raters, `detected_at` is the as_of
  timestamp, `value` is the IsolationForest score s, and `score` is its population percentile (0–100).
- `forecasts[]`: `entity_type`, `model_params`, `backtest_all_models`. Points start at the first
  month after the last *complete* month, so the partial as_of month is itself forecast.
- `lapse.metrics`: `train_base_rate`, `baseline_brier`, `base_rate_brier`, `calibration_method`.
  `lapse.population.mean_p`. When `status` is `insufficient_data`, `metrics` and `population` are `null`.
- `decisions[]`: an abstained decision has `answer: null`, `confidence: null` and empty
  `option_scores`.
- `warnings[]`: `confidence_kind` (always `"margin"`: a normalised evidence strength, never a
  probability), `entity_type`, `entity`.
- `actions[]`: `effort_basis: "declared estimate per action type"`.
- `summary.counts`: `anomalies_suppressed`, `decisions_abstained`. `counts.anomalies` counts only
  non-suppressed anomalies.
- Scenario output: `aggregate` (`sum` for counts, `mean` for share/rating series), `horizon_months`,
  and `trend: {source: model|theil_sen, slope_per_month, slope_ci}`. Scenario input accepts
  `trend_source: "auto"|"model"` (default `auto`). Scenario kinds: `continue|slow|reverse|shock|custom`.
- EvaluationReport: `forecast.per_series[].selection_origins`, `forecast.summary.median_naive_mase`,
  `lapse.status`, `anomaly.injection.baseline_rule`, `attack_types[].{trials, baseline_f1,
  baseline_auc}`, `anomaly.series.{detection_by_magnitude, n_points}`,
  `change_point.detection_by_magnitude`, `config`.

### Methods and thresholds
- **Validation.** Checks are schema, null rate, rating range, 0.5-star grid, unknown movie ids,
  timestamp range (before 1995 or after the wall clock), duplicate (user, movie, ts) rows,
  "frozen" users (all ratings in one second, > 5 % fails), catalogue genres, and rows after as_of
  (info, leakage guard). Rows after as_of are dropped *before* any check runs. Invalid rows are
  dropped, duplicates are kept once, and off-grid ratings are kept (warning). Score = weighted share
  passed (error 3, warning 1, info 0.5).
- **Freshness.** `movielens`: static snapshot, `fresh: null`. `app`: expected daily; fresh when its last
  event is less than 2 days old, `null` with a detail when there are no events. `model`: artefact age.
  If the active model was trained on data after as_of (a replay), model risks and model decisions are
  skipped or abstained with that reason, because replaying them would leak the future.
- **Trends.** 24 complete months; Mann–Kendall (Kendall tau-b vs time) and Theil–Sen with a 95 % CI.
  `up`/`down` requires p < 0.05 *and* a CI that excludes 0. About 55 series are tested per run, so
  risks and decisions also need BH q ≤ 0.10. Change point: max pooled-t over splits with at least 4
  months per side. Its null is a seeded permutation, or simulated AR(1) with a bias-corrected lag-1
  estimate when the residuals are autocorrelated (φ > 0.1). It is reported when p < 0.01.
- **Series anomalies.** Robust z = 0.6745·(x − median)/MAD against the 24 complete months *before* the
  point (at least 12 required), on log1p scale for counts. It scans the last 12 complete months of
  `volume:all`, `active_users:all`, `share:genre:*` and `rating:genre:*`. Per-genre volume is not
  scanned, because one burst would be reported about 15 times. Flag at |z| ≥ 3.5 (Iglewicz–Hoaglin);
  severity medium ≥ 5, high ≥ 7, critical ≥ 10. The min-volume guard is 50 ratings (month or baseline
  median). A series anomaly raises a warning only while it falls in the last 3 complete months.
- **Rater anomalies.** IsolationForest (200 trees, seed 42) on ratings per active day, max ratings in
  one day, busiest-day share, extreme-rating share, rating sd, leave-one-out mean |rating − item mean|,
  and long-tail share (item has ≤ 5 other ratings). The top 2 % are flagged. Severity comes from the raw
  score s (medium ≥ 0.66, high ≥ 0.72, critical ≥ 0.80). Raters idle for more than 365 days are listed
  but suppressed.
- **Live feedback.** Negative (dislike or not_interested) share over the last 7 days vs the prior 28
  days; one-sided two-proportion z test; needs ≥ 30 events per window, otherwise skipped with the
  reason.
- **Forecasts.** Naive, moving average (6 months) and damped Holt (alpha/beta/phi grid, one-step SSE on
  each origin's training part), all on log1p volume with at most 60 months of history. 24
  rolling origins, horizon 6. Selection by backtest MASE (scale = in-sample one-step naive MAE per
  origin). 80 % intervals use finite-sample (conformal) order statistics of the backtest residuals
  per step. Coverage is honest: each origin uses only residuals realised before it.
- **Lapse.** P(no rating in 180 d). Cutoffs every 6 months from 1998, over users with at least one
  rating in the prior 365 d. Nine features (recency, tenure, volumes, active days, 30-day share,
  mean/sd rating). Standardise, then logistic regression. Platt scaling is used only if it lowers
  2-fold validation ECE; on this data it did not, so the method is `none`. The latest 30 % of
  cutoffs are the test set. The evaluated model is the one that scores users at as_of.
- **Risks.** score = 100 × likelihood × impact × confidence × data_quality. Levels: medium ≥ 15,
  high ≥ 30, critical ≥ 50. Impact is measured where the data allows it (genre share / 0.25, lapse
  volume share, exact top-50 trending churn when the flagged raters are removed). Otherwise it is a
  *declared* weight (`risk_declared_impact`), labelled as such in `factors[].detail`.
- **Decisions.** `retrain_model` (rule: new-event share ≥ 5 % or dataset version mismatch).
  `serving_model` (paired bootstrap over 592 users × 2 000 resamples; option scores are P(best);
  confidence is P(best > runner-up); `random` is excluded). `genre_programming` (z = Theil–Sen
  slope/se; margin; non-hold answers need q ≤ 0.10). `rater_action` (margin of anomaly strength ×
  influence; active flagged raters only, at most 5). `reengagement_campaign` (exact Poisson-binomial
  P(≥ 5 high-risk users lapse); abstains if the holdout AUC is below 0.65).
- **Signals.** Strength definitions are in the `signals.py` docstring. One signal per `dedup_key`,
  and at most 5 forecast signals, because genre forecasts move together.

### Results on the real data
Runtime: 0.8–0.9 s per full run (evaluation latency: mean 815 ms, p95 842 ms over 5 runs; the largest
stages are lapse at about 210 ms and explain/serialisation at about 180 ms).

As of 2018-09-24 (default): status **watch**. Quality 1.0. 18 signals; 4 non-flat trends
(active users up, Adventure/Documentary share up, Romance share down), none of which survives the FDR
(q ≥ 0.43). One change point: active users shifted up in 2017-12 (10.4 → 14.1 per month,
p = 0.004). 10 active and 12 suppressed anomalies. 4 risks. **1 warning**:
`risk:audience_lapse:all`, medium (40 of 60 recently active raters have P(lapse) ≥ 0.7; score 22.3).
24 decisions: retrain = no (rule, 0 new events), serving = hybrid (P = 1.00 that it beats itemknn),
18 × genre hold, 3 × rater ignore, re-engagement = yes. 1 action (P2 re-engage 40 users).

As of 2017-07-01 (replay): status **alert**. **1 warning**: the `share:genre:Horror` spike in 2017-05
(high), the genre-specific trace of the Q2-2017 burst. `volume:all` itself is *not* flagged: on log
scale, months with 1 700+ ratings occurred twice in the preceding 24 months, so the burst is not
unusual for this stream. Retrain and serving decisions abstain (the model was trained on data after
as_of).

### Offline evaluation (`experiments/intel-eval-20260923T134257Z/`)
| component | result |
|---|---|
| forecast (21 series, selected on origins 1–12, scored on 13–24) | median MASE 0.93 vs naive 1.15; beats naive on 21/21; mean coverage80 0.95 (conservative) |
| lapse (temporal holdout, 1 017 train / 493 test) | AUC 0.886 vs recency rule 0.779; Brier 0.115 vs recency 0.155 vs base rate 0.200; ECE 0.061 |
| shilling, random / average / bandwagon (12 attackers × 3 trials each, synthetic) | AUC 0.95 / 0.99 / 0.92; at the deployed 2 % budget P 0.23 / 0.44 / 0.13, R 0.25 / 0.47 / 0.14; baseline (LOO deviation) F1 0 / 0 / 0, AUC 0.87 / 0.00 / 0.84 |
| series anomalies (756 injections into real series) | 3σ 0 % (below the 3.5 cut-off, by design), 5σ 73 %, 8σ 76 %; every spike detected, misses are share drops bounded at 0; flag rate on real uninjected points 2.0 % |
| change points (400 synthetic AR(1) series, φ = 0.56 from real volume) | detection 1σ 23 %, 2σ 42 %; false alarms 7 % (nominal 1 %); location error 1.6 months |

Known weaknesses, reported rather than tuned away:
- Change-point false alarms stay above nominal on short autocorrelated series.
  An earlier permutation-only null gave 15 %; the AR(1) null brings it to 7 %.
- The rater detector ranks attackers well but spends most of its 2 % review budget on genuine
  outlying heavy users.
- Series trends almost never survive FDR on this bursty, 10–15-user-per-month stream.

## 8. Implementation notes (backend)

Code: `backend/jev_api/services/intel.py` (inputs, runs, persistence, warning lifecycle, scenarios),
`backend/jev_api/routers/intel.py` (every `/intel/*` endpoint), `backend/jev_api/metrics.py` (the
in-process registry behind `GET /admin/metrics`, in `routers/admin.py`), tables in `models.py` and
migration `0002_intelligence`. Tests: `tests/integration/test_intel_api.py`.

### Runs
- Inputs = `load_default_inputs` (processed files, active model manifest and its experiment run)
  plus the app DB: `ratings` (timestamp = `updated_at`), `recommendation_feedback` and served
  `recommendations`, plus `suppressed_keys` built from dismissed warnings still inside their window
  (`key → severity at dismissal`). `now` is the wall clock.
- One run at a time: a process-wide lock; a second trigger gets 409. `POST /intel/runs` runs in the
  request's worker thread and returns the finished run. Bad dates, future dates and dates outside
  the MovieLens range are rejected with 422 *before* a run row exists.
- Row lifecycle: `running` is committed first. It then becomes `succeeded` (with as_of, versions,
  summary, `stage_ms` = the pipeline's stages plus `load_inputs` and `persist`, and the full result)
  or `failed` (with the error text). Any exception in loading, the pipeline or persistence ends as a
  failed run; it never reaches the client as a 500.
- The full result is a deferred JSON column. List endpoints read it through a 4-entry in-process
  cache. The latest `PipelineResult` object stays in memory, so scenarios on the latest run reuse
  its fitted models (~10 ms). Otherwise, and with `as_of`, the inputs are prepared again (~170 ms).
- "Latest" = the most recently started successful run, replays included. A replay therefore
  becomes what the console shows until the next run.
- Startup: when `JEV_INTEL_RUN_ON_STARTUP` is set and the latest successful run is older than
  `JEV_INTEL_MIN_INTERVAL_HOURS` (or missing), a daemon thread runs the pipeline (trigger
  `startup`). It never blocks startup and logs, rather than raises, any failure. Tests disable it.
- Every run logs one structured line (`intelligence run <status>`) with run_id, trigger, as_of,
  duration, counts and error. For API-triggered runs the line carries the request id.

### Warning lifecycle
- Uniqueness: at most one open (`new|acknowledged|investigating`) warning per key. The service
  enforces it, and so does the DB, through a partial unique index (`uq_intel_warnings_open_key`,
  SQLite and PostgreSQL).
- Per candidate: an open warning with that key → update title/description/severity/confidence/
  evidence/trigger/source, `occurrences += 1`, `last_seen_at`, `last_seen_run_id`. A dismissed key
  whose `suppressed_until` (dismissal + `JEV_INTEL_SUPPRESS_DAYS`) has not passed → skipped, unless
  the severity is above the severity at dismissal. The pipeline applies the same rule through
  `suppressed_keys`, so such anomalies show `suppressed: true`. Otherwise a new `new` warning;
  `reopened_from` points at the previous row for the key when that row was resolved, or was
  dismissed and then expired or escalated.
- Every status change writes an `intel_warning_events` row. The system is the actor for creation
  (the note says detected / reopened / escalated); a PATCH records the user's email. Open warnings
  are never auto-resolved when a run stops producing them. Operators close them.

### Decisions, feedback, metrics
- `intel_decisions` stores every decision of every run (unique per `(run_id, decision_id)`).
  Decision ids are deterministic per as_of, so the same `dec-…` recurs across runs with the same
  as_of. Feedback is keyed by that contract id, and each copy shows the combined counts.
- Feedback checks the verdict against the target type (422), then that the target exists (404).
  For actions and predictions, "exists" means present in the latest run.
- `/admin/metrics` counters are per process and reset on restart. Requests are labelled by route
  template (`GET /intel/series/{series_id:path}`), never by raw path. The DB-derived groups
  (decisions by spec/answer, open warnings, data freshness) are computed on each request.

### Deviations from / additions to section 6
- `GET /intel/decisions/{id}` also accepts the contract id `dec-…`, because the feedback list links
  by that id. `GET /intel/runs/{id}` accepts the numeric id or the uuid.
- Page and list responses add filters beyond the contract: signals `direction`; trends `metric`,
  `entity`; anomalies `kind, severity, entity_type, suppressed`; risks `kind, level`; actions
  `priority`; warnings `key`; decisions `run_id, entity`; feedback `target_type`; predictions
  `series_id`; scenarios `series_id`.
- `GET /intel/runs` takes `limit/offset`. The feedback list adds `limit/offset` to `{items, total,
  summary}`. Warnings add `confidence_kind, entity_type, entity, suppressed_until`.
- `GET /intel/evaluation` returns `run_dir` as the directory name (`intel-eval-…`), not a filesystem
  path.
- Verified: migration upgrade → downgrade → upgrade on SQLite (test) and PostgreSQL 17 (manually,
  including the partial unique index and CHECK constraints); `alembic check` reports no drift.
