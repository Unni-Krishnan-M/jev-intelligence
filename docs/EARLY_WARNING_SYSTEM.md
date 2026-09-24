# JEV early-warning system

_Phase 2, WS4a. From evidence to an operator's warning, the warning lifecycle (including stale-warning
auto-resolution and replay isolation), and how warning quality is evaluated. Numbers:
[INTELLIGENCE_ENGINE_AUDIT.md](INTELLIGENCE_ENGINE_AUDIT.md). Decisions:
[DECISION_ENGINE.md](DECISION_ENGINE.md)._

## 1. Pipeline to a warning

```
observations ─→ series ─→ DETECT: trends (+ change point, recent move) · series anomalies · forecasts
                          (+ domain extras: raters, live feedback, lapse, …)
             ─→ ASSESS: risks (score = 100 · likelihood · impact · confidence · data quality)
             ─→ DECIDE: one early_warning_level decision per situation (ewl-1.1.0)
             ─→ RECOMMEND: a warning for each situation at WARNING or URGENT_ACTION
             ─→ the API's warning lifecycle (live runs only)
```

- **Situations** group evidence by subject: `series:<id>`, or `entity:<type>:<entity>`.
- **Warning keys.**
  - Situation mode (generic domains): `warning:<situation>`.
  - Component mode (movie, v1.1 keys): `risk:<kind>:<entity>`, or the anomaly's dedup key.
- **Every warning carries:**
  - its `decision_id` and `early_warning_level`;
  - a `trigger` (rule, condition, observed, threshold);
  - its evidence;
  - `confidence` with `confidence_kind`. A risk-sourced warning takes the risk's kind; anomaly and
    situation warnings are `margin`.
- **Where suppression applies.**
  - Keys an operator dismissed stay quiet for `JEV_INTEL_SUPPRESS_DAYS` (30) unless the severity
    escalates.
  - Suppression is live state, so a replay ignores it.
- **Recovery-aware trends** (core-1.1.0).
  - A trend whose series has already fallen back from its window extreme by more than 2σ√3 is
    *reversing*.
  - It earns no early-warning points and raises no `adverse_trend` risk. A reversing series therefore
    stops producing a trend warning as soon as it turns, not when the 24-period window finally goes
    flat.
- **Change points** (core-1.1.0). The null's AR(1) coefficient is estimated on the series' own
  history before the window, which gives about 1–2 % false alarms at a nominal 1 % (it was 7–10 %).

## 2. Warning lifecycle

| Status | Meaning | Next |
|---|---|---|
| `new` | raised by a live run | acknowledged, investigating, resolved, dismissed |
| `acknowledged` | an operator saw it | investigating, resolved, dismissed |
| `investigating` | being worked | resolved, dismissed |
| `resolved` | closed (by an operator or by auto-resolution) | terminal; the key can reopen |
| `dismissed` | false positive or not useful | terminal; the key is suppressed for 30 days unless its severity escalates |

**Rules applied on every live run, per domain:**
1. **Update.** An open warning with the same key is updated: last seen, occurrences, severity,
   evidence, decision id.
2. **Suppress.** A key dismissed within the suppression window stays quiet unless its severity
   escalated.
3. **Reopen.** A resolved key, or a key whose suppression expired or whose severity escalated, that
   fires again opens a **new** warning with `reopened_from` pointing at the previous one.
4. **Auto-resolve (new).** An open warning whose key is absent from the last `K` consecutive **live**
   runs is moved to `resolved` (details below).

At most one open warning per (domain, key): a partial unique index on PostgreSQL and SQLite.

**Auto-resolution details.**
- `K` = `JEV_INTEL_AUTO_RESOLVE_RUNS`, default 3; 0 turns it off. It sits in `config.py`, intel
  section.
- "Consecutive live runs" means live, succeeded runs of the domain after the run that last saw the
  key, in (started_at, id) order, the current run included.
- Replays and failed runs do not count.
- Severities up to `JEV_INTEL_AUTO_RESOLVE_MAX_SEVERITY` (default `medium`) resolve on their own. A
  high or critical warning that goes stale stays open for an operator, as the P2.4 acceptance
  criteria require.
- Each auto-resolution writes:
  - a `intel_warning_events` row: actor `system`, `from_status` → `resolved`, the run id, and a note
    ("auto-resolved: key absent from N consecutive live runs (threshold K); last seen in run …");
  - an audit row `warning.auto_resolve`: actor `system`, detail = domain, key, from, severity,
    `absent_live_runs`, threshold, run id, last seen run id;
  - `closed_at`;
  - the metric `intel_warnings.auto_resolved`.
- If the key fires again later, it reopens with `reopened_from`, as for an operator resolution.

**Why K = 3 and not 1.** The open/close hysteresis comes from `K`: a warning that disappears for one
or two runs and comes back is not closed and reopened (flip-flop). The lifecycle simulation on
us-unemployment (§4) shows the trade-off:
- the higher K, the fewer open/close transitions;
- but a stale warning stays open longer, which costs open-state precision.

K = 3 is the acceptance criterion's value. It is documented, not tuned.

## 3. Replay isolation

A run with an explicit `as_of` is a replay (`intel_runs.mode = "replay"`).

**What a replay does and does not touch:**

| Concern | Rule | Where |
|---|---|---|
| Live warnings | a replay never creates, updates, reopens or auto-resolves one; no warning events | `IntelService._persist_success` |
| Where replay warnings live | in the run's result only: `GET /intel/runs/{run_id}/warnings` (an `IntelPage` with `mode`) | `routers/intel.run_warnings` |
| Operator suppression | ignored (live state) | `IntelService._run_locked` |
| App events | cut at as_of; ingestion cut at `knowledge_time` (default as_of) | WS1 `events.app_frames`, `domains/movie/ingest.prepare` |
| Live-window stages | live feedback and app freshness run on the replay clock (as_of), not now | `ingest.prepare` (`event_clock_ts`), `raters.live_feedback` |
| Startup refresh | considers live runs only: a fresh replay does not make the data "fresh" | `IntelService._startup_refresh` → `latest_run(mode="live")` |

**Which run a read means.** Every default read means the latest **live** run:
- the run-backed lists: signals, trends, anomalies, risks, actions, predictions, series, evidence,
  decision batches;
- `/intel/status`, `/intel/decisions`, `/intel/history/*`, `/intel/domains`, `/admin/metrics`;
- feedback target checks, scenarios and the startup refresh.

Parameters that select a replay:
- `?run_id=` always selects that run, replay or live;
- `?mode=replay` selects the latest replay;
- `?mode=any` selects the latest run of either mode (the pre-Phase-2 behaviour);
- `GET /intel/runs` lists every run by default (`?mode=live|replay` filters). Every run carries
  `mode`, and every page carries the `mode` of the run it was read from.

Tests (`tests/integration/test_intel_replay_lineage.py`):
- `test_replay_never_touches_live_warnings`: warning rows and events are byte-identical before and
  after two replays, and the replay's own warnings are listed under the run;
- `test_replay_is_not_the_default_latest_run`: every list, status, decisions, history, metrics and
  domains read;
- `test_startup_refresh_ignores_replays`: stale live runs plus a brand-new replay still trigger the
  refresh; a fresh live run does not;
- `test_replay_uses_the_replay_clock_for_app_events`: a 2026 member rating is in the live run's app
  source and absent from a 2018 replay;
- `test_auto_resolve_after_k_live_runs` and `test_auto_resolve_leaves_severe_warnings_to_an_operator`;
- `test_knowledge_time_is_threaded_to_the_event_cut`.

**Replay determinism holds for a fixed code version, data files and active model.** A replay excludes
events that arrived after its `as_of`, but a movie replay reads the model manifest that is active at
replay time, so a promotion or rollback between two replays of the same `as_of` changes the
model-based decisions and risks. Each run records the model it used (`run.model_version`,
`intel_runs.model_version`); `load_default_inputs(..., model_version=...)` pins it from Python
(`tests/intel/test_replay_model_version.py`), the API does not. See
[STREAMING_ARCHITECTURE.md](STREAMING_ARCHITECTURE.md) §4.1.

## 4. Evaluation

**The official harness.** `uv run python scripts/evaluate_domains.py` (owned by WS4b) replays each
domain monthly and scores warnings against what happened next. Each replay is leak-free: it sees
only data published by its as_of. A unit is warned when a warning was raised for its situation. It is
confirmed when the adverse condition materialised within h periods:
- movie and us-unemployment: an adverse move beyond 2σ√k, with h = 6;
- cta-ridership: a seasonal outcome.

**Metrics** (`core.evaluation.confusion`):
- precision, recall, false-positive rate;
- **base rate**, **lift** = precision / base rate, and flag rate;
- the flip rate of `early_warning_level` between consecutive replays, and the WARNING-boundary flip
  rate;
- cta: also the per-unit warned-flip rate;
- every warning evaluation stores its per-unit rows (`warnings.rows`) and a **month-cluster bootstrap
  CI of the lift** plus the year-stratified lift (`warnings.lift_uncertainty`,
  `core.evaluation.lift_uncertainty`). Units of one replay month share its data, so an
  independent-units test (hypergeometric) overstates significance. Example (CTA 2015–24, reviewer
  reproduction): lift 1.44 has CI 0.52–2.36 and a year-stratified value of 1.18; an untouched
  2003–09 window gives 1.10 (CI 0.00–2.33).

**Protocol for policy changes (WS4a).**
- A candidate is chosen on **tuning windows**: us-unemployment 1990–94 and 1999–2003; cta 2010–14.
- It is then scored once on the **evaluation windows**: us-unemployment 2006–10 and 2019–21 (the
  script's windows); cta 2015–24.
- Both configurations are run at the same code state (the old policy is a config switch:
  `trend_reversal_periods = 0`, `change_point_history_max = 0`), so other workstreams' concurrent
  changes cannot confound the comparison.
- Harnesses: `scripts/evaluate_domains.py` for the official report; the WS4a variant runs reuse its
  scoring functions.

**The lifecycle simulation.** The official harness scores each replay's warnings, not the
lifecycle's open state. The auto-resolve K study replays monthly live-run sequences through the
lifecycle rules and scores "open at month t" against the outcome. Results are in the audit, §4.3.

**Limits.**
- Outcomes are scored against today's data vintage (no ALFRED vintages).
- Movie lapse precision is uninformative: its base rate is 1.0.
- Genre-decline recall is 0: it never fires.
- These are reported, not hidden.
