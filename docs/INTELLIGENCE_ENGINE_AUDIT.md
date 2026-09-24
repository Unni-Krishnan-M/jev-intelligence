# Intelligence engine audit (Phase 2, WS4a)

_2026-09-24. Scope: the core engine (`ml/jev_ml/core/**` except forecast/series/scenario, which WS4b
owns), the movie domain, the `/intel` API and its warning lifecycle. The companion docs are
[DECISION_ENGINE.md](DECISION_ENGINE.md) and [EARLY_WARNING_SYSTEM.md](EARLY_WARNING_SYSTEM.md).
Every number below comes from a run on this machine.
- Official report: `experiments/platform-eval-20260924T174851Z/` (after) against
  `platform-eval-20260924T171830Z/` (before).
- Variant studies: same-code-state switches, using the scoring functions of
  `scripts/evaluate_domains.py`._

## 1. Stage-by-stage audit

| Stage | What it assumes | Thresholds (defaults) | Failure modes / limits |
|---|---|---|---|
| Ingest / validate | Timestamps are event time. MovieLens rows after as_of are dropped (leakage guard). Generic sources are cut at as_of minus the publication lag. | earliest 1995-01-01; rating grid 0.5 | Revisions after first publication are not modelled (no ALFRED vintages). **Was:** a replay mixed later app events into an earlier analysis. **Fixed** (§2.1). |
| Series | Complete periods only; count series use log1p | min volume 50 (count specs) | Short series skip their trend or forecast stage; this shows as an abstention, not as silence |
| Trends | Mann–Kendall on the last 24 periods; Theil–Sen CI must exclude 0; BH across series | α 0.05, FDR q ≤ 0.10 | Autocorrelation inflates MK significance. A significant window trend persists for months after the series turns. **Now:** reversal detection (§2.4). |
| Change points | Single mean shift; max-t over splits ≥ 4 periods from each end; AR(1) null | α 0.01, 499 sims | **Was:** 7–10 % false alarms at nominal 1 %, because φ was estimated from the 24 in-window points. **Now:** φ from the pre-window history, about 1.3 % (§2.5). Detection of 1σ shifts is low (≈ 5 %). |
| Series anomalies | Robust z (Iglewicz–Hoaglin) against a 24-period trailing baseline; WS4b adds a seasonal baseline | \|z\| ≥ 3.5; severity 5 / 7 / 10 | Level-basis anomalies on trending series; weekend effects on daily data (WS4b's seasonal basis) |
| Forecasts (WS4b) | Rolling-origin backtest; naive unless better | 80 % conformal band | us-unemployment: MASE 1.52 vs naive 1.41, coverage 0.61 (WS4b's area) |
| Risks | score = 100 · likelihood · impact · confidence · data quality | medium 15, high 30, critical 50 | Impact is "measured" as relative change / 0.25 when no weight is declared, so an arbitrary scale. **Was:** risk confidence had no kind. **Fixed** (§2.6). |
| Early-warning decision | Points per component; corroboration +1 per extra stage | MONITOR 1, WARNING 3, URGENT 5 | Margin confidence is distance from a threshold, not a probability. Flip rate between monthly replays is 0.13–0.25 for unemployment and 0.73 for daily cta. |
| Warnings | A warning only downstream of a WARNING or URGENT decision; one per key | max 25 per run | Precision is modest (lift 1.2–1.4 on development windows; the CTA lift CI includes 1 and an untouched CTA window gives 1.10). Movie lapse precision is uninformative (base rate 1.0). Genre decline never fires. |
| Lifecycle | One open warning per (domain, key); dismiss means 30-day suppression | — | **Was:** replays wrote live warnings, and stale warnings never closed. **Fixed** (§2.1, §2.3). |
| Decisions (movie) | Rule policies over bootstrap and model evidence | per-policy versions | Model-governance decisions abstain in replays that predate the model (by design) |
| Evaluation | Leak-free monthly replays, outcomes from today's vintage | h = 6 periods | Tuning and evaluation windows were not separated before this work; now they are (§3) |

## 2. What changed

### 2.1 Replay/live separation (P0)
- **Warnings.** `_persist_success` runs `_upsert_warnings` and `_auto_resolve` only for
  `run.mode == "live"`. A replay's warnings stay in its result, readable via
  `GET /intel/runs/{id}/warnings`.
- **Reads.** `latest_run` now defaults to `mode="live"`. Every default read therefore means the
  latest live run:
  - the run-backed lists and evidence, decision batches, status, decisions, history;
  - domains, metrics, feedback targets, scenarios and the startup refresh.

  `?run_id=` always wins. `?mode=replay` and `?mode=any` select replays.
- **Mode exposure.** `IntelRunOut.mode` is exposed. Pages and decision batches carry the `mode` of
  their run, and `GET /intel/runs` takes `?mode=`.
- **Suppression and clock.** A replay ignores operator suppression (live state). In the pure
  pipeline, app events are cut at as_of and live-window stages use the replay clock:
  `domains/movie/ingest.py` (`event_clock_ts`), `raters.live_feedback`, the live signal. WS1's event
  log performs the same cut in the API.
- **knowledge_time.** WS1's `knowledge_time` is threaded through `IntelService.run` and
  `POST /intel/runs`. It is rejected without an as_of and rejected in the future.
- Tests are listed in [EARLY_WARNING_SYSTEM.md](EARLY_WARNING_SYSTEM.md) §3.

### 2.2 Evidence lineage
- `jev_ml.core.lineage` resolves decision → evidence → run objects → series → source. The API adds
  the run node (mode, versions, `config_hash`, `input_fingerprint`, WS1's `event_watermark`).
- Endpoints: `GET /intel/decisions/{id}/lineage` and `GET /intel/warnings/{id}/lineage`.
- `run.config_hash` and `run.input_fingerprint` are new fields of the result (no migration).
- Real runs: 55/55 movie and 9/9 unemployment decisions resolve, and 1/1 and 6/6 warnings (smoke
  test). Every decision and warning of real movie (live and 2017-07-01) and unemployment (live and
  2008-06-01) runs resolves in `tests/core/test_decision_quality.py`.

### 2.3 Stale-warning auto-resolution
- A warning is resolved after K = 3 consecutive live runs without its key
  (`JEV_INTEL_AUTO_RESOLVE_RUNS`).
- Each resolution writes a system warning event and a `warning.auto_resolve` audit row.
- Severities up to `JEV_INTEL_AUTO_RESOLVE_MAX_SEVERITY` (default `medium`, per the P2.4 criterion)
  resolve on their own; high and critical wait for an operator.

### 2.4 Recovery-aware trends (policy `ewl-1.1.0`, core `core-1.1.0`)
- **Rule.** `trend.recent_move` flags a trend as reversing when the latest value has retreated from
  the window's extreme, in the trend's direction, by more than 2σ√3. σ is the MAD of period
  changes, the same noise scale as the outcome definition.
- **Effect.** A reversing trend or change point earns 0 early-warning points and raises no
  `adverse_trend` risk.
- **Switches.** `trend_reversal_periods = 0` turns it off; `trend_reversal_basis` selects the basis.

### 2.5 Change-point null
- The AR(1) null's φ now comes from up to 60 periods before the trend window (at least 24 needed).
  Series with less history keep the in-window estimate; `phi_source` says which was used.
- Switch: `change_point_history_max = 0` restores the old null.

### 2.6 Confidence semantics
- Risks carry `confidence_kind` (`rule` / `margin` / `evidence`), and a risk-sourced warning takes it
  over (it was always `margin`).
- `confusion()` reports `lift` and `flag_rate` next to the base rate wherever precision is reported.
- Test: every emitted confidence has a kind in `CONFIDENCE_KINDS`.

### 2.7 Other changes
- **Circular imports.** `user_intel.py`, `user_scenario.py` and `risks.py` no longer import through
  `jev_ml.intel`. The service imports `jev_ml.core` and `jev_ml.domains.movie` directly; the
  inputs-based scenario and `run_movie_pipeline` moved to `domains/movie/scenario.py`, and the shim
  re-exports them.
- **Daily-series cosmetics** (reported by WS4b):
  - `period_age_days` takes the series frequency: daily freshness was off by up to 6 days;
  - risk and change-point titles use the full date for weekly and daily series;
  - the trend unit says "/day".

**Versions.** `CORE_VERSION` `core-1.1.0`; `EWL_POLICY_VERSION` `ewl-1.1.0`. Movie goldens were
regenerated for exactly two differences: the new change-point null drops some weak change points and
their signals, and risk-sourced warnings carry the risk's confidence kind. With both switched off the
new code reproduces the old goldens byte for byte (verified). The golden writer now stores the v1.1
view the test compares.

## 3. Evaluation protocol
- Candidates are chosen on **tuning windows**: us-unemployment 1990-01..1994-12 and 1999-01..2003-12;
  cta 2010..2014.
- They are scored on the **evaluation windows**: us-unemployment 2006-01..2010-12 and
  2019-01..2021-12 (the official windows); cta 2015..2024. Movie: the last 24 months (no tuning
  window, so no policy was tuned on it).
- **Only one choice was made on tuning data:** the reversal basis (`drawdown` over `recent`). The
  thresholds (2σ√3, 3 periods) were fixed a priori to match the outcome definition's noise scale;
  they were not searched.

## 4. Before / after

### 4.1 Warnings, us-unemployment (official report; units = monitored series × monthly replay)

| Window | Version | Precision | Base rate | Lift | FPR | Recall | Flag rate | EWL flip | Boundary flip |
|---|---|---|---|---|---|---|---|---|---|
| pooled eval | before (1.0) | 0.360 | 0.295 | 1.22 | 0.387 | 0.519 | 0.426 | 0.165 | 0.073 |
| pooled eval | **after (1.1)** | **0.415** | 0.295 | **1.41** | **0.300** | 0.508 | 0.362 | **0.153** | 0.076 |
| 2006–10 | before | 0.443 | 0.349 | 1.27 | 0.437 | 0.649 | 0.511 | 0.129 | 0.053 |
| 2006–10 | after | 0.451 | 0.349 | 1.29 | 0.414 | 0.635 | 0.491 | 0.126 | 0.061 |
| 2019–21 | before | 0.109 | 0.206 | 0.53 | 0.319 | 0.151 | 0.284 | 0.224 | 0.106 |
| 2019–21 | after | 0.214 | 0.206 | 1.04 | 0.144 | 0.151 | 0.145 | 0.196 | 0.101 |

Ablation on the evaluation windows (pooled, same code state):

| Variant | Precision | Lift | FPR | Recall |
|---|---|---|---|---|
| cp-history only | 0.368 | 1.25 | 0.365 | 0.508 |
| reversal, basis `recent` | 0.390 | 1.32 | 0.333 | 0.508 |
| reversal, basis `drawdown` (kept) | 0.415 | 1.41 | 0.300 | 0.508 |

Tuning windows (pooled, base rate 0.230):

| Variant | Precision | Lift | FPR | Recall |
|---|---|---|---|---|
| 1.0 | 0.261 | 1.14 | 0.425 | 0.503 |
| `recent` | 0.260 | 1.13 | 0.416 | 0.488 |
| `drawdown` | 0.271 | 1.18 | 0.392 | 0.488 |

Here `drawdown` was chosen, and the evaluation windows then confirmed it.

### 4.2 cta-ridership (daily; units = the 8 entities × monthly replay; evaluation 2015–24; base rate 0.0625)

| Version | Precision | Lift | FPR | Recall | Warned | EWL flip | Boundary flip | Unit warned-flip |
|---|---|---|---|---|---|---|---|---|
| before (1.0) | 0.080 | 1.27 | 0.309 | 0.400 | 302 | 0.680 | 0.281 | 0.271 |
| **after (1.1)** | **0.090** | **1.44** | **0.224** | 0.333 | 222 | 0.740 | 0.359 | 0.282 |

The official report gives the same after-values: precision 0.0901, FPR 0.2244, recall 0.3333,
lift 1.44. Its flip rates are 0.730 (EWL) and 0.292 (unit), counted over the continuous 120-replay
sequence.

On the tuning years 2010–14:
- lift rose from 1.67 to 2.25 and FPR fell from 0.199 to 0.147, with recall unchanged at 0.333;
- the unit warned-flip rate fell from 0.255 to 0.196.

**Interpretation (evaluation review).** Policy 1.1 together with WS4b's special-day class reduces false positives
3.4× against the plain generic config (FPR 0.77 → 0.22), but the warning skill is not statistically established.
Lift 1.44 has a month-cluster bootstrap 95 % CI of 0.52–2.36, the year-stratified lift is 1.18, and on the untouched
2003–09 window the same config gives lift 1.10 (CI 0.00–2.33; 3 of 16 events). The 2015–24 window has been scored at
least four times (WS4b v0, before and after 1.1 here, and the ablations and K study), so it is a development window,
not a hold-out. Source: [EVALUATION_AND_VALIDITY_REVIEW.md](EVALUATION_AND_VALIDITY_REVIEW.md) §3.

### 4.3 Movie (last 24 months, same code state)
Unchanged by the 1.1 policy:
- precision 1.0, base rate 0.108, lift 9.26, FPR 0, recall 0.46;
- EWL flip 0.233, boundary flip 0.026.

The headline is dominated by `audience_lapse`, whose base rate is 1.0 (lift 1.0: uninformative).
`genre_demand_decline` never fires (recall 0). Both are reported, not fixed: fixing them needs a
movie-specific policy change that would need its own tuning window.

### 4.4 Change points (synthetic AR(1), window 24, m = 4, α 0.01; 60 periods of history)

| True φ | Oracle null FA | History φ FA (kept) | Window φ FA (old) | Detection 2σ: oracle / history / old |
|---|---|---|---|---|
| 0.30 | 1.1 % | 1.2 % | 3.5 % | 0.46 / 0.44 / 0.52 |
| 0.56 | 0.9 % | 1.5 % | 9.2 % | 0.23 / 0.24 / 0.44 |
| 0.80 | 1.3 % | 2.9 % | 20.2 % | 0.12 / 0.14 / 0.43 |

"Oracle" is the critical value under the true φ (200k simulations): the best any valid test can do.

**Movie protocol** (`evaluate_change_points`, φ = 0.56 from the real volume, 2000 trials):
- false alarms 9.9 % → **1.3 %**;
- detection 30.8 % → 13.5 % (1σ 5.6 %, 2σ 21.4 %).

The old detection rate was mostly false alarms. The new rates match the oracle's power, so detection
did not collapse: it is now honest. It meets the P2.4 bar (≤ 2 % at nominal 1 %) at the real φ.

### 4.5 Auto-resolve K (lifecycle simulation, us-unemployment, new policy; open state scored against the outcome)

| K | Severities | Eval precision | Lift | FPR | Recall | Open/close transitions per unit-month |
|---|---|---|---|---|---|---|
| never (before) | — | 0.292 | 0.99 | 0.557 | 0.548 | 0.021 |
| 1 | all | 0.415 | 1.41 | 0.300 | 0.508 | 0.044 |
| 3 | all | 0.386 | 1.31 | 0.346 | 0.519 | 0.038 |
| 3 | ≤ medium (default) | 0.307 | 1.04 | 0.509 | 0.537 | 0.025 |
| 5 | all | 0.360 | 1.22 | 0.387 | 0.519 | 0.036 |

On the tuning windows the ordering is the same: K 1 → 3 → never gives lift 1.18 → 1.14 → 0.85.

## 5. Negative results and open issues
1. **Change-point alternatives rejected** (φ 0.56, same study):
   - a studentised max-t (T·√((1−φ)/(1+φ))): 7 % false alarms;
   - a sieve bootstrap of the AR(1) residuals: 8 %;
   - φ + 1 SE: 5 %, and 2σ detection falls to 0.32 with the SE inflation;
   - a detrended-history φ: 2–5 %.

   None matched the history estimate.
2. **The cp-history change alone does not improve warnings.** Tuning precision went 0.261 → 0.258;
   evaluation lift went 1.22 → 1.25. It is kept for calibration (§4.4), not for warning precision.
3. **Reversal basis `recent`** (change over the last 3 periods) was weaker than `drawdown` on the
   tuning windows, and it raised the flip rate on the evaluation windows (0.165 → 0.174). Rejected.
4. **cta pays in recall and stability.**
   - The reversal rule costs recall on 2015–24: 0.40 → 0.33, i.e. 4 fewer true positives out of 60.
   - It raises the EWL flip rate (0.68 → 0.74) and the boundary flip rate (0.28 → 0.36): on noisy
     daily data the reversal state itself toggles.
   - It is kept because it helps precision, lift and FPR in both the tuning and evaluation windows,
     and it was not tuned per domain. A cta-specific setting would need a separate window.
5. **Hysteresis in the pure pipeline was not implemented.** A lower exit threshold for warnings that
   were already open needs cross-run state that `run_domain` does not have, and the official harness
   could not measure it. The lifecycle's K is the hysteresis, and it is not free: K = 3 against K = 1
   trades 0.03 of precision for 15 % fewer open/close transitions.
6. **The acceptance criterion's severity gate for auto-resolve costs open-state precision.** On
   us-unemployment most warnings are high, so with `max_severity = medium` stale warnings stay open.
   The open-state lift is 1.04, against 1.31 when every severity auto-resolves. The default follows
   the criterion (an operator closes high and critical). Deployments without operators should set
   `JEV_INTEL_AUTO_RESOLVE_MAX_SEVERITY=critical`.
7. **2019–21 is still barely better than chance** (lift 1.04). COVID's shock and immediate recovery
   break both the trend and the anomaly evidence.
8. **Movie warning quality is not measurable as built** (§4.3).
9. **Evaluation versus serving path.** The official harness calls `run_domain` directly: no
   suppression, no app events. The API differs only in live state, and replays ignore suppression
   now, so API replays of generic domains without pushed observations should equal the harness runs.
   That equality is not tested directly (determinism within each path is), and a shared RunContext
   builder is not done.
