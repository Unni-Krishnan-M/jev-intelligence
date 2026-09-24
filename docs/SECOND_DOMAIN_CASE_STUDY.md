# Second reference domain: Chicago Transit Authority daily ridership

This case study shows that the domain-independent JEV engine can run on a serious second dataset through the generic
adapter, with a YAML config and no domain code. It separates what the generic engine does from the few
seasonality-aware options this domain needs, and measures both on identical terms. All numbers come from
`experiments/platform-eval-20260924T174628Z` (final run; the tuning numbers are from separate, unsaved tuning replays,
see §5.4), except the lift CIs and the untouched-window results in §5.2, which come from the independent reproduction
in [EVALUATION_AND_VALIDITY_REVIEW.md](EVALUATION_AND_VALIDITY_REVIEW.md) §3. Every run used a fixed `now` of
2026-09-24. **Headline:** forecasts beat naive and seasonal naive on 8 of 8 series; the warnings cut false positives
3.4× against the plain generic config but show no statistically established skill (lift CI includes 1).

## 1. Dataset, licence, provenance

| | |
|---|---|
| Datasets | *CTA – Ridership – Daily Boarding Totals* (`6iiy-9s97`): bus, rail and total boardings per day, 2001-01-01..2026-06-30, 9,312 rows, with CTA's `day_type` (W weekday, A Saturday, U Sunday/holiday). *CTA – Ridership – 'L' Station Entries – Daily Totals* (`5neh-572f`, about 1.33 M rows): only the five busiest stations of 2019 are fetched: Lake/State, Clark/Lake, Chicago/State, Washington/Dearborn, O'Hare Airport. |
| Publisher | City of Chicago Data Portal (attribution: Chicago Transit Authority) |
| Licence | Dataset licence field: "See Terms of Use". The City of Chicago Data Portal Terms of Use (https://www.chicago.gov/city/en/narr/foia/data_disclaimer.html) provide the data "as is" for any lawful use, with no warranty. The City may change or withdraw a dataset. No API key is needed. |
| Access | Socrata SoQL CSV. Pages use `$limit=50000` / `$offset` with a total `$order` (`service_date`, or `date, station_id`). The row count is checked against `count(*)`. The station ranking is computed on the server (`sum(rides)` for 2019, `$group`, `$order ... DESC`, `$limit 5`) and then those stations are fetched with `$where station_id in (...)`. |
| Output | `data/raw/domains/cta-ridership/cta_ridership_long.csv`: 74,506 rows (8 entities × ~9,312 days). SHA-256 `9ca153c3784a3ef9871c7aa5cd55b1233152de791f3b133925a9e8733919c01d`. Every raw page is stored with its own SHA-256 in `provenance.json`, together with the URLs, the query parameters, the station ranking and the fetch time. |
| Git | `data/raw/*` is gitignored (`git check-ignore` → `.gitignore:13:data/raw/*`). The data is never committed. |
| Data issues found | July 2012 is missing for every series (31 days). They are NaN gaps and are never scored. The station data has 10 duplicate rows; the core validator keeps one of each and reports a warning check. Some station days are near zero (e.g. Chicago/State 51 entries on 2026-06-07, probably station work). |

## 2. Domain assumptions

- **Adverse direction:** down. Falling ridership is the adverse event for an operator: lost fare revenue and a signal of
  changed travel behaviour.
- **Weekly cycle:** a Saturday carries about 55 % of a weekday's boardings and a Sunday about 40 %. The cycle is
  multiplicative, so every count series is modelled on the log scale (`is_count: true`).
- **Calendar:** CTA runs a Sunday schedule on six holidays. The rule-based calendar `weekday_us_holidays` covers New
  Year's Day, Memorial Day, Independence Day, Labor Day, Thanksgiving and Christmas Day, each observed on the nearest
  weekday. It agreed with CTA's own `day_type` on **99.91 % of 9,312 days** when this was written; the 8
  disagreements are early-2000s observance differences. The adapter checks this on every run (quality check
  `chicago-data-portal_calendar`, in each run's `data.quality`); the figure is that run-time check, not a number
  stored in the evaluation report.
  Forecasts need the rule because future days have no `day_type`.
- **Publication lag:** replays take the *operator's* view (`availability_lag_days: 1`: a day's taps are known the
  next morning). The public portal lags about 45 days. The COVID replay is also reported under that lag (§4.2).
- **Grid:** `grid_end: last_observation`. A published day is final. The default run (as of 2026-09-24, data to
  2026-06-30) therefore reports the source as stale (86 days behind, SLA 75 days) instead of showing empty days.

## 3. Generic engine vs domain-aware options

Everything below runs in the same core pipeline (`run_domain`). "Domain-aware" means config options only. Each option
is domain-independent code, opted into per series and justified by a property of this data.

| Stage | Generic (plain config) | Domain-aware (`configs/domains/cta-ridership.yaml`) | Why it is justified |
|---|---|---|---|
| Series | daily `level` per entity (log scale) | same, plus `vs_last_year:<entity>` = 28-day total / the same 28 weekday-aligned days 364 days earlier (`rolling: 28, ratio_lag: 364`) | the ratio cancels the weekly and annual cycles, so a falling ratio is a genuine decline |
| Trend | Mann–Kendall on raw daily values (mostly measures the weekly cycle) | on the ratio series only (91-day window) | as above |
| Anomalies | robust z against the trailing 24 days | `anomaly_basis: seasonal`: against the previous 24 days of the same class (Mondays with Mondays; Sundays with Sundays; MLK Day, Presidents' Day, the Friday after Thanksgiving, 24–31 Dec and the six holidays with each other) | a Sunday is not a drop |
| Forecast | naive, moving average, damped Holt (v1.2) | naive, seasonal naive (lag 7), seasonal naive yearly (364 d), calendar naive (last same-class day), Holt–Winters (log, damped, season = weekday), Holt–Winters with the calendar class as season | the weekly cycle and holidays are known in advance |
| Benchmark for "beats naive" | random walk | seasonal naive | the honest benchmark for a seasonal series |
| Thresholds, risk, early-warning policy, warnings | core defaults, windows in days (identical in both) | identical | — |

The generic engine itself needed three additive capabilities, all opt-in: daily and weekly grids, season classes, and
seasonal models with a seasonal anomaly baseline. Monthly output is byte-identical to v1.2. The movie golden stages
(series, anomalies, forecasts, predictions) and the unemployment synthetic golden are unchanged.

## 4. Real runs (domain-aware config)

A run takes 0.4–1.0 s on this laptop (8 entities, 16 series, 9,300 days; the forecast stage takes 0.15–0.33 s).

### 4.1 Default run (latest data: 2026-06-30)

- **Status:** no warnings. 3 rising and 3 falling ratio trends; 2 active low-severity anomalies: a bus drop on Wed
  2026-06-17 (z −3.9, 521 k vs a same-weekday median of 663 k) and a Lake/State spike on 2026-06-29.
- **Trends:** the year-over-year ratio is 1.03 for total and 1.02 for bus. It falls at Lake/State (0.98 against 1.04 in
  the prior quarter), Chicago/State (1.05 against 1.10) and Washington/Dearborn. These trends reach MONITOR, not
  WARNING.
- **Forecasts (14 days):** total uses `calendar_naive` (backtest MASE 0.63 vs seasonal naive 0.79 vs random walk
  2.29). 2026-07-14 is forecast at 1.09 M (80 % band 0.99–1.23 M). Rail uses `seasonal_naive_yearly`.
- **Risk:** the source is stale (86 days, SLA 75), which raises a staleness risk.
- **Scenarios** (`boardings:total`, 14 days): the selected model has no trend, so a Theil–Sen slope of the last 91
  days is used: −0.0002/day on log scale, 95 % CI −0.0011..+0.0008, i.e. no trend. The scenarios are therefore close
  together:
  - continue: 13.75 M boardings;
  - stall: +0.16 %;
  - reverse: +0.32 %, because the fitted slope is slightly negative;
  - −15 % shock from day 8: −7.6 %.
  Stall and reverse differ from continue by less than the interval width, and the output says so through the slope's
  CI.

### 4.2 As of 2020-03-20 (the COVID collapse)

The run as of 2020-03-20 sees data through 2020-03-19. Summary: **9 warnings (7 critical, 2 high), every entity at
URGENT_ACTION.** Boardings on Thu 2020-03-19 were 469 k against a same-weekday median of 1.50 M (z −17.4). The drops
started on Fri 03-13 (1.17 M against 1.46 M, z −3.7). The adverse-anomaly risks score 94.6 (critical). O'Hare also
has an adverse ratio trend (high). The Holt–Winters-calendar forecast for the following two weeks is about 506 k/day.

**When did it fire?** These are daily as-of replays from 2020-02-20 to 2020-03-25, operator lag of 1 day:

| as of (sees data through) | entities warned (of 8) |
|---|---|
| 2020-02-20 .. 03-02 | 1 (Washington/Dearborn: a pre-existing warning from mid-February, see §5.2) |
| 2020-03-03 .. 03-12 | 0 |
| 2020-03-13 (03-12) | 2 (Lake/State, Washington/Dearborn: first Loop station drops on Thu 03-12) |
| **2020-03-14 (Fri 03-13)** | **6** (total, rail, Clark/Lake, Lake/State, O'Hare, plus Washington/Dearborn) |
| 2020-03-15 (03-14) | 7 (bus added) |
| 2020-03-16 (03-15) onwards | 8 (Chicago/State added) |

So JEV flagged the onset of the COVID drop from the first two days of falling ridership (data through Fri 13 March
2020), before the steepest decline, in a replay of a known event: total boardings then fell to 778 k on Mon 03-16 and
469 k on Thu 03-19, against about 1.5 M before, and the stay-at-home order followed on 21 March. This is detection
of drops that had already been observed (03-12 and 03-13, z −3.7), not a forecast, and the replay date range was
chosen with hindsight. Under the **public portal lag (45 days)**, the same
evidence is first visible as of 2020-04-26/27 (data through 03-12/13, 2 and then 6 entities warned). All 8 entities
warn from 2020-04-29. A consumer of the public data therefore gets the warning about six weeks after the operator.
The lead time comes from the operator's data, not from the method.

The plain generic config warned **every day** in this period on 7–8 of 8 entities, including all of February. Its
"warning" carries no information about COVID (§5.2).

### 4.3 As of 2008-10-01 (calm period, pre-financial-crisis)

- **Status:** no warnings. Two small rail spikes (Sep 2008 was a record-ridership month, during the fuel-price peak).
  The ratio trends are mixed (total 1.08 against 1.05, trending down within the quarter).
- **Sept–Oct 2008 daily replays** (61 as-of dates):
  - domain-aware: at least one warning on 19 of 61 days (58 warnings in total);
  - plain generic: warnings on 61 of 61 days (427 warnings).
- The domain-aware warnings of 14–20 September come from drops on Sat–Sun 13–14 September 2008. Examples: O'Hare
  Airport 1,276 entries against a same-class median of 7,406 (z −11.6); bus −24 % on the Sunday. That weekend had
  record rainfall and flooding in Chicago (external fact, not verified from this data). These are genuine anomalies,
  but not sustained declines, so §5 scores them as false positives.

### 4.4 As of 2021-07-01 (recovery) and what-if scenarios

- **Status:** no warnings. The year-over-year ratios are far above 1 (rail 2.2, Loop stations 3–3.6 against the 2020
  trough).
- **Forecast:** `holt_winters_calendar` for total.
- **Scenarios** (Theil–Sen slope +0.0026/day on log scale, 95 % CI +0.0018..+0.0034):

| scenario | 14-day total | vs continue | day 14 (80 % band) |
|---|---|---|---|
| Recovery continues | 7.51 M | 0 | 612 k (574–694 k) |
| Recovery stalls (trend × 0) | 7.36 M | −2.0 % | 590 k |
| Recovery reverses (trend × −1) | 7.22 M | −3.9 % | 569 k |
| Shock −15 % from day 8 | 6.93 M | −7.8 % | 520 k |

## 5. Evaluation

Protocol (`scripts/evaluate_domains.py`, `ml/jev_ml/domains/generic/evaluation.py`). All replays are leak-free: each
as-of run sees only the data published by then (test `test_cta_as_of_is_leak_free`).

- **Windows:** tuning 2010–2014, evaluation 2015–2024. The seasonal-model candidates, grids, horizons and windows were
  fixed before any replay. §5.4 lists the one change made after the tuning replays.
- **Forecasts:** the served procedure is replayed every 7th day, with horizon 14 days:
  - the candidates' backtests on the 56 daily origins before the issue time, restricted to targets already observed,
    select the model;
  - its residuals give the 80 % conformal interval;
  - the forecast is scored on the next 14 days.
  MASE is scaled by the in-sample lag-7 naive MAE for both configs. The relative MAE is the model MAE divided by the
  benchmark MAE on the same points.
- **Warnings:** 120 monthly as-of replays (the 1st of each month, 2015-01..2024-12) × 8 entities = 960 units.
  - A unit is warned when any warning names one of the entity's series.
  - **Confirmed** is defined as follows:
    - Y(d) = log(7-day total ending d) − log(the same weekday-aligned 7-day total 364 days earlier);
    - σ = 1.4826 × the MAD of the 52 week-over-week changes of Y before the replay;
    - the unit is confirmed if, for some k ∈ {1..4} weeks, both Y(d0 + 7k) and Y(d0 + 7k + 7) are more than 2σ√k below
      Y(d0).
    In words: a sustained (two consecutive weeks) adverse move beyond week-to-week noise within 4 weeks. The rule is
    the same for both configs and independent of them.
  - Precision, FPR, recall, base rate and lift (precision / base rate) follow.
- **Flip rate:** the share of consecutive monthly replays in which the early-warning level changes, per situation
  (`decision_consistency`) and per unit's warned status (comparable across configs).

### 5.1 Forecasts vs naive and seasonal naive (evaluation window 2015–2024, 522 issue times per series)

| series | domain-aware MAE (per day) | domain-aware rel. MAE vs naive | vs seasonal naive | coverage80 | plain rel. MAE vs naive | vs seasonal naive | coverage80 |
|---|---|---|---|---|---|---|---|
| total | 68,478 | 0.313 | **0.748** | 0.775 | 1.161 | 2.772 | 0.809 |
| bus | 38,324 | 0.314 | **0.759** | 0.773 | 1.133 | 2.740 | 0.816 |
| rail | 33,332 | 0.336 | **0.744** | 0.778 | 1.170 | 2.593 | 0.817 |
| Chicago/State | 753 | 0.516 | 0.801 | 0.780 | 1.071 | 1.664 | 0.810 |
| Clark/Lake | 892 | 0.235 | 0.702 | 0.775 | 1.039 | 3.102 | 0.840 |
| Lake/State | 1,175 | 0.412 | 0.783 | 0.785 | 1.141 | 2.169 | 0.813 |
| O'Hare Airport | 752 | 0.677 | 0.861 | 0.766 | 0.939 | 1.194 | 0.806 |
| Washington/Dearborn | 641 | 0.332 | 0.774 | 0.773 | 1.171 | 2.733 | 0.819 |
| **median** | — | **0.334** | **0.766** | **0.775** | 1.137 | 2.663 | 0.815 |

- **MASE (lag-7 scale), medians:**
  - domain-aware: 0.862;
  - seasonal naive: 1.093;
  - random walk: 2.381;
  - plain: 2.742.
- **Domain-aware:** beats both the naive and the seasonal-naive benchmark on **8 of 8** series. It cuts seasonal-naive
  error by 23 % (median) and random-walk error by 67 %.
- **Plain generic:** worse than naive on 7 of 8 series, and 2.7 × the seasonal-naive error.
- **Model choice:** the selection among domain-aware candidates varies over time. For total: Holt–Winters-calendar 210
  times, yearly seasonal naive 163, calendar naive 83, Holt–Winters 42, seasonal naive 24 (of 522).
- **Tuning window (2010–2014):** median relative MAE vs seasonal naive 0.736, coverage 0.786. The evaluation window is
  close to this, so there is no sign of overfitting.
- **Coverage:** 0.766–0.785 against the nominal 0.80. That is slightly low but inside the Phase-2 band of 0.72–0.88.
  The plain config's wider intervals (0.81) come from errors that are about 3 × larger. The served backtest over the
  last 56 origins shows lower coverage for some station series at a single issue time (0.62–0.75), because the
  interval is estimated from few residuals.
- **Wide station intervals:** station intervals can be very wide. For example, Chicago/State's day-14 band as of
  2026-06-30 is 155–620 k. Near-zero days in the station data enter the residuals. This is a data property; no outlier
  filter was added.

### 5.2 Warnings: generic vs domain-aware (evaluation window 2015–2024, 960 units, base rate 0.0625)

| config | warned | TP | FP | FN | precision | **lift** | FPR | recall | active anomalies / replay (share on weekends) | ewl flip rate (situations) | warned-status flip rate (units) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| plain generic | 738 (77 %) | 45 | 693 | 15 | 0.061 | **0.98** | 0.770 | 0.750 | 21.9 (84 %) | 0.232 | 0.187 |
| domain-aware | 222 (23 %) | 20 | 202 | 40 | 0.090 | **1.44** | 0.224 | 0.333 | 7.4 (30 %) | 0.730 | 0.292 |

By year, domain-aware (warned / TP / FP / FN):

| 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|---|---|---|
| 20/5/15/5 | 17/0/17/0 | 10/1/9/1 | 10/0/10/2 | 25/0/25/5 | 35/8/27/17 | 19/2/17/10 | 40/4/36/0 | 29/0/29/0 | 17/0/17/0 |

What this shows:

- **The generic engine alone does not work on daily data.** It warns on 77 % of units and has no skill (lift 0.98).
  84 % of its anomalies fall on weekends: every Saturday and Sunday is a "drop" against the trailing weekday-dominated
  baseline. Its low flip rate only reflects that it is almost always warned.
- **The domain-aware config reduces false positives 3.4× (FPR 0.77 → 0.22), but its warning skill is not
  statistically established.** Lift 1.44 has a month-cluster bootstrap 95 % CI of 0.52–2.36, the year-stratified
  lift is 1.18, and on the untouched 2003–09 window the same config gives lift 1.10 (CI 0.00–2.33; 3 of 16 events).
  Recall falls from 0.75 to 0.33. The hypergeometric p = 0.041 assumes independent units, which 8 correlated
  entities × 120 months are not. Source: the reproduction in
  [EVALUATION_AND_VALIDITY_REVIEW.md](EVALUATION_AND_VALIDITY_REVIEW.md) §3 (rows 3–7); the per-unit rows and
  CIs are kept locally in `experiments/cta-warning-reanalysis-20260924/`, and `scripts/evaluate_domains.py`
  now stores the rows and this CI in every report (`warnings.rows`, `warnings.lift_uncertainty`).
- **Remaining false positives:** most are single-day seasonal anomalies (weather, one-off events) and ratio-trend
  risks. A single bad day is a real anomaly but, by construction, not a sustained decline.
- **Missed events:** most are in 2020–2021. The monthly replays on the 1st cannot see the mid-month COVID collapse in
  advance, and later pandemic waves developed gradually.
- **No lead-time claim:** there were no confirmed events in 2016, 2023 and 2024, and recall in calm years is
  meaningless with so few positives.
- **The domain-aware situation flip rate is high (0.73).** The ratio-series situations toggle between NO_ACTION and
  MONITOR as their Mann–Kendall trends appear and disappear. The 28-day ratio is heavily autocorrelated, so the
  trend p-values are anti-conservative (q ≈ 0 for small drifts). This is a known weakness of testing a smoothed
  series and is reported, not fixed. The warned-status flip rate (0.29) is what an operator would see.
- **The Phase-2 bar (lift ≥ 1.5 on the held-out window) is not met:** lift 1.44, CI includes 1.

**Untouched windows (reviewer reproduction, same final config and scoring).**

| window | observable units | warned | TP | positives | lift [month-cluster 95 % CI] | FPR |
|---|---|---|---|---|---|---|
| 2015–24 (development window, §5.4) | 960 | 222 | 20 | 60 | 1.44 [0.52, 2.36]; year-stratified 1.18 | 0.224 |
| 2003–09 (never used for tuning or evaluation) | 664 | 113 | 3 | 16 | 1.10 [0.00, 2.33] | 0.170 |
| 2025-01..2026-05 | 136 | 28 | 0 | 3 | 0 (0 of 3 events caught) | 0.211 |

Without the special-day class (v0), the 2003–09 window gives 160 warned, 4 TP, FPR 0.241 and lift 1.04 [0.27, 1.98]:
the special-day class generalises as an FPR reduction (0.241 → 0.170), but neither version shows warning skill.

### 5.3 COVID and calm-period checks

Covered in §4.2 and §4.3:

- 2020: COVID onset flagged from the first two days of falling ridership (data through 13 March 2020), in a replay
  of a known event (all 8 entities by the 15 March data); with the public lag, as of 27–29 April 2020.
- Sept–Oct 2008: 19 of 61 warned days (domain-aware) against 61 of 61 (plain).

### 5.4 The one post-tuning change, and first-pass results (reported, not hidden)

The first full run (`experiments/platform-eval-20260924T171806Z`) used the same config without `anomaly_calendar`.
Special days were then in the weekday classes.

| | domain-aware v0 (eval window) | final (eval window) |
|---|---|---|
| warned / precision / lift | 358 / 0.059 / **0.94** | 222 / 0.090 / 1.44 |
| FPR / recall | 0.374 / 0.35 | 0.224 / 0.33 |

v0 had no skill. The change was chosen on the **2010–2014 tuning replays**:

- There, the most active adverse anomalies were on Christmas Eve to New Year's Eve, Thanksgiving and the day after,
  MLK Day and Presidents' Day. These are reduced-ridership days that are not CTA "Sunday service" holidays.
- Giving them their own anomaly class lowered the tuning-window FPR from 0.262 (v0). The 0.147 quoted here in an
  earlier version is the tuning FPR *after* WS4a's early-warning policy 1.1 as well (INTELLIGENCE_ENGINE_AUDIT §4.2:
  0.199 → 0.147 from policy 1.1 alone), so the special-day class accounts for roughly 0.262 → 0.199. These tuning
  replays were not saved; the figures come from unsaved tuning runs.
- The tuning window has only **3 confirmed events in 480 units**, so recall could not be tuned. The change was kept
  because it removes a known calendar confound, not because of the evaluation window.

Other variants tried on the tuning window only (none adopted; from unsaved tuning runs, no stored artefact):

| variant | warned | FPR |
|---|---|---|
| 7-day ratio instead of 28-day | 118 | 0.245 |
| anomalies as context only | 50 | 0.105 (recall 0) |
| no ratio trend | 92 | 0.191 |
| special days + 7-day ratio | 64 | 0.132 |
| special days, no ratio trend | 61 | 0.126 |

The 0.94 → 1.44 change combines two changes scored on this window: the special-day class (0.94 → 1.27, WS4b) and
WS4a's early-warning policy 1.1 (1.27 → 1.44). The 2015–24 window has been scored at least four times (WS4b v0,
WS4a before and after 1.1, and the WS4a ablations and K study), so it is a development window, not a hold-out. The
2003–09 window is the untouched check (§5.2): lift 1.10, CI 0.00–2.33.

### 5.5 US unemployment re-evaluated

- **Served config unchanged.** Its synthetic golden is byte-identical.
- **v1.2 protocol** (24 origins in 2024–26, select on half, score on half): MASE 1.52 against naive 1.41; coverage
  0.61. These numbers are unchanged.
- **Rolling replay of the served forecast** (monthly issue times 2005–2026, h = 6, training-only selection):
  - v1.2 trio (naive, moving average, damped Holt on levels): median relative MAE vs naive **0.993**; below 1 on 10 of
    17 series (best: CA 0.84, South region 0.86; worst: NC 1.12); coverage **0.69**.
  - Adding `drift` and `theta`: median **0.986**, again 10 of 17 below 1, coverage 0.69.
- **Verdict:** over the long window the forecasts are about as good as naive, not clearly better. The 24-origin
  protocol happens to score a calm period in which naive is hard to beat. The 80 % intervals under-cover in both
  protocols (0.61–0.69), because the residuals of calm periods do not anticipate shocks.
- **Decision:** a 0.7 % gain does not justify changing what is served. No monthly seasonal model applies, since the
  data is seasonally adjusted.

## 6. Limitations

- **No demonstrated warning skill.** The domain-aware config reduces false positives 3.4× (FPR 0.77 → 0.22), but its
  warning skill is not statistically established: lift 1.44 has a month-cluster bootstrap 95 % CI of 0.52–2.36, the
  year-stratified lift is 1.18, and on the untouched 2003–09 window the same config gives lift 1.10 (CI 0.00–2.33;
  3 of 16 events). The early-warning policy (`ewl-1.0.0`) is shared with the other
  domains and was not tuned for daily data. A persistence requirement for daily anomalies would be the natural next
  step (a WS4a policy question).
- **The ratio-trend test is anti-conservative** on a smoothed series, which causes the high situation flip rate.
- **Operator vs public lag:** the COVID onset detection assumes next-day data. The public portal gives a warning about 6
  weeks later.
- **Current vintage only:** revisions of the portal data are not modelled (as for unemployment).
- **Calendar:** Easter, school breaks and events (marathon, parades) are not in the calendar. Weather is not modelled
  at all, so storms produce real but unsustained anomalies.
- **Stations:** the stations are the top 5 of 2019. That choice was made before the evaluation, but it favours
  downtown stations. Their near-zero days widen the intervals.
- **Cosmetic issues in WS4a modules:** see docs/platform.md (daily signal freshness; a month label in risk titles for
  daily anomalies).

## 7. Reproduction

```bash
uv run python scripts/download_domain_data.py cta-ridership        # ~1 min, no key; writes provenance.json
uv run python scripts/run_intelligence.py --domain generic:cta-ridership                 # default (latest data)
uv run python scripts/run_intelligence.py --domain generic:cta-ridership --as-of 2020-03-20
uv run python scripts/run_intelligence.py --domain generic:cta-ridership --as-of 2008-10-01
uv run python scripts/evaluate_domains.py --domains cta-ridership,us-unemployment       # ~6 min; writes
#   experiments/platform-eval-<ts>/{report.json, REPORT.md, runs/cta_{default,2020-03-20,2008-10-01,2021-07-01}.json}
uv run pytest -q tests/core/test_forecast_seasonal.py tests/domains   # synthetic, no network
```

Scenarios on any result: `jev_ml.core.scenario.run_scenario(result, {"series_id": "boardings:total",
"horizon_months": 14, "scenarios": [...]})`. The horizon is in days for this domain.
