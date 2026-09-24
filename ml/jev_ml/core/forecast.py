"""PREDICT (forecasts): naive, moving average and damped Holt with a rolling-origin backtest.

* Count series are modelled on log1p scale (multiplicative bursts), others on their raw scale.
* Models: ``naive`` (last value), ``moving_average`` (mean of the last k months), ``holt_damped``
  (additive damped trend, hand-implemented; alpha/beta/phi picked from a small grid by in-sample
  one-step SSE on the training part of each origin only).
* Backtest: for each of the last N origins, fit on the data *before* the origin and forecast
  h = 1..horizon; errors are scored on the raw scale. MASE uses the in-sample one-step naive MAE of that
  origin's training window as scale (Hyndman & Koehler 2006); ``naive_mase`` is the naive model's
  MASE on the same points, so ``mase < naive_mase`` means the model beats naive.
* The model with the lowest backtest MASE is selected per series (ties -> simpler model).
* 80 % intervals: 10 %/90 % finite-sample (conformal) order statistics of the selected model's
  backtest residuals per horizon step (transformed scale). ``coverage80`` is measured honestly:
  each origin's interval is
  built only from residuals whose target month precedes that origin.

Seasonal path (opt-in per series: the spec declares ``forecast_models`` and/or ``seasonal_period``;
every other series takes the v1.2 path above unchanged, byte for byte):

* Time alignment is kept: gaps stay NaN (never 0), are filled for *fitting* with the value of the
  previous period of the same season class (else the previous value) and are never scored.
* Extra models: ``drift`` (random walk with the mean historical change), ``theta`` (Hyndman &
  Billah 2003: simple exponential smoothing plus half the OLS slope), ``seasonal_naive`` (value one
  season ago), ``seasonal_naive_yearly`` (364 days / 52 weeks / 12 months ago),
  ``calendar_naive`` (the latest period of the same season class, e.g. the last Sunday or holiday
  for a holiday), ``holt_winters`` (additive damped-trend Holt–Winters, ETS(A,Ad,A) in
  error-correction form, season = position in the cycle) and ``holt_winters_calendar`` (the same
  with the season index taken from the calendar class, so a holiday gets the Sunday index). On
  log1p scale for counts, i.e. multiplicative seasonality. Holt–Winters parameters come from a
  small grid (``HW_*``) by one-step SSE over each origin's own training window; one causal pass
  over the data yields the state at every origin, so no origin sees a later value.
* MASE is scaled by the in-sample *seasonal* naive MAE (lag m) when a seasonal period is declared
  (Hyndman & Koehler 2006), and the benchmark ``naive_mase`` is then the **seasonal naive** model's
  MASE (the naive benchmark of a seasonal series); ``random_walk_mase`` keeps the plain naive
  model's. So "beats naive" downstream (risk skill, signals) means "beats seasonal naive".
* ``rolling_evaluation`` replays the whole served procedure (training-only selection among the
  candidates on the previous origins, conformal intervals from residuals realised before the issue
  time) at many issue times and scores it against later data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import fnum, period_str, short_hash, stable_id
from jev_ml.core.config import FORECAST_VERSION, CoreConfig
from jev_ml.core.series import FREQ_NAMES, Series, season_classes

MODELS = ("naive", "moving_average", "holt_damped")
TREND_MODELS = ("drift", "theta")
SEASONAL_MODELS = (
    "seasonal_naive",
    "seasonal_naive_yearly",
    "calendar_naive",
    "holt_winters",
    "holt_winters_calendar",
)
ALL_MODELS = MODELS + TREND_MODELS + SEASONAL_MODELS
YEARLY_LAG = {"D": 364, "W": 52, "M": 12}
# Holt–Winters grid (additive, log scale for counts): smoothing of level (alpha), season (gamma) and
# damped trend (beta, phi); phi = 0 is "no trend"
HW_ALPHAS = (0.05, 0.1, 0.2, 0.35, 0.5)
HW_GAMMAS = (0.02, 0.05, 0.1, 0.2)
HW_TRENDS = ((0.0, 0.0), (0.02, 0.9), (0.05, 0.98))
THETA_ALPHAS = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
# movie series routed to forecasting (kept for callers of the v1.1 API; the pipeline routes by the
# series spec's ``forecast`` / ``share_forecast`` flags)
FORECAST_PREFIXES = ("volume:all", "active_users:all", "volume:genre:")
SHARE_PREFIX = "share:genre:"


@dataclass
class ForecastState:
    """Everything a scenario needs to re-use a fitted forecast (not serialised)."""

    series_id: str
    model: str
    transform: str  # log1p | identity
    history_months: pd.PeriodIndex
    history: np.ndarray  # raw complete values
    path: np.ndarray  # transformed-scale point forecast, length horizon
    trend: np.ndarray  # transformed-scale trend component of `path` (zeros for no-trend models)
    q_lo: np.ndarray  # residual quantiles per step (transformed scale)
    q_hi: np.ndarray
    params: dict[str, Any] = field(default_factory=dict)
    backtest: dict[str, Any] = field(default_factory=dict)
    bt: _Backtest | None = None  # the selected model's backtest (window aggregates, not serialised)
    context: SeasonContext | None = None  # seasonal path only (None: the v1.2 path)

    def months_ahead(self, h: int) -> list[pd.Period]:
        last = self.history_months[-1]
        return [last + i for i in range(1, h + 1)]


def _tf(transform: str, x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0)) if transform == "log1p" else x


def _itf(transform: str, x: np.ndarray) -> np.ndarray:
    return np.maximum(np.expm1(x), 0.0) if transform == "log1p" else x


# --------------------------------------------------------------------------------------------------
# models (inputs on transformed scale; return (path, trend, params))


def fc_naive(y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return np.full(horizon, y[-1]), np.zeros(horizon), {}


def fc_ma(y: np.ndarray, horizon: int, k: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return np.full(horizon, float(np.mean(y[-k:]))), np.zeros(horizon), {"window": int(min(k, len(y)))}


def holt_fit(
    y: np.ndarray, alphas: tuple[float, ...], betas: tuple[float, ...], phis: tuple[float, ...]
) -> tuple[float, float, dict[str, float]]:
    """Grid-fit additive damped Holt by one-step SSE (vectorised over the grid).

    Returns (final level, final trend, params)."""
    grid = np.array([(a, b, p) for a in alphas for b in betas for p in phis], dtype=float)
    a, b, ph = grid[:, 0], grid[:, 1], grid[:, 2]
    n = len(y)
    level = np.full(len(grid), y[0])
    trend = np.full(len(grid), (y[min(3, n - 1)] - y[0]) / max(min(3, n - 1), 1))
    sse = np.zeros(len(grid))
    for t in range(1, n):
        pred = level + ph * trend
        err = y[t] - pred
        if t >= 2:
            sse += err**2
        new_level = pred + a * err
        trend = ph * trend + a * b * err  # error-correction form of Holt's trend update
        level = new_level
    i = int(np.argmin(sse))
    return float(level[i]), float(trend[i]), {"alpha": float(a[i]), "beta": float(b[i]), "phi": float(ph[i])}


def fc_holt(y: np.ndarray, horizon: int, cfg: CoreConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    lvl, tr, params = holt_fit(y, cfg.holt_alphas, cfg.holt_betas, cfg.holt_phis)
    phi = params["phi"]
    damp = np.cumsum(phi ** np.arange(1, horizon + 1))
    trend = damp * tr
    return lvl + trend, trend, {**params, "level": lvl, "trend": tr}


def run_model(
    name: str, y: np.ndarray, horizon: int, cfg: CoreConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = y[-cfg.forecast_history_months :]
    if name == "naive":
        return fc_naive(y, horizon)
    if name == "moving_average":
        return fc_ma(y, horizon, cfg.forecast_ma_window)
    if name == "holt_damped":
        return fc_holt(y, horizon, cfg)
    raise ValueError(f"unknown forecast model {name}")


# --------------------------------------------------------------------------------------------------
# backtest


@dataclass
class _Backtest:
    origins: list[int]
    resid: np.ndarray  # (n_origins, horizon) transformed-scale residuals (NaN where no truth)
    abs_err: np.ndarray  # raw-scale |error|
    scaled: np.ndarray  # |error| / MASE scale
    smape: np.ndarray
    pred_raw: np.ndarray
    truth_raw: np.ndarray
    target: np.ndarray  # index of target month (-1 = none)


def backtest_model(
    name: str, raw: np.ndarray, transform: str, cfg: CoreConfig, horizon: int, n_origins: int
) -> _Backtest | None:
    n = len(raw)
    first = max(cfg.forecast_min_history, n - n_origins)
    origins = list(range(first, n))
    if not origins:
        return None
    y = _tf(transform, raw)
    shape = (len(origins), horizon)
    resid = np.full(shape, np.nan)
    abs_err = np.full(shape, np.nan)
    scaled = np.full(shape, np.nan)
    smape = np.full(shape, np.nan)
    pred_raw = np.full(shape, np.nan)
    truth_raw = np.full(shape, np.nan)
    target = np.full(shape, -1)
    for j, o in enumerate(origins):
        hist = raw[max(0, o - cfg.forecast_history_months) : o]
        scale = float(np.mean(np.abs(np.diff(hist)))) if len(hist) > 1 else 0.0
        path, _, _ = run_model(name, y[:o], horizon, cfg)
        steps = min(horizon, n - o)
        f_raw = _itf(transform, path[:steps])
        t_raw = raw[o : o + steps]
        resid[j, :steps] = y[o : o + steps] - path[:steps]
        abs_err[j, :steps] = np.abs(t_raw - f_raw)
        if scale > 0:
            scaled[j, :steps] = abs_err[j, :steps] / scale
        denom = np.abs(t_raw) + np.abs(f_raw)
        smape[j, :steps] = np.where(denom > 0, 2 * abs_err[j, :steps] / np.where(denom > 0, denom, 1), 0.0)
        pred_raw[j, :steps] = f_raw
        truth_raw[j, :steps] = t_raw
        target[j, :steps] = np.arange(o, o + steps)
    return _Backtest(origins, resid, abs_err, scaled, smape, pred_raw, truth_raw, target)


def step_quantiles(resid: np.ndarray, cfg: CoreConfig, horizon: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-step residual quantiles; steps with too few residuals pool steps <= h."""
    lo_q = (1 - cfg.forecast_interval) / 2
    hi_q = 1 - lo_q
    lo = np.full(horizon, np.nan)
    hi = np.full(horizon, np.nan)
    for h in range(horizon):
        col = resid[:, h] if h < resid.shape[1] else np.array([])
        col = col[np.isfinite(col)]
        if len(col) < cfg.forecast_min_residuals:
            pooled = resid[:, : min(h + 1, resid.shape[1])]
            col = pooled[np.isfinite(pooled)]
        if len(col) < cfg.forecast_min_residuals:
            continue
        # finite-sample (split-conformal) order statistics: with n residuals the k-th smallest,
        # k = ceil((n + 1) * q), covers a new residual with probability >= q; plain empirical
        # quantiles of few residuals under-cover
        srt = np.sort(col)
        n = len(srt)
        k_hi = min(n, int(np.ceil((n + 1) * hi_q)))
        k_lo = max(1, int(np.floor((n + 1) * lo_q)))
        lo[h], hi[h] = srt[k_lo - 1], srt[k_hi - 1]
    return (lo, hi) if np.isfinite(lo).any() else None


def honest_coverage(bt: _Backtest, cfg: CoreConfig, horizon: int) -> float | None:
    """Coverage where each origin's interval only uses residuals realised before that origin."""
    hits = total = 0
    for j, o in enumerate(bt.origins):
        past = np.where(bt.target < o, bt.resid, np.nan)
        q = step_quantiles(past, cfg, horizon)
        if q is None:
            continue
        lo, hi = q
        for h in range(horizon):
            r = bt.resid[j, h]
            if not np.isfinite(r) or not np.isfinite(lo[h]):
                continue
            total += 1
            hits += int(lo[h] - 1e-12 <= r <= hi[h] + 1e-12)
    return hits / total if total else None


def _metrics(bt: _Backtest) -> dict[str, float | None]:
    return {
        "mase": fnum(np.nanmean(bt.scaled), 4) if np.isfinite(bt.scaled).any() else None,
        "smape": fnum(np.nanmean(bt.smape), 4),
        "mae": fnum(np.nanmean(bt.abs_err), 3),
    }


def select_and_backtest(
    raw: np.ndarray, transform: str, cfg: CoreConfig, horizon: int, n_origins: int
) -> dict[str, Any] | None:
    """Backtest every model; return selection info or None when history is too short."""
    results: dict[str, tuple[_Backtest, dict[str, float | None]]] = {}
    for name in MODELS:
        bt = backtest_model(name, raw, transform, cfg, horizon, n_origins)
        if bt is None:
            return None
        results[name] = (bt, _metrics(bt))
    naive_mase = results["naive"][1]["mase"]

    def key(name: str) -> tuple[float, int]:
        m = results[name][1]["mase"]
        return (m if m is not None else float("inf"), MODELS.index(name))

    best = min(MODELS, key=key)
    bt, met = results[best]
    return {
        "model": best,
        "bt": bt,
        "metrics": {
            "origins": len(bt.origins),
            **met,
            "coverage80": fnum(honest_coverage(bt, cfg, horizon), 4),
            "naive_mase": naive_mase,
        },
        "all_models": {n: r[1] for n, r in results.items()},
    }


def forecast_series(
    s: Series, cfg: CoreConfig, as_of_key: str, data_version: str, horizon: int | None = None
) -> tuple[dict[str, Any], ForecastState] | None:
    horizon = horizon or cfg.forecast_horizon
    if s.seasonal:
        return forecast_series_seasonal(s, cfg, as_of_key, data_version, horizon)
    months, raw, _ = s.complete()
    raw = np.where(np.isfinite(raw), raw, 0.0) if s.is_count else raw
    if not s.is_count:
        ok = np.isfinite(raw)
        months, raw = months[ok], raw[ok]
    if len(raw) < cfg.forecast_min_history + 2:
        return None
    transform = "log1p" if s.is_count else "identity"
    sel = select_and_backtest(raw, transform, cfg, horizon, cfg.forecast_backtest_origins)
    if sel is None:
        return None
    model = sel["model"]
    y = _tf(transform, raw)
    path, trend, params = run_model(model, y, horizon, cfg)
    q = step_quantiles(sel["bt"].resid, cfg, horizon)
    q_lo, q_hi = q if q is not None else (np.full(horizon, np.nan), np.full(horizon, np.nan))
    state = ForecastState(
        series_id=s.id,
        model=model,
        transform=transform,
        history_months=months,
        history=raw,
        path=path,
        trend=trend,
        q_lo=q_lo,
        q_hi=q_hi,
        params=params,
        backtest=sel["metrics"],
        bt=sel["bt"],
    )
    hist_n = min(len(raw), cfg.forecast_history_months)
    fc_params = {
        "horizon": horizon,
        "origins": cfg.forecast_backtest_origins,
        "history": cfg.forecast_history_months,
        "ma": cfg.forecast_ma_window,
        "grid": [cfg.holt_alphas, cfg.holt_betas, cfg.holt_phis],
        "interval": cfg.forecast_interval,
    }
    out = {
        "id": stable_id("fc", "forecast", s.id, as_of_key),
        "series_id": s.id,
        "entity": s.entity,
        "entity_type": s.entity_type,
        "metric": s.metric,
        "model": model,
        "model_version": f"{FORECAST_VERSION}-{short_hash([fc_params, data_version, as_of_key])}",
        "model_params": {k: fnum(v, 5) for k, v in params.items()},
        "horizon_months": horizon,
        "issued_at": as_of_key,
        "features_used": [
            f"{'log1p(' + s.metric + ')' if transform == 'log1p' else s.metric} history, {hist_n} m"
        ],
        "points": scenario_points(state, path, horizon),
        "backtest": sel["metrics"],
        "backtest_all_models": sel["all_models"],
        "horizon_unit": "week" if s.freq == "W" else "month",
        "entity_id": s.entity_id or f"{s.entity_type}:{s.entity}",
        "adverse_direction": s.adverse_direction,
    }
    return out, state


def rerun(state: ForecastState, horizon: int, cfg: CoreConfig) -> tuple[np.ndarray, np.ndarray]:
    """The selected model's (path, trend) from the end of the history for any horizon (scenarios)."""
    if state.context is None:
        y = np.log1p(np.maximum(state.history, 0)) if state.transform == "log1p" else state.history
        path, trend, _ = run_model(state.model, y, horizon, cfg)
        return path, trend
    ctx = state.context
    y = _impute(_tf(state.transform, state.history), ctx.classes(0), ctx.period)
    n = len(y)
    P, T, _ = model_paths(state.model, y, np.array([n]), horizon, ctx, cfg)
    return P[0], T[0]


def interval_at(state: ForecastState, h: int) -> tuple[float, float]:
    """Residual quantiles for step h (0-based). Beyond the backtest horizon the last step's
    quantiles are widened by sqrt(h / horizon) (random-walk growth assumption)."""
    horizon = len(state.q_lo)
    if h < horizon:
        return float(state.q_lo[h]), float(state.q_hi[h])
    f = np.sqrt((h + 1) / horizon)
    return float(state.q_lo[-1] * f), float(state.q_hi[-1] * f)


def scenario_points(state: ForecastState, path: np.ndarray, horizon: int) -> list[dict[str, Any]]:
    months = state.months_ahead(horizon)
    mean = _itf(state.transform, path)
    pts = []
    for h in range(horizon):
        lo, hi = interval_at(state, h)
        if np.isfinite(lo):
            lo_v = float(_itf(state.transform, np.array([path[h] + lo]))[0])
            hi_v = float(_itf(state.transform, np.array([path[h] + hi]))[0])
        else:
            lo_v = hi_v = float("nan")
        pts.append(
            {
                "t": period_str(months[h]),
                "mean": fnum(mean[h], 4),
                "lo80": fnum(lo_v, 4),
                "hi80": fnum(hi_v, 4),
            }
        )
    return pts


def window_mean_forecast(state: ForecastState, window: int, cfg: CoreConfig) -> dict[str, Any] | None:
    """Mean of the first ``window`` forecast months with a conformal interval for that mean.

    The interval uses the selected model's backtest residuals *of the window mean* (origins whose
    whole window was observed), with the same finite-sample order statistics as the per-step
    intervals; averaging per-step bounds instead would ignore how errors of adjacent months
    correlate. ``coverage`` is measured honestly: each origin's interval uses only window residuals
    whose last target month precedes that origin. Identity-transform series only (shares, ratings);
    None when there are too few complete windows."""
    bt = state.bt
    if bt is None or state.transform != "identity" or window < 1 or window > bt.resid.shape[1]:
        return None
    res = bt.resid[:, :window]
    full = np.isfinite(res).all(axis=1)
    wres = np.where(full, res.mean(axis=1) if res.size else np.nan, np.nan)
    last_target = bt.target[:, window - 1]
    q = _order_stat_interval(wres[full], cfg)
    if q is None:
        return None
    hits = total = 0
    for j, o in enumerate(bt.origins):
        if not full[j]:
            continue
        past = wres[full & (last_target < o) & (last_target >= 0)]
        qj = _order_stat_interval(past, cfg)
        if qj is None:
            continue
        total += 1
        hits += int(qj[0] - 1e-12 <= wres[j] <= qj[1] + 1e-12)
    mean = float(np.mean(state.path[:window]))
    months = state.months_ahead(window)
    return {
        "months": [period_str(m) for m in months],
        "mean": mean,
        "lo": mean + q[0],
        "hi": mean + q[1],
        "n_residuals": int(full.sum()),
        "coverage": hits / total if total else None,
        "coverage_origins": total,
        "nominal": cfg.forecast_interval,
    }


def _order_stat_interval(r: np.ndarray, cfg: CoreConfig) -> tuple[float, float] | None:
    r = r[np.isfinite(r)]
    if len(r) < cfg.forecast_min_residuals:
        return None
    lo_q = (1 - cfg.forecast_interval) / 2
    srt = np.sort(r)
    n = len(srt)
    k_hi = min(n, int(np.ceil((n + 1) * (1 - lo_q))))
    k_lo = max(1, int(np.floor((n + 1) * lo_q)))
    return float(srt[k_lo - 1]), float(srt[k_hi - 1])


def forecast_shares(
    series: list[Series], cfg: CoreConfig, as_of_key: str, data_version: str, min_share: float = 0.0
) -> tuple[list[dict[str, Any]], dict[str, ForecastState]]:
    """Forecasts of share series (spec flag ``share_forecast``; published separately as
    ``predictions.share_forecasts``). Only series whose recent mean share (last
    ``trend_window_months`` complete periods) is >= ``min_share``."""
    out, states = [], {}
    for s in series:
        if not s.share_forecast:
            continue
        _, raw, _ = s.complete()
        recent = raw[-cfg.trend_window_months :]
        recent = recent[np.isfinite(recent)]
        if not len(recent) or float(np.mean(recent)) < min_share:
            continue
        res = forecast_series(s, cfg, as_of_key, data_version)
        if res is None:
            continue
        out.append(res[0])
        states[s.id] = res[1]
    return out, states


def forecast_all(
    series: list[Series], cfg: CoreConfig, as_of_key: str, data_version: str
) -> tuple[list[dict[str, Any]], dict[str, ForecastState]]:
    """Forecasts of every series whose spec sets ``forecast``."""
    out, states = [], {}
    for s in series:
        if not s.forecast:
            continue
        res = forecast_series(s, cfg, as_of_key, data_version)
        if res is None:
            continue
        out.append(res[0])
        states[s.id] = res[1]
    return out, states


# --------------------------------------------------------------------------------------------------
# seasonal path (opt-in per series; see the module docstring)


@dataclass
class SeasonContext:
    """Season information of one series' complete history (and its future periods)."""

    freq: str  # M | W | D
    period: int | None  # seasonal period m (None: no seasonality declared)
    calendar: str | None
    periods: pd.PeriodIndex  # the history's periods
    _cache: dict[int, np.ndarray] = field(default_factory=dict)

    def classes(self, n_future: int) -> np.ndarray:
        """Season classes of the history followed by ``n_future`` future periods."""
        if n_future not in self._cache:
            last = self.periods[-1]
            fut = pd.period_range(last + 1, periods=n_future, freq=last.freq) if n_future else None
            idx = self.periods if fut is None else self.periods.append(fut)
            self._cache[n_future] = season_classes(idx, self.period, self.calendar)
        return self._cache[n_future]

    @property
    def yearly_lag(self) -> int:
        return YEARLY_LAG[self.freq]


def candidate_models(s: Series) -> tuple[str, ...]:
    if s.forecast_models is not None:
        return tuple(s.forecast_models)
    base: tuple[str, ...] = ("naive", "seasonal_naive", "holt_winters")
    return base + (("calendar_naive", "holt_winters_calendar") if s.calendar else ())


def _impute(y: np.ndarray, classes: np.ndarray, period: int | None) -> np.ndarray:
    """Fill NaN (gaps) with the value of the latest earlier period of the same season class, else
    the previous value; leading NaNs take the first value. Fitting only: gaps are never scored."""
    out = np.asarray(y, dtype=float).copy()
    bad = np.flatnonzero(~np.isfinite(out))
    if not len(bad):
        return out
    last_of: dict[int, float] = {}
    prev = np.nan
    for t in range(len(out)):
        v = out[t]
        if not np.isfinite(v):
            c = int(classes[t]) if period else -1
            v = last_of.get(c, prev) if period else prev
            out[t] = v
        if np.isfinite(v):
            if period:
                last_of[int(classes[t])] = float(v)
            prev = float(v)
    first = np.flatnonzero(np.isfinite(out))
    if len(first):
        out[: first[0]] = out[first[0]]
    return out


def _hw_pass(
    y: np.ndarray,
    cls: np.ndarray,
    origins: np.ndarray,
    horizon: int,
    cap: int,
    period: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Additive damped-trend Holt–Winters for every grid point in one causal pass.

    ``cls`` holds the season class of every period (history + horizon); the state at origin o is
    the state after y[:o]. At each origin the grid point with the lowest one-step SSE over that
    origin's training window (the ``cap`` periods before it, after a two-cycle warm-up) is used.
    Returns (paths, trend parts, params) per origin (transformed scale)."""
    grid = np.array(
        [(a, b, ph, g) for a in HW_ALPHAS for (b, ph) in HW_TRENDS for g in HW_GAMMAS], dtype=float
    )
    A, B, PH, GM = grid[:, 0], grid[:, 1], grid[:, 2], grid[:, 3]
    G = len(grid)
    n_cls = int(cls.max()) + 1 if len(cls) else 1
    o_min, o_max = int(origins.min()), int(origins.max())
    s0 = max(0, o_min - cap)
    warm = min(2 * period, max(1, o_min - s0 - 1))
    init = y[s0 : s0 + warm]
    level = np.full(G, float(np.mean(init)))
    seas = np.zeros((G, n_cls))
    for c in range(n_cls):
        m = cls[s0 : s0 + warm] == c
        if m.any():
            seas[:, c] = float(np.mean(init[m])) - level[0]
    trend = np.zeros(G)
    T = o_max - s0 + 1
    cum = np.zeros((T + 1, G))  # cum[k] = SSE of one-step errors of y[s0 + warm .. s0 + k - 1]
    states: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    want = set(int(o) for o in origins)
    for t in range(s0, o_max + 1):
        if t in want:
            states[t] = (level.copy(), trend.copy(), seas.copy())
        if t == o_max or t >= len(y):
            cum[t - s0 + 1] = cum[t - s0]
            continue
        c = int(cls[t])
        pred = level + PH * trend + seas[:, c]
        err = y[t] - pred
        cum[t - s0 + 1] = cum[t - s0] + (err**2 if t >= s0 + warm else 0.0)
        new_level = pred - seas[:, c] + A * err
        trend = PH * trend + A * B * err
        seas[:, c] = seas[:, c] + GM * err
        level = new_level
    paths = np.full((len(origins), horizon), np.nan)
    trends = np.zeros((len(origins), horizon))
    params: list[dict[str, float]] = []
    for j, o in enumerate(origins):
        o = int(o)
        lo = max(s0 + warm, o - cap)
        sse = cum[o - s0] - cum[max(lo - s0, 0)]
        g = int(np.argmin(sse))
        lv, tr, se = states[o]
        damp = np.cumsum(PH[g] ** np.arange(1, horizon + 1)) if PH[g] > 0 else np.zeros(horizon)
        tpart = damp * tr[g]
        paths[j] = lv[g] + tpart + se[g, cls[o : o + horizon]]
        trends[j] = tpart
        params.append(
            {
                "alpha": float(A[g]),
                "beta": float(B[g]),
                "phi": float(PH[g]),
                "gamma": float(GM[g]),
                "level": float(lv[g]),
                "trend": float(tr[g]),
            }
        )
    return paths, trends, params


def _theta(y: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(y)
    t = np.arange(n, dtype=float)
    b = float(np.polyfit(t, y, 1)[0]) if n > 2 else 0.0
    best = (np.inf, 0.5, float(y[-1]))
    for a in THETA_ALPHAS:
        lv, sse = float(y[0]), 0.0
        for v in y[1:]:
            e = float(v) - lv
            sse += e * e
            lv += a * e
        if sse < best[0]:
            best = (sse, a, lv)
    _, a, lv = best
    tr = (b / 2.0) * (np.arange(horizon) + (1 - (1 - a) ** n) / a)
    return lv + tr, tr, {"alpha": a, "slope": b, "level": lv}


def model_paths(
    name: str, y: np.ndarray, origins: np.ndarray, horizon: int, ctx: SeasonContext, cfg: CoreConfig
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Transformed-scale forecasts from every origin (row j: fitted on y[:origins[j]] only).
    Returns (paths, trend parts, params of the last origin)."""
    cap = cfg.forecast_history_months
    O = len(origins)
    steps = np.arange(horizon)
    P = np.full((O, horizon), np.nan)
    T = np.zeros((O, horizon))
    params: dict[str, Any] = {}
    m = ctx.period or 1
    if name in ("naive", "moving_average", "holt_damped"):
        for j, o in enumerate(origins):
            P[j], T[j], params = run_model(name, y[: int(o)], horizon, cfg)
        return P, T, params
    if name == "drift":
        for j, o in enumerate(origins):
            w = y[max(0, int(o) - cap) : int(o)]
            d = (w[-1] - w[0]) / (len(w) - 1) if len(w) > 1 else 0.0
            T[j] = d * (steps + 1)
            P[j] = w[-1] + T[j]
        return P, T, {"drift": float(d)}
    if name == "theta":
        for j, o in enumerate(origins):
            P[j], T[j], params = _theta(y[max(0, int(o) - cap) : int(o)], horizon)
        return P, T, params
    if name in ("seasonal_naive", "seasonal_naive_yearly"):
        lag = m if name == "seasonal_naive" else ctx.yearly_lag
        k = np.ceil((steps + 1) / lag).astype(int)
        idx = origins[:, None] + steps[None, :] - lag * k[None, :]
        ok = idx >= 0
        P = np.where(ok, y[np.clip(idx, 0, len(y) - 1)], np.nan)
        return P, T, {"lag": lag}
    cls = ctx.classes(max(0, int(origins.max()) + horizon - len(y)))
    if name == "calendar_naive":
        hist_cls = cls[: len(y)]
        pos_of = {int(c): np.flatnonzero(hist_cls == c) for c in np.unique(hist_cls)}
        for h in range(horizon):
            tc = cls[origins + h]
            for c in np.unique(tc):
                sel = np.flatnonzero(tc == c)
                pos = pos_of.get(int(c), np.zeros(0, dtype=np.int64))
                k = np.searchsorted(pos, origins[sel]) - 1  # latest same-class period before o
                fall = origins[sel] - 1
                P[sel, h] = y[np.where(k >= 0, pos[np.clip(k, 0, max(len(pos) - 1, 0))] if len(pos) else fall, fall)]
        return P, T, {}
    if name in ("holt_winters", "holt_winters_calendar"):
        if name == "holt_winters":
            pos = np.arange(len(cls)) % m  # position in the cycle, ignoring the calendar
            c_use = pos
        else:
            c_use = cls
        P, T, plist = _hw_pass(y, c_use, origins, horizon, cap, m)
        return P, T, plist[-1]
    raise ValueError(f"unknown forecast model {name}")


def _scales(raw: np.ndarray, origins: np.ndarray, lag: int, cap: int) -> np.ndarray:
    """In-sample (seasonal) naive MAE of each origin's training window (MASE scale)."""
    d = np.full(len(raw), np.nan)
    if lag < len(raw):
        d[lag:] = np.abs(raw[lag:] - raw[:-lag])
    ok = np.isfinite(d)
    cs = np.r_[0.0, np.cumsum(np.where(ok, d, 0.0))]
    cn = np.r_[0, np.cumsum(ok)]
    out = np.zeros(len(origins))
    for j, o in enumerate(origins):
        lo = max(0, int(o) - cap)
        n = cn[int(o)] - cn[lo]
        out[j] = (cs[int(o)] - cs[lo]) / n if n else 0.0
    return out


def _bt_from_paths(
    P: np.ndarray, y: np.ndarray, raw: np.ndarray, origins: np.ndarray, transform: str, scales: np.ndarray
) -> _Backtest:
    n = len(raw)
    horizon = P.shape[1]
    tgt = origins[:, None] + np.arange(horizon)[None, :]
    inside = tgt < n
    tc = np.clip(tgt, 0, n - 1)
    truth_raw = np.where(inside, raw[tc], np.nan)
    truth_y = np.where(inside, y[tc], np.nan)
    f_raw = _itf(transform, P)
    resid = truth_y - P
    abs_err = np.abs(truth_raw - f_raw)
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(scales[:, None] > 0, abs_err / np.where(scales > 0, scales, 1.0)[:, None], np.nan)
        denom = np.abs(truth_raw) + np.abs(f_raw)
        smape = np.where(denom > 0, 2 * abs_err / np.where(denom > 0, denom, 1), 0.0)
    smape = np.where(np.isfinite(abs_err), smape, np.nan)
    target = np.where(inside, tgt, -1)
    return _Backtest(
        [int(o) for o in origins],
        resid,
        abs_err,
        scaled,
        smape,
        np.where(inside, f_raw, np.nan),
        truth_raw,
        target,
    )


@dataclass
class SeasonalData:
    months: pd.PeriodIndex
    raw: np.ndarray  # complete values, NaN = gap (leading gaps trimmed)
    y: np.ndarray  # transformed, gaps filled for fitting
    transform: str
    ctx: SeasonContext
    models: tuple[str, ...]
    lag: int  # MASE scale lag (the seasonal period, else 1)
    benchmark: str


def seasonal_data(s: Series, cfg: CoreConfig) -> SeasonalData | None:
    months, raw, _ = s.complete()
    raw = np.asarray(raw, dtype=float)
    fin = np.flatnonzero(np.isfinite(raw))
    if not len(fin):
        return None
    months, raw = months[fin[0] :], raw[fin[0] :]
    transform = "log1p" if s.is_count else "identity"
    ctx = SeasonContext(s.freq, s.seasonal_period, s.calendar, months)
    y = _impute(_tf(transform, raw), ctx.classes(0), s.seasonal_period)
    models = candidate_models(s)
    lag = int(s.seasonal_period or 1)
    bench = "seasonal_naive" if s.seasonal_period and "seasonal_naive" in models else "naive"
    return SeasonalData(months, raw, y, transform, ctx, models, lag, bench)


def _usable(name: str, first_origin: int, ctx: SeasonContext) -> bool:
    need = {"seasonal_naive_yearly": ctx.yearly_lag}.get(name, 0)
    if name in ("holt_winters", "holt_winters_calendar"):
        need = 2 * (ctx.period or 1) + 2
    return first_origin >= need


def backtest_all(
    sd: SeasonalData, cfg: CoreConfig, horizon: int, origins: np.ndarray
) -> tuple[dict[str, _Backtest], dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]]]:
    """Backtests of every usable candidate at ``origins`` (the last origin may be len(history): the
    served forecast, which has no truth yet). Also returns each model's raw paths."""
    scales = _scales(sd.raw, origins, sd.lag, cfg.forecast_history_months)
    bts, paths = {}, {}
    for name in sd.models:
        if not _usable(name, int(origins.min()), sd.ctx):
            continue
        P, T, par = model_paths(name, sd.y, origins, horizon, sd.ctx, cfg)
        paths[name] = (P, T, par)
        bts[name] = _bt_from_paths(P, sd.y, sd.raw, origins, sd.transform, scales)
    return bts, paths


def _bt_rows(bt: _Backtest, rows: np.ndarray) -> _Backtest:
    return _Backtest(
        [bt.origins[i] for i in rows],
        bt.resid[rows],
        bt.abs_err[rows],
        bt.scaled[rows],
        bt.smape[rows],
        bt.pred_raw[rows],
        bt.truth_raw[rows],
        bt.target[rows],
    )


def _m(bt: _Backtest) -> dict[str, float | None]:
    fin = np.isfinite(bt.abs_err)
    return {
        "mase": fnum(np.nanmean(bt.scaled), 4) if np.isfinite(bt.scaled).any() else None,
        "smape": fnum(np.nanmean(bt.smape), 4) if fin.any() else None,
        "mae": fnum(np.nanmean(bt.abs_err), 3) if fin.any() else None,
        "rmse": fnum(np.sqrt(np.nanmean(bt.abs_err**2)), 3) if fin.any() else None,
    }


def forecast_series_seasonal(
    s: Series, cfg: CoreConfig, as_of_key: str, data_version: str, horizon: int
) -> tuple[dict[str, Any], ForecastState] | None:
    """The seasonal path of ``forecast_series`` (module docstring)."""
    sd = seasonal_data(s, cfg)
    if sd is None or len(sd.raw) < cfg.forecast_min_history + 2:
        return None
    n = len(sd.raw)
    first = max(cfg.forecast_min_history, n - cfg.forecast_backtest_origins)
    origins = np.arange(first, n + 1)  # the last one (n) is the served forecast
    bts_all, paths = backtest_all(sd, cfg, horizon, origins)
    if "naive" not in bts_all:
        return None
    rows = np.arange(len(origins) - 1)
    bts = {k: _bt_rows(b, rows) for k, b in bts_all.items()}
    met = {k: _m(b) for k, b in bts.items()}
    order = [m for m in sd.models if m in bts]

    def key(name: str) -> tuple[float, int]:
        v = met[name]["mase"]
        return (v if v is not None else float("inf"), order.index(name))

    best = min(order, key=key)
    bt = bts[best]
    P, T, par = paths[best]
    path, trend = P[-1], T[-1]
    q = step_quantiles(bt.resid, cfg, horizon)
    q_lo, q_hi = q if q is not None else (np.full(horizon, np.nan), np.full(horizon, np.nan))
    bench = sd.benchmark if sd.benchmark in met else "naive"
    backtest = {
        "origins": len(bt.origins),
        **met[best],
        "coverage80": fnum(honest_coverage(bt, cfg, horizon), 4),
        "naive_mase": met[bench]["mase"],
        "benchmark": bench,
        "random_walk_mase": met["naive"]["mase"],
        "seasonal_naive_mase": met["seasonal_naive"]["mase"] if "seasonal_naive" in met else None,
        "mase_scale": f"in-sample naive MAE at lag {sd.lag}" + (" (seasonal)" if sd.lag > 1 else ""),
    }
    state = ForecastState(
        series_id=s.id,
        model=best,
        transform=sd.transform,
        history_months=sd.months,
        history=sd.raw,
        path=path,
        trend=trend,
        q_lo=q_lo,
        q_hi=q_hi,
        params=par,
        backtest=backtest,
        bt=bt,
        context=sd.ctx,
    )
    unit = FREQ_NAMES[s.freq]
    hist_n = min(n, cfg.forecast_history_months)
    fc_params = {
        "horizon": horizon,
        "origins": cfg.forecast_backtest_origins,
        "history": cfg.forecast_history_months,
        "models": list(sd.models),
        "seasonal_period": s.seasonal_period,
        "calendar": s.calendar,
        "hw_grid": [HW_ALPHAS, HW_TRENDS, HW_GAMMAS],
        "interval": cfg.forecast_interval,
    }
    out = {
        "id": stable_id("fc", "forecast", s.id, as_of_key),
        "series_id": s.id,
        "entity": s.entity,
        "entity_type": s.entity_type,
        "metric": s.metric,
        "model": best,
        "model_version": f"{FORECAST_VERSION}-{short_hash([fc_params, data_version, as_of_key])}",
        "model_params": {k: fnum(v, 5) for k, v in par.items()},
        "horizon_months": horizon,
        "issued_at": as_of_key,
        "features_used": [
            f"{'log1p(' + s.metric + ')' if sd.transform == 'log1p' else s.metric} history, {hist_n} {unit}s"
            + (f"; season classes ({s.calendar or 'cycle position'}, period {s.seasonal_period})" if s.seasonal_period else "")
        ],
        "points": scenario_points(state, path, horizon),
        "backtest": backtest,
        "backtest_all_models": met,
        "horizon_unit": unit,
        "entity_id": s.entity_id or f"{s.entity_type}:{s.entity}",
        "adverse_direction": s.adverse_direction,
        "candidate_models": order,
        "seasonal_period": s.seasonal_period,
        "calendar": s.calendar,
    }
    return out, state


# --------------------------------------------------------------------------------------------------
# evaluation of the served procedure (seasonal path and v1.2 models alike)


def rolling_evaluation(
    s: Series,
    cfg: CoreConfig,
    issue_idx: np.ndarray | None = None,
    start: str | None = None,
    end: str | None = None,
    step: int = 1,
    models: tuple[str, ...] | None = None,
    horizon: int | None = None,
    scale_lag: int | None = None,
    benchmark_period: int | None = None,
) -> dict[str, Any] | None:
    """Replay the served forecast at many issue times and score it on later data.

    At each issue time o (every ``step``-th complete period in [start, end]): the candidates'
    backtests on the ``forecast_backtest_origins`` origins before o, restricted to targets < o,
    select the model (lowest MASE, ties -> earlier candidate); its residuals give the conformal
    80 % interval; the model's forecast from o is scored against the truth of o .. o + h - 1.
    Works for any series: ``models`` overrides the candidates (default: the series' own). Returns
    per-issue-time aggregates: MAE, RMSE, MASE, the random-walk and seasonal-naive benchmarks' MASE
    on the same points, coverage80 and the share of issue times each model was selected.
    ``scale_lag`` fixes the MASE scale lag and ``benchmark_period`` the seasonal-naive benchmark's
    period, so that differently configured series are scored on identical terms; the relative MAEs
    (model MAE / benchmark MAE on the same points) are scale-free."""
    horizon = horizon or cfg.forecast_horizon
    if models is not None or not s.seasonal:
        s = _with_models(s, models or MODELS)
    sd = seasonal_data(s, cfg)
    if sd is None:
        return None
    if scale_lag is not None:
        sd.lag = int(scale_lag)
    n = len(sd.raw)
    if issue_idx is None:
        per = sd.months
        lo = 0 if start is None else int(np.searchsorted(per.start_time, pd.Timestamp(start)))
        hi = n - 1 if end is None else int(np.searchsorted(per.start_time, pd.Timestamp(end), side="right")) - 1
        issue_idx = np.arange(max(lo, 1), hi + 1, step)
    N = cfg.forecast_backtest_origins
    issue_idx = issue_idx[issue_idx >= cfg.forecast_min_history + N]
    issue_idx = issue_idx[issue_idx + horizon <= n]
    if not len(issue_idx):
        return None
    origins = np.arange(int(issue_idx.min()) - N, int(issue_idx.max()) + 1)
    bts, _ = backtest_all(sd, cfg, horizon, origins)
    if "naive" not in bts:
        return None
    names = [m for m in sd.models if m in bts]
    extra_bench = {}
    bp = benchmark_period or sd.ctx.period
    if bp and ("seasonal_naive" not in bts or bp != sd.ctx.period):
        bctx = SeasonContext(sd.ctx.freq, bp, None, sd.ctx.periods)
        extra_sd = SeasonalData(**{**sd.__dict__, "models": ("seasonal_naive",), "ctx": bctx})
        extra_bench = backtest_all(extra_sd, cfg, horizon, origins)[0]
    bench_bt = {**{k: bts[k] for k in ("naive", "seasonal_naive") if k in bts}, **extra_bench}
    sel_count = dict.fromkeys(names, 0)
    rows_scaled: dict[str, list[np.ndarray]] = {"model": [], **{k: [] for k in bench_bt}}
    ae: list[np.ndarray] = []
    hits = total = 0
    for o in issue_idx:
        r = int(o - origins[0])
        win = slice(r - N, r)
        scores = []
        for name in names:
            b = bts[name]
            sc = np.where(b.target[win] < o, b.scaled[win], np.nan)
            scores.append(float(np.nanmean(sc)) if np.isfinite(sc).any() else np.inf)
        best = names[int(np.argmin(scores))]
        sel_count[best] += 1
        b = bts[best]
        past = np.where(b.target[win] < o, b.resid[win], np.nan)
        q = step_quantiles(past, cfg, horizon)
        rows_scaled["model"].append(b.scaled[r])
        ae.append(b.abs_err[r])
        for k, bb in bench_bt.items():
            rows_scaled[k].append(bb.scaled[r])
        if q is not None:
            lo_q, hi_q = q
            res = b.resid[r]
            ok = np.isfinite(res) & np.isfinite(lo_q)
            total += int(ok.sum())
            hits += int(((lo_q - 1e-12 <= res) & (res <= hi_q + 1e-12) & ok).sum())
    A = np.array(ae)
    mase = {k: float(np.nanmean(np.array(v))) for k, v in rows_scaled.items()}
    rows = issue_idx - origins[0]

    def rel(k: str) -> float | None:
        if k not in bench_bt:
            return None
        ref = bench_bt[k].abs_err[rows]
        ok = np.isfinite(ref) & np.isfinite(A)
        return fnum(float(A[ok].mean() / ref[ok].mean()), 4) if ok.any() and ref[ok].mean() > 0 else None
    return {
        "series_id": s.id,
        "issue_times": len(issue_idx),
        "first_issue": str(sd.months[int(issue_idx[0])]),
        "last_issue": str(sd.months[int(issue_idx[-1])]),
        "horizon": horizon,
        "candidates": names,
        "selected": {k: v for k, v in sel_count.items() if v},
        "mae": fnum(np.nanmean(A), 4),
        "rmse": fnum(np.sqrt(np.nanmean(A**2)), 4),
        "mase": fnum(mase["model"], 4),
        "random_walk_mase": fnum(mase.get("naive"), 4),
        "seasonal_naive_mase": fnum(mase.get("seasonal_naive"), 4),
        "mase_scale_lag": sd.lag,
        "relative_mae_vs_naive": rel("naive"),
        "relative_mae_vs_seasonal_naive": rel("seasonal_naive"),
        "coverage80": fnum(hits / total, 4) if total else None,
        "interval_points": total,
    }


def _with_models(s: Series, models: tuple[str, ...]) -> Series:
    from dataclasses import replace

    return replace(s, forecast_models=tuple(models))
