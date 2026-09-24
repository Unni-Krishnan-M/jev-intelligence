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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import fnum, period_str, short_hash, stable_id
from jev_ml.core.config import FORECAST_VERSION, CoreConfig
from jev_ml.core.series import Series

MODELS = ("naive", "moving_average", "holt_damped")
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
