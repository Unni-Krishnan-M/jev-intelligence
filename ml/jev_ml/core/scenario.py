"""What-if scenarios on top of a series' selected forecast model.

The baseline is the selected model's forecast path f_h (transformed scale: log1p for counts) with
its trend component T_h (damped Holt: T_h = sum_{i<=h} phi^i * b; naive/moving average: T_h = 0).

* continue: g_h = f_h (identical to the baseline).
* slow / reverse / any ``trend_multiplier`` m: g_h = f_h + (m - 1) * T_h. m = 0.5 halves the
  trend, m = -1 mirrors it (the slope changes sign).
* shock: from ``shock_month`` (1-based) onwards the level is multiplied by (1 + level_shift_pct/100)
  (i.e. + log(1 + pct/100) on log scale; for non-count series the value is scaled directly).

Intervals: the same empirical backtest residual quantiles as the baseline are added around every
scenario path (scenarios shift the trend, not the noise). Beyond the backtest horizon the last
step's quantiles are widened by sqrt(h / horizon). If the selected model has no trend component, trend
multipliers would have no effect. ``trend_source`` (added field) controls this:

* ``"model"``: only the selected model's own trend (may be zero -> assumptions say so).
* ``"auto"`` (default): the model's trend if it has one; otherwise the Theil–Sen slope of the last
  `trend_window_months` complete months on the model's scale is added to the model's level
  (T_h = h * slope), so the *scenario baseline* = model level + observed trend. The output labels
  this (``trend.source = "theil_sen"`` plus the slope's 95 % CI) and the baseline then differs
  from the plain forecast in ``predictions.forecasts``, which stays untouched.

Domain-independent: ``run_scenario`` works on any ``PipelineResult`` (its series, fitted forecast
states and as_of are reused); ``scenario_on_series`` works on a plain series list (adapters use it
when no result is at hand).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jev_ml.core.common import fnum, period_str
from jev_ml.core.config import CoreConfig
from jev_ml.core.forecast import ForecastState, _itf, forecast_series, interval_at, rerun
from jev_ml.core.series import FREQ_NAMES, Series
from jev_ml.core.trends import theil_sen

KINDS = ("continue", "slow", "reverse", "shock", "custom")
DEFAULT_MULT = {"continue": 1.0, "slow": 0.5, "reverse": -1.0, "shock": 1.0, "custom": 1.0}
MAX_SCENARIOS = 6


def _validate(spec: dict[str, Any], known: list[str]) -> tuple[str, int, list[dict[str, Any]]]:
    if not isinstance(spec, dict):
        raise ValueError("scenario spec must be an object")
    sid = spec.get("series_id")
    if sid not in known:
        raise ValueError(f"unknown series_id {sid!r}")
    h = spec.get("horizon_months", 12)
    if not isinstance(h, int) or isinstance(h, bool) or not 1 <= h <= 24:
        raise ValueError("horizon_months must be an integer in 1..24")
    scen = spec.get("scenarios") or [{"name": "Trend continues", "kind": "continue"}]
    if not isinstance(scen, list) or not 1 <= len(scen) <= MAX_SCENARIOS:
        raise ValueError(f"scenarios must be a list of 1..{MAX_SCENARIOS} items")
    names = set()
    for s in scen:
        if not isinstance(s, dict) or s.get("kind") not in KINDS:
            raise ValueError(f"each scenario needs kind in {list(KINDS)}")
        name = str(s.get("name") or s["kind"])
        if name in names:
            raise ValueError(f"duplicate scenario name {name!r}")
        names.add(name)
        m = s.get("trend_multiplier", DEFAULT_MULT[s["kind"]])
        if not isinstance(m, int | float) or not -5 <= float(m) <= 5:
            raise ValueError("trend_multiplier must be a number in -5..5")
        pct = s.get("level_shift_pct", 0.0 if s["kind"] != "shock" else None)
        if s["kind"] == "shock" and pct is None:
            raise ValueError("shock scenarios need level_shift_pct")
        if pct is not None and (not isinstance(pct, int | float) or not -95 <= float(pct) <= 500):
            raise ValueError("level_shift_pct must be a number in -95..500")
        sm = s.get("shock_month", 1)
        if spec.get("trend_source", "auto") not in ("auto", "model"):
            raise ValueError("trend_source must be 'auto' or 'model'")
        if not isinstance(sm, int) or not 1 <= sm <= h:
            raise ValueError("shock_month must be an integer in 1..horizon_months")
    return str(sid), int(h), scen


def _state_for(
    series: list[Series],
    states: dict[str, ForecastState],
    sid: str,
    as_of_key: str,
    data_version: str,
    cfg: CoreConfig,
) -> ForecastState:
    if sid in states:
        return states[sid]
    s = next(x for x in series if x.id == sid)
    res = forecast_series(s, cfg, as_of_key, data_version, horizon=max(cfg.forecast_horizon, 1))
    if res is None:
        raise ValueError(f"series {sid!r} has too little history to forecast")
    return res[1]


def _points(state: ForecastState, path: np.ndarray, horizon: int) -> list[dict[str, Any]]:
    months = state.months_ahead(horizon)
    out = []
    for h in range(horizon):
        lo, hi = interval_at(state, h)
        mean = float(_itf(state.transform, np.array([path[h]]))[0])
        if np.isfinite(lo):
            lo_v = float(_itf(state.transform, np.array([path[h] + lo]))[0])
            hi_v = float(_itf(state.transform, np.array([path[h] + hi]))[0])
        else:
            lo_v = hi_v = float("nan")
        out.append(
            {"t": period_str(months[h]), "mean": fnum(mean, 4), "lo80": fnum(lo_v, 4), "hi80": fnum(hi_v, 4)}
        )
    return out


def run_scenario(source: Any, spec: dict[str, Any], config: CoreConfig | None = None) -> dict[str, Any]:
    """Run a what-if analysis on a finished ``PipelineResult`` (see docs/intelligence.md, scenario
    contract). Its series, fitted forecast states and as_of are reused. Raises ValueError for
    invalid specs."""
    cfg = config or source.config
    prep = source.prepared
    data_version = str(source.data["run"].get("data_version") or getattr(prep, "data_version", ""))
    return scenario_on_series(
        source.series_objects,
        source.forecast_states,
        source.data["run"]["as_of"],
        data_version,
        spec,
        cfg,
    )


def scenario_on_series(
    series: list[Series],
    states: dict[str, ForecastState],
    as_of: str,
    data_version: str,
    spec: dict[str, Any],
    cfg: CoreConfig,
) -> dict[str, Any]:
    """The scenario engine on a plain series list (``states`` may hold fitted forecasts to reuse)."""
    known = [s.id for s in series]
    sid, horizon, scen = _validate(spec, known)
    state = _state_for(series, states, sid, as_of, data_version, cfg)
    y = np.log1p(np.maximum(state.history, 0)) if state.transform == "log1p" else state.history
    path, trend = rerun(state, horizon, cfg)
    trend_info: dict[str, Any] = {"source": "model", "slope_per_month": None, "slope_ci": None}
    if np.any(np.abs(trend) > 1e-12):
        trend_info["slope_per_month"] = fnum(float(trend[0]), 6)
    elif spec.get("trend_source", "auto") == "auto":
        w = y[-cfg.trend_window_months :]
        w = w[np.isfinite(w)]
        if len(w) >= cfg.trend_min_points and not np.allclose(w, w[0]):
            slope, _, lo, hi = theil_sen(w)
            trend = slope * np.arange(1, horizon + 1)
            path = path + trend
            trend_info = {
                "source": "theil_sen",
                "slope_per_month": fnum(slope, 6),
                "slope_ci": [fnum(lo, 6), fnum(hi, 6)],
            }
    base_pts = _points(state, path, horizon)
    agg = "sum" if state.transform == "log1p" else "mean"

    def total(pts: list[dict[str, Any]]) -> float:
        vals = [p["mean"] or 0.0 for p in pts]
        return float(np.sum(vals) if agg == "sum" else np.mean(vals))

    base_total = total(base_pts)
    has_trend = bool(np.any(np.abs(trend) > 1e-12))
    # period wording: months for the v1.2 path (unchanged text), the series' own unit otherwise
    unit = "month" if state.context is None else FREQ_NAMES[state.context.freq]
    last_label = period_str(state.history_months[-1])
    last_label = last_label[:7] if unit == "month" else last_label
    out_s = []
    for s in scen:
        kind = s["kind"]
        m = float(s.get("trend_multiplier", DEFAULT_MULT[kind]))
        g = path + (m - 1.0) * trend
        assumptions = [
            f"model: {state.model} fitted on {len(state.history)} complete {unit}s up to {last_label}"
        ]
        if trend_info["source"] == "theil_sen":
            assumptions.append(
                f"{state.model} has no trend: baseline = its level + Theil–Sen trend of the last "
                f"{cfg.trend_window_months} {unit}s ({trend_info['slope_per_month']}/{unit} on "
                f"{'log1p' if state.transform == 'log1p' else 'raw'} scale, 95 % CI {trend_info['slope_ci']})"
            )
        if kind != "continue" and m != 1.0:
            assumptions.append(
                f"trend x {m:g}"
                if has_trend
                else f"trend x {m:g} has no effect: the selected {state.model} model carries no trend"
            )
        pct = s.get("level_shift_pct")
        if pct:
            sm = int(s.get("shock_month", 1))
            if state.transform == "log1p":
                g = g.copy()
                g[sm - 1 :] = g[sm - 1 :] + np.log1p(float(pct) / 100.0)
            else:
                g = g.copy()
                g[sm - 1 :] = g[sm - 1 :] * (1 + float(pct) / 100.0)
            assumptions.append(f"level shift {float(pct):+g} % from {unit} {sm} onwards")
        if kind == "continue" and m == 1.0 and not pct:
            assumptions.append("identical to the baseline forecast")
        if len(state.q_lo) < horizon:
            assumptions.append(
                f"intervals beyond {unit} {len(state.q_lo)} widened by sqrt(h/{len(state.q_lo)})"
            )
        pts = _points(state, g, horizon)
        tot = total(pts)
        out_s.append(
            {
                "name": str(s.get("name") or kind),
                "kind": kind,
                "assumptions": assumptions,
                "points": pts,
                "total": fnum(tot, 3),
                "delta_vs_baseline": fnum(tot - base_total, 3),
                "delta_pct": fnum(100 * (tot - base_total) / base_total, 3) if base_total else None,
                "end_level": pts[-1]["mean"],
            }
        )
    ranked = sorted(out_s, key=lambda x: (-(x["total"] or 0.0), x["name"]))
    comparison = [
        {
            "name": x["name"],
            "total": x["total"],
            "delta_pct": x["delta_pct"],
            "end_level": x["end_level"],
            "rank": i + 1,
        }
        for i, x in enumerate(ranked)
    ]
    hist_n = min(len(state.history), 36)
    return {
        "series_id": sid,
        "as_of": as_of,
        "model": state.model,
        "aggregate": agg,
        "trend": trend_info,
        "horizon_months": horizon,
        "baseline": {"points": base_pts, "total": fnum(base_total, 3)},
        "history": [
            {"t": period_str(state.history_months[i]), "v": fnum(state.history[i], 4)}
            for i in range(len(state.history) - hist_n, len(state.history))
        ],
        "scenarios": out_s,
        "comparison": comparison,
        "uncertainty_note": (
            "80 % intervals from rolling-origin residuals; scenarios shift the trend, not the noise."
        ),
    }
