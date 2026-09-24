"""Evaluation of the engine itself, per domain (platform contract section 7).

* **Forecasts** (``evaluate_forecasts``): rolling-origin backtest of every series whose spec sets
  ``forecast``. To avoid selection bias the model is *selected* on the first half of the origins
  and *scored* on the second half; errors on the raw scale: MAE, RMSE, MASE (vs the in-sample
  one-step naive MAE), the naive model's MASE on the same points, and the honest 80 % interval
  coverage (each origin's interval uses only residuals realised before it).
* **Warning precision / false-positive rate** (``warning_replay``): the pipeline is re-run as of the
  first day of consecutive months (leak-free: each replay sees only data available then). Each
  replay yields *units* (for example every monitored series) with ``warned`` = a warning was raised
  for it. A domain-specific ``outcome`` function says whether the unit's adverse condition
  materialised in the following ``h`` periods (``confirmed``), or ``None`` when that is not
  observable (the horizon runs past the data). From the observable units: precision = TP / (TP + FP),
  false-positive rate = FP / (FP + TN), recall = TP / (TP + FN), base rate = (TP + FN) / n.
  Outcomes are scored against the data as it is *now* (the latest vintage), which is also what the
  replays see: revisions after first publication are not modelled.
* **Series outcome** (``series_adverse_move``, used by the generic adapter and the movie genre-share
  risk): confirmed when, for some k in 1..h, the series moves in its adverse direction by more than
  2 sigma sqrt(k) from its value in the replay's last complete period, sigma = 1.4826 x MAD of the 24
  period-over-period changes up to that period (floored at the data resolution): an adverse move
  beyond random-walk noise.
* **Decision consistency** (``decision_consistency``): the flip rate of ``early_warning_level``
  between consecutive monthly replays, per situation (a situation without a decision in a replay
  counts as NO_ACTION), and the rate of flips across the WARNING boundary (warned <-> not warned).
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import fnum
from jev_ml.core.config import EWL_LEVELS, CoreConfig
from jev_ml.core.forecast import MODELS, backtest_model, step_quantiles
from jev_ml.core.series import Series

EWL = "early_warning_level"


# ------------------------------------------------------------------------------------------ forecasts


def _subset(bt: Any, rows: np.ndarray) -> dict[str, float | None]:
    sc, ae = bt.scaled[rows], bt.abs_err[rows]
    return {
        "mase": float(np.nanmean(sc)) if np.isfinite(sc).any() else None,
        "mae": float(np.nanmean(ae)) if np.isfinite(ae).any() else None,
        "rmse": float(np.sqrt(np.nanmean(ae**2))) if np.isfinite(ae).any() else None,
    }


def _coverage(bt: Any, cfg: CoreConfig, horizon: int, rows: np.ndarray) -> float | None:
    hits = total = 0
    for j in rows:
        o = bt.origins[j]
        q = step_quantiles(np.where(bt.target < o, bt.resid, np.nan), cfg, horizon)
        if q is None:
            continue
        lo, hi = q
        for h in range(horizon):
            r = bt.resid[j, h]
            if np.isfinite(r) and np.isfinite(lo[h]):
                total += 1
                hits += int(lo[h] - 1e-12 <= r <= hi[h] + 1e-12)
    return hits / total if total else None


def evaluate_forecasts(series: list[Series], cfg: CoreConfig) -> dict[str, Any]:
    horizon = cfg.forecast_horizon
    per: list[dict[str, Any]] = []
    for s in series:
        if not s.forecast:
            continue
        _, raw, _ = s.complete()
        if s.is_count:
            raw, tr = np.where(np.isfinite(raw), raw, 0.0), "log1p"
        else:
            raw, tr = raw[np.isfinite(raw)], "identity"
        if len(raw) < cfg.forecast_min_history + 2:
            continue
        bts = {m: backtest_model(m, raw, tr, cfg, horizon, cfg.forecast_backtest_origins) for m in MODELS}
        if any(b is None for b in bts.values()):
            continue
        n_o = len(bts["naive"].origins)  # type: ignore[union-attr]
        sel_rows, score_rows = np.arange(n_o // 2), np.arange(n_o // 2, n_o)
        sel = {m: _subset(b, sel_rows)["mase"] for m, b in bts.items()}
        best = min(MODELS, key=lambda m: (sel[m] if sel[m] is not None else np.inf, MODELS.index(m)))
        met = _subset(bts[best], score_rows)
        naive = _subset(bts["naive"], score_rows)
        per.append(
            {
                "series_id": s.id,
                "model": best,
                "mae": fnum(met["mae"], 4),
                "rmse": fnum(met["rmse"], 4),
                "mase": fnum(met["mase"], 4),
                "naive_mae": fnum(naive["mae"], 4),
                "naive_mase": fnum(naive["mase"], 4),
                "coverage80": fnum(_coverage(bts[best], cfg, horizon, score_rows), 4),
                "origins": len(score_rows),
                "selection_origins": len(sel_rows),
            }
        )

    def med(k: str) -> float | None:
        v = [p[k] for p in per if p[k] is not None]
        return fnum(np.median(v), 4) if v else None

    both = [p for p in per if p["mase"] is not None and p["naive_mase"] is not None]
    covs = [p["coverage80"] for p in per if p["coverage80"] is not None]
    return {
        "protocol": (
            f"rolling origin, horizon {horizon}; model selected on the first half of "
            f"{cfg.forecast_backtest_origins} origins, scored on the second half"
        ),
        "per_series": per,
        "summary": {
            "n_series": len(per),
            "median_mae": med("mae"),
            "median_rmse": med("rmse"),
            "median_mase": med("mase"),
            "median_naive_mase": med("naive_mase"),
            "share_beating_naive": fnum(np.mean([p["mase"] < p["naive_mase"] for p in both]), 4)
            if both
            else None,
            "mean_coverage80": fnum(np.mean(covs), 4) if covs else None,
        },
    }


# --------------------------------------------------------------------------------------- replays


def monthly_as_ofs(start: str, end: str) -> list[str]:
    return [p.strftime("%Y-%m-01") for p in pd.period_range(start[:7], end[:7], freq="M")]


def run_replays(
    run: Callable[[str], Any], as_ofs: list[str], keep: Callable[[Any], Any] | None = None
) -> list[dict[str, Any]]:
    """``run(as_of) -> PipelineResult``; returns per replay {as_of, result (dict), extra, ms}."""
    out = []
    for a in as_ofs:
        t0 = time.perf_counter()
        res = run(a)
        ms = 1000 * (time.perf_counter() - t0)
        out.append({"as_of": a, "result": res.to_dict(), "extra": keep(res) if keep else None, "ms": ms})
    return out


def series_adverse_move(
    values: np.ndarray, last: int, h: int, adverse: str, resolution: float | None = None
) -> tuple[bool | None, str]:
    """Outcome of an adverse-move unit (module docstring). ``values`` = the full current series,
    ``last`` = index of the replay's last complete period."""
    if last + h >= len(values) or last < 1:
        return None, f"the {h} periods after the replay are not in the data yet"
    x0 = values[last]
    if not np.isfinite(x0):
        return None, "no value in the replay's last period"
    d = np.diff(values[max(0, last - 24) : last + 1])
    d = d[np.isfinite(d)]
    if len(d) < 6:
        return None, "fewer than 6 past changes to estimate the noise"
    sigma = max(1.4826 * float(np.median(np.abs(d - np.median(d)))), resolution or 0.0, 1e-12)
    sign = {"up": 1.0, "down": -1.0}
    for k in range(1, h + 1):
        x = values[last + k]
        if not np.isfinite(x):
            continue
        moves = [sign[adverse] * (x - x0)] if adverse in sign else [abs(x - x0)]
        if max(moves) > 2 * sigma * np.sqrt(k):
            return True, f"moved {x - x0:+.4g} by period +{k} (> 2 sigma sqrt(k), sigma {sigma:.4g})"
    return False, f"no adverse move beyond 2 sigma sqrt(k) within {h} periods (sigma {sigma:.4g})"


def confusion(units: list[dict[str, Any]]) -> dict[str, Any]:
    obs = [u for u in units if u["confirmed"] is not None]
    tp = sum(u["warned"] and u["confirmed"] for u in obs)
    fp = sum(u["warned"] and not u["confirmed"] for u in obs)
    fn = sum((not u["warned"]) and u["confirmed"] for u in obs)
    tn = sum((not u["warned"]) and not u["confirmed"] for u in obs)
    return {
        "units": len(units),
        "observable": len(obs),
        "warned": sum(u["warned"] for u in obs),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": fnum(tp / (tp + fp), 4) if tp + fp else None,
        "false_positive_rate": fnum(fp / (fp + tn), 4) if fp + tn else None,
        "recall": fnum(tp / (tp + fn), 4) if tp + fn else None,
        "base_rate": fnum((tp + fn) / len(obs), 4) if obs else None,
    }


def warning_replay(
    replays: list[dict[str, Any]],
    units: Callable[[dict[str, Any]], list[dict[str, Any]]],
    outcome: Callable[[dict[str, Any], dict[str, Any]], tuple[bool | None, str]],
    excluded: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score warnings per kind. ``units(replay)`` -> [{kind, unit, warned, ...}];
    ``outcome(replay, unit)`` -> (confirmed | None, detail)."""
    rows = []
    for r in replays:
        for u in units(r):
            conf, detail = outcome(r, u)
            rows.append({**u, "as_of": r["as_of"], "confirmed": conf, "detail": detail})
    kinds = sorted({u["kind"] for u in rows})
    by_kind = {k: confusion([u for u in rows if u["kind"] == k]) for k in kinds}
    warned = [u for u in rows if u["warned"]]
    return {
        "per_kind": by_kind,
        "overall": confusion(rows),
        "excluded_kinds": excluded or {},
        "examples": [
            {k: u[k] for k in ("as_of", "kind", "unit", "confirmed", "detail")}
            for u in (warned[:6] + [u for u in rows if not u["warned"] and u["confirmed"]][:3])
        ],
    }


def decision_consistency(replays: list[dict[str, Any]]) -> dict[str, Any]:
    levels = []
    for r in replays:
        levels.append(
            {
                d.get("situation") or d["entity"]: d["answer"] or "ABSTAINED"
                for d in r["result"]["decisions"]
                if d["key"] == EWL
            }
        )
    flips = boundary = pairs = up = down = 0
    order = {lv: i for i, lv in enumerate(EWL_LEVELS)}
    for a, b in itertools.pairwise(levels):
        for k in set(a) | set(b):
            la, lb = a.get(k, "NO_ACTION"), b.get(k, "NO_ACTION")
            pairs += 1
            if la != lb:
                flips += 1
                if la in order and lb in order:
                    up += order[lb] > order[la]
                    down += order[lb] < order[la]
            wa, wb = la in ("WARNING", "URGENT_ACTION"), lb in ("WARNING", "URGENT_ACTION")
            boundary += wa != wb
    return {
        "replays": len(replays),
        "situation_pairs": pairs,
        "flips": flips,
        "flip_rate": fnum(flips / pairs, 4) if pairs else None,
        "escalations": up,
        "de_escalations": down,
        "warning_boundary_flips": boundary,
        "warning_boundary_flip_rate": fnum(boundary / pairs, 4) if pairs else None,
    }


def generic_units(replay: dict[str, Any]) -> list[dict[str, Any]]:
    """Every monitored series of a replay; warned = a warning was raised for its situation."""
    d = replay["result"]
    warned = {w["key"] for w in d["warnings"]}
    return [
        {
            "kind": "series_warning",
            "unit": s["id"],
            "warned": f"warning:series:{s['id']}" in warned,
            "adverse_direction": s["adverse_direction"],
            "last_complete": d["run"]["last_complete_month"],
        }
        for s in d["series"]
        if s["adverse_direction"] != "none"
    ]


def series_outcome_fn(
    full: dict[str, Series], h: int
) -> Callable[[dict[str, Any], dict[str, Any]], tuple[bool | None, str]]:
    def fn(replay: dict[str, Any], u: dict[str, Any]) -> tuple[bool | None, str]:
        s = full.get(u["unit"])
        if s is None or not u.get("last_complete"):
            return None, "series or last complete period missing"
        months, values, _ = s.complete()
        p = (
            pd.Period(u["last_complete"][:7], freq="M")
            if s.freq == "M"
            else pd.Period(u["last_complete"], freq="W")
        )
        idx = np.flatnonzero(months == p)
        if not len(idx):
            return None, "replay period not in the current series"
        return series_adverse_move(values, int(idx[0]), h, u["adverse_direction"], s.resolution)

    return fn


__all__ = [
    "confusion",
    "decision_consistency",
    "evaluate_forecasts",
    "generic_units",
    "monthly_as_ofs",
    "run_replays",
    "series_adverse_move",
    "series_outcome_fn",
    "warning_replay",
]
