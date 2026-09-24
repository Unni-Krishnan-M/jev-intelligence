"""Evaluation helpers for generic domains with a seasonal (daily) grid (docs/SECOND_DOMAIN_CASE_STUDY.md).

* ``plain_config(raw)``: the *generic* baseline of a domain-aware config. Same data, frequency,
  entities, thresholds and adverse direction; every seasonality-aware option is removed:
  ``seasonal_period``, ``calendar``, ``forecast_models`` (-> the v1.2 trio naive / moving average /
  damped Holt), ``anomaly_basis: seasonal`` (-> ``level``: trailing baseline), derived series
  (``rolling`` / ``ratio_lag``) are dropped and the trend runs on the raw series.
* ``seasonal_adverse_move``: the confirmation rule used for BOTH configs (so the comparison is on
  identical terms). With Y(d) = log(7-day total ending d) - log(7-day total ending d - 364) (weekday
  aligned year-over-year change of a whole week: weekly and annual seasonality cancel) and d0 the
  replay's last complete day, sigma = 1.4826 x MAD of the 52 week-over-week changes
  Y(d0 - 7j) - Y(d0 - 7(j + 1)), a unit is *confirmed* when for some k in {7, 14, 21, 28} days both
  Y(d0 + k) and Y(d0 + k + 7) lie more than 2 sigma sqrt(k / 7) below Y(d0): a sustained (two
  consecutive weeks) adverse move beyond week-to-week noise within four weeks. Not observable when
  the data does not reach d0 + 35 or a needed week has a gap.
* ``entity_units``: one unit per (replay, entity) with a daily series; warned = any warning on one
  of the entity's series (the daily series or its ratio series).
* ``warned_flip_rate``: share of consecutive-replay pairs in which a unit's warned status changes
  (comparable across configs, unlike situation-level flip rates when the series sets differ).
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import fnum
from jev_ml.core.series import Series

SEASONAL_KEYS = ("seasonal_period", "calendar", "anomaly_calendar", "forecast_models", "rolling", "ratio_lag")


def plain_config(raw: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(raw)
    out["key"] = f"{raw['key']}-plain"
    out["name"] = f"{raw['name']} (plain generic config)"
    specs = []
    for sp in out["series"]:
        if sp.get("rolling") or sp.get("ratio_lag"):
            continue
        sp = {k: v for k, v in sp.items() if k not in SEASONAL_KEYS}
        if sp.get("anomaly_basis") == "seasonal":
            sp["anomaly_basis"] = "level"
        sp["trend"] = True
        specs.append(sp)
    out["series"] = specs
    out.pop("calendar_check", None)
    return out


def weekly_yoy(values: np.ndarray) -> np.ndarray:
    """Y(d) of the module docstring for every day (NaN where a 7-day window has a gap)."""
    v = pd.Series(np.asarray(values, dtype=float))
    w = v.rolling(7, min_periods=7).sum().to_numpy()
    lw = np.log(np.where(w > 0, w, np.nan))
    y = np.full(len(lw), np.nan)
    y[364:] = lw[364:] - lw[:-364]
    return y


def seasonal_adverse_move(yoy: np.ndarray, last: int, horizon_weeks: int = 4) -> tuple[bool | None, str]:
    need = last + 7 * (horizon_weeks + 1)
    if need >= len(yoy) or last < 364 + 7 * 53:
        return None, "the weeks after the replay (or the year before it) are not in the data"
    y0 = yoy[last]
    past = yoy[last - 7 * np.arange(53)]
    d = -np.diff(past)  # Y(d0 - 7j) - Y(d0 - 7(j+1))
    d = d[np.isfinite(d)]
    if not np.isfinite(y0) or len(d) < 26:
        return None, "gap in the weeks around the replay"
    sigma = max(1.4826 * float(np.median(np.abs(d - np.median(d)))), 1e-6)
    for w in range(1, horizon_weeks + 1):
        a, b = yoy[last + 7 * w], yoy[last + 7 * (w + 1)]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        band = 2 * sigma * np.sqrt(w)
        if a - y0 < -band and b - y0 < -band:
            return True, f"yoy {100 * (a - y0):+.1f} pp by week +{w}, sustained (band {100 * band:.1f} pp)"
    return False, f"no sustained adverse move beyond 2 sigma sqrt(k) within {horizon_weeks} weeks"


def entity_units(replay: dict[str, Any], prefix: str = "boardings:") -> list[dict[str, Any]]:
    d = replay["result"]
    warned_entities = {
        w["key"].split(":", 2)[2].split(":", 1)[1]
        for w in d["warnings"]
        if w["key"].startswith("warning:series:")
    }
    return [
        {
            "kind": "entity_warning",
            "unit": s["id"],
            "entity": s["entity"],
            "warned": s["entity"] in warned_entities,
            "last_complete": d["run"]["last_complete_month"],
        }
        for s in d["series"]
        if s["id"].startswith(prefix)
    ]


def seasonal_outcome_fn(
    full: dict[str, Series],
) -> Callable[[dict[str, Any], dict[str, Any]], tuple[bool | None, str]]:
    cache: dict[str, tuple[pd.PeriodIndex, np.ndarray]] = {}

    def fn(replay: dict[str, Any], u: dict[str, Any]) -> tuple[bool | None, str]:
        s = full.get(u["unit"])
        if s is None or not u.get("last_complete"):
            return None, "series or last complete day missing"
        if s.id not in cache:
            months, values, _ = s.complete()
            cache[s.id] = (months, weekly_yoy(values))
        months, yoy = cache[s.id]
        idx = np.flatnonzero(months == pd.Period(u["last_complete"], freq="D"))
        if not len(idx):
            return None, "replay day not in the current series"
        return seasonal_adverse_move(yoy, int(idx[0]))

    return fn


def warned_flip_rate(
    replays: list[dict[str, Any]], units: Callable[[dict[str, Any]], list[dict[str, Any]]]
) -> dict[str, Any]:
    states = [{u["unit"]: u["warned"] for u in units(r)} for r in replays]
    pairs = flips = 0
    for a, b in itertools.pairwise(states):
        for k in set(a) & set(b):
            pairs += 1
            flips += a[k] != b[k]
    return {"pairs": pairs, "flips": flips, "flip_rate": fnum(flips / pairs, 4) if pairs else None}


def anomaly_counts(replays: list[dict[str, Any]]) -> dict[str, Any]:
    """Active (unsuppressed) anomalies per replay and the share of them on a Saturday, Sunday or
    holiday service day (the day_type proxy: weekday of the anomaly date >= 5)."""
    n = weekend = 0
    for r in replays:
        for a in r["result"]["anomalies"]:
            if a["suppressed"]:
                continue
            n += 1
            weekend += pd.Timestamp(a["detected_at"]).dayofweek >= 5
    return {
        "active_anomalies": n,
        "per_replay": fnum(n / len(replays), 3) if replays else None,
        "share_on_weekends": fnum(weekend / n, 3) if n else None,
    }
