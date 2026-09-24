"""UNDERSTAND/DETECT -> Signal objects: compact, deduplicated observations for the console.

Strength (0..1) per kind (core kinds; adapters add their own through ``DomainExtras.signals``):

* trend: 1 - BH q-value of the Mann–Kendall test (non-flat trends only);
* change_point: 1 - null-distribution p-value;
* anomaly: the anomaly's ``anomaly_score`` (robust z -> min(1, |z| / (2 * threshold)); adapters
  define theirs). Suppressed anomalies produce no signal;
* forecast: min(1, |next-horizon total vs last-horizon total| / 100 %) x skill,
  skill = clip(1 - mase/naive_mase); only when |change| >= 25 % and the model beats naive in the
  backtest; at most 5 (the strongest), because related series' forecasts move together;
* quality: 1.0 for a failed error-level check, 0.5 for a warning-level check;
* live (staleness): the staleness risk's likelihood.

Additive platform fields: ``entity_id``, ``baseline``, ``change``, ``confidence`` and
``confidence_kind`` (trend/change point: 1 - q / 1 - p, kind ``evidence``; anomaly: its normalised
score, ``margin``; forecast: backtest skill, ``margin``; quality/staleness: 1.0, ``rule``).

Dedup: one signal per ``dedup_key``; the strongest wins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import clip01, evidence, fnum, iso, stable_id
from jev_ml.core.config import CoreConfig
from jev_ml.core.series import Series
from jev_ml.core.trends import window_label

FORECAST_SIGNAL_MIN_CHANGE = 0.25
FORECAST_SIGNAL_MAX = 5


def make_signal(
    kind: str,
    key: str,
    entity_type: str,
    entity: str,
    title: str,
    value: Any,
    unit: str | None,
    strength: float,
    direction: str | None,
    source: str | None,
    observed_at: str | None,
    window: str | None,
    freshness_days: float | None,
    ev: list[dict[str, Any]],
    as_of_key: str,
    baseline: Any = None,
    change: Any = None,
    confidence: float | None = None,
    confidence_kind: str | None = None,
    entity_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": stable_id("sig", kind, key, as_of_key),
        "dedup_key": key,
        "kind": kind,
        "entity_type": entity_type,
        "entity": entity,
        "title": title,
        "value": value,
        "unit": unit,
        "strength": fnum(clip01(strength), 4),
        "direction": direction,
        "source": source,
        "observed_at": observed_at,
        "window": window,
        "freshness_days": fnum(freshness_days, 2),
        "evidence": ev,
        "entity_id": entity_id or f"{entity_type}:{entity}",
        "baseline": baseline,
        "change": change,
        "confidence": fnum(confidence, 4),
        "confidence_kind": confidence_kind,
    }


def period_age_days(as_of_ts: float, period_start: str) -> float:
    """Days from the end of the period starting at ``period_start`` to as_of (>= 0)."""
    p = (
        pd.Period(period_start[:7], freq="M")
        if period_start.endswith("-01")
        else pd.Period(period_start, freq="W")
    )
    end = p.end_time.tz_localize("UTC").timestamp()
    return max(0.0, (as_of_ts - end) / 86400.0)


def _sub(a: Any, b: Any) -> float | None:
    return fnum(a - b) if a is not None and b is not None else None


def build_signals(
    trends: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    series_last: dict[str, float],
    series_by_id: dict[str, Series],
    quality: dict[str, Any],
    staleness_risks: list[dict[str, Any]],
    as_of_ts: float,
    as_of_iso: str | None,
    now_iso: str | None,
    cfg: CoreConfig,
    as_of_key: str,
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    sigs: list[dict[str, Any]] = []

    def src(sid: str | None) -> str | None:
        s = series_by_id.get(sid or "")
        return s.source if s is not None else None

    for t in trends:
        win = window_label(t)
        s = series_by_id.get(t["series_id"])
        if t["direction"] != "flat":
            q = t["q_value"] if t["q_value"] is not None else 1.0
            label = "rising" if t["direction"] == "up" else "falling"
            unit = (s.trend_unit if s is not None and s.trend_unit else None) or (
                f"{t['metric']}/month" if (s is None or s.freq == "M") else f"{t['metric']}/week"
            )
            sigs.append(
                make_signal(
                    "trend",
                    f"trend:{t['series_id']}",
                    t["entity_type"],
                    t["entity"],
                    f"{t['entity']} {t['metric']} {label}",
                    t["slope"],
                    unit,
                    1 - q,
                    t["direction"],
                    src(t["series_id"]),
                    t["window"]["end"],
                    win,
                    period_age_days(as_of_ts, t["window"]["end"]),
                    [evidence("test", "Mann–Kendall p", t["p_value"], f"BH q {t['q_value']}", t["id"])],
                    as_of_key,
                    baseline=t["prior_mean"],
                    change=_sub(t["recent_mean"], t["prior_mean"]),
                    confidence=1 - q,
                    confidence_kind="evidence",
                    entity_id=t.get("entity_id"),
                )
            )
        cp = t.get("change_point")
        if cp:
            up = (cp["after_mean"] or 0) > (cp["before_mean"] or 0)
            sigs.append(
                make_signal(
                    "change_point",
                    f"cp:{t['series_id']}",
                    t["entity_type"],
                    t["entity"],
                    f"{t['entity']} {t['metric']} shifted {'up' if up else 'down'} in {cp['date'][:7]}",
                    cp["after_mean"],
                    None,
                    1 - (cp["p_value"] or 1.0),
                    "up" if up else "down",
                    src(t["series_id"]),
                    cp["date"],
                    win,
                    period_age_days(as_of_ts, cp["date"]),
                    [
                        evidence(
                            "test",
                            "permutation p",
                            cp["p_value"],
                            f"mean {cp['before_mean']} -> {cp['after_mean']}",
                            t["series_id"],
                        )
                    ],
                    as_of_key,
                    baseline=cp["before_mean"],
                    change=_sub(cp["after_mean"], cp["before_mean"]),
                    confidence=1 - (cp["p_value"] or 1.0),
                    confidence_kind="evidence",
                    entity_id=t.get("entity_id"),
                )
            )
    for a in anomalies:
        if a["suppressed"]:
            continue
        if a["method"] == "robust_z":
            fresh = period_age_days(as_of_ts, a["detected_at"])
            source = a.get("source") or src(a.get("series_id"))
        else:
            fresh, source = 0.0, a.get("source")
        direction = "down" if a["kind"] == "series_drop" else "up"
        s_val = float(a.get("anomaly_score") or 0.0)
        sigs.append(
            make_signal(
                "anomaly",
                a["dedup_key"],
                a["entity_type"],
                a["entity"],
                f"{a['kind'].replace('_', ' ')}: {a['series_id'] or a['entity']}",
                a["value"],
                None,
                s_val,
                direction,
                source,
                a["detected_at"],
                None,
                fresh,
                a["evidence"][:2],
                as_of_key,
                baseline=a["baseline"],
                change=a["deviation"],
                confidence=s_val,
                confidence_kind="margin",
                entity_id=a.get("entity_id"),
            )
        )
    fc_sigs: list[dict[str, Any]] = []
    for f in forecasts:
        bt = f["backtest"]
        if bt["mase"] is None or bt["naive_mase"] is None or bt["mase"] >= bt["naive_mase"]:
            continue
        last = series_last.get(f["series_id"])
        if not last:
            continue
        nxt = float(sum(p["mean"] or 0 for p in f["points"]))
        change = nxt / last - 1
        if abs(change) < FORECAST_SIGNAL_MIN_CHANGE:
            continue
        skill = clip01(1 - bt["mase"] / bt["naive_mase"])
        n = 7 if f["points"][0]["t"].endswith("-01") else 10
        fc_sigs.append(
            make_signal(
                "forecast",
                f"forecast:{f['series_id']}",
                f["entity_type"],
                f["entity"],
                f"{f['entity']} {f['metric']} forecast {'up' if change > 0 else 'down'} {abs(change):.0%}",
                fnum(change, 4),
                "change vs last period",
                min(1.0, abs(change)) * skill,
                "up" if change > 0 else "down",
                src(f["series_id"]),
                f["points"][0]["t"],
                f"{f['points'][0]['t'][:n]}..{f['points'][-1]['t'][:n]}",
                0.0,
                [
                    evidence(
                        "model",
                        f"{f['model']} backtest MASE",
                        bt["mase"],
                        f"naive {bt['naive_mase']}",
                        f["id"],
                    )
                ],
                as_of_key,
                baseline=fnum(last),
                change=fnum(change, 4),
                confidence=skill,
                confidence_kind="margin",
                entity_id=f.get("entity_id"),
            )
        )
    sigs += sorted(fc_sigs, key=lambda x: (-(x["strength"] or 0), x["dedup_key"]))[:FORECAST_SIGNAL_MAX]
    for c in quality["checks"]:
        if c["passed"] or c["severity"] == "info":
            continue
        sigs.append(
            make_signal(
                "quality",
                f"quality:{c['name']}",
                "source",
                c["source"],
                f"Check failed: {c['name']}",
                c["value"],
                None,
                1.0 if c["severity"] == "error" else 0.5,
                None,
                c["source"],
                as_of_iso,
                None,
                0.0,
                [evidence("test", c["name"], c["value"], c["detail"])],
                as_of_key,
                baseline=c["threshold"],
                confidence=1.0,
                confidence_kind="rule",
            )
        )
    sigs.extend(extra or [])
    for r in staleness_risks:
        sigs.append(
            make_signal(
                "live",
                f"live:stale:{r['entity']}",
                "source",
                r["entity"],
                r["title"],
                r["likelihood"],
                None,
                r["likelihood"],
                None,
                r["entity"],
                now_iso,
                None,
                None,
                r["evidence"],
                as_of_key,
                confidence=1.0,
                confidence_kind="rule",
            )
        )
    best: dict[str, dict[str, Any]] = {}
    for sg in sigs:
        cur = best.get(sg["dedup_key"])
        if cur is None or (sg["strength"] or 0) > (cur["strength"] or 0):
            best[sg["dedup_key"]] = sg
    return sorted(best.values(), key=lambda x: (-(x["strength"] or 0), x["dedup_key"]))


def series_last_totals(states: dict[str, Any], horizon: int) -> dict[str, float]:
    """Sum of the last ``horizon`` complete periods per forecast series (the comparison base)."""
    return {k: float(np.nansum(v.history[-horizon:])) for k, v in states.items()}


__all__ = ["build_signals", "iso", "make_signal", "series_last_totals"]
