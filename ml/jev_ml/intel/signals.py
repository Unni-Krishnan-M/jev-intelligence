"""UNDERSTAND/DETECT -> Signal objects: compact, deduplicated observations for the console.

Strength (0..1) per kind:
* trend: 1 - BH q-value of the Mann–Kendall test (non-flat trends only).
* change_point: 1 - permutation p-value.
* anomaly: robust z -> min(1, |z| / (2 * threshold)); isolation forest -> clip((s - 0.5) / 0.3);
  rate test -> min(1, |z| / 4). Suppressed anomalies produce no signal.
* forecast: min(1, |next-horizon total vs last-horizon total| / 100 %) x skill,
  skill = clip(1 - mase/naive_mase);
  only when |change| >= 25 % and the model beats naive in the backtest; at most 5 (strongest).
* quality: 1.0 for a failed error-level check, 0.5 for a warning-level check.
* live: negative-feedback test -> 1 - p; app staleness -> the staleness risk likelihood.
* model: new-data share / retrain threshold (capped at 1); hybrid lead -> bootstrap P(hybrid better).

Dedup: one signal per ``dedup_key``; the strongest wins.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jev_ml.intel.common import clip01, evidence, fnum, iso, stable_id
from jev_ml.intel.config import IntelConfig
from jev_ml.intel.trends import window_label

FORECAST_SIGNAL_MIN_CHANGE = 0.25
# genre forecasts move together with platform volume; keeping only the strongest few avoids a
# dozen near-identical "forecast down" signals after a burst month
FORECAST_SIGNAL_MAX = 5


def _sig(
    kind: str,
    key: str,
    entity_type: str,
    entity: str,
    title: str,
    value: Any,
    unit: str | None,
    strength: float,
    direction: str | None,
    source: str,
    observed_at: str | None,
    window: str | None,
    freshness_days: float | None,
    ev: list[dict[str, Any]],
    as_of_key: str,
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
    }


def _days(as_of_ts: float, month: str) -> float:
    import pandas as pd

    end = pd.Period(month[:7], freq="M").end_time.tz_localize("UTC").timestamp()
    return max(0.0, (as_of_ts - end) / 86400.0)


def build_signals(
    trends: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    series_last: dict[str, float],
    prep: Any,
    live: dict[str, Any],
    stale: dict[str, Any] | None,
    boot: dict[str, Any] | None,
    staleness_risks: list[dict[str, Any]],
    cfg: IntelConfig,
    as_of_key: str,
) -> list[dict[str, Any]]:
    sigs: list[dict[str, Any]] = []
    a_ts = prep.as_of_ts
    for t in trends:
        win = window_label(t)
        if t["direction"] != "flat":
            q = t["q_value"] if t["q_value"] is not None else 1.0
            label = "rising" if t["direction"] == "up" else "falling"
            unit = {"share": "share/month", "rating": "stars/month"}.get(t["metric"], f"{t['metric']}/month")
            sigs.append(
                _sig(
                    "trend",
                    f"trend:{t['series_id']}",
                    t["entity_type"],
                    t["entity"],
                    f"{t['entity']} {t['metric']} {label}",
                    t["slope"],
                    unit,
                    1 - q,
                    t["direction"],
                    "movielens",
                    t["window"]["end"],
                    win,
                    _days(a_ts, t["window"]["end"]),
                    [evidence("test", "Mann–Kendall p", t["p_value"], f"BH q {t['q_value']}", t["id"])],
                    as_of_key,
                )
            )
        cp = t.get("change_point")
        if cp:
            up = (cp["after_mean"] or 0) > (cp["before_mean"] or 0)
            sigs.append(
                _sig(
                    "change_point",
                    f"cp:{t['series_id']}",
                    t["entity_type"],
                    t["entity"],
                    f"{t['entity']} {t['metric']} shifted {'up' if up else 'down'} in {cp['date'][:7]}",
                    cp["after_mean"],
                    None,
                    1 - (cp["p_value"] or 1.0),
                    "up" if up else "down",
                    "movielens",
                    cp["date"],
                    win,
                    _days(a_ts, cp["date"]),
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
                )
            )
    for a in anomalies:
        if a["suppressed"]:
            continue
        if a["method"] == "robust_z":
            s = abs(a["score"]) / (2 * cfg.anomaly_z_threshold)
            observed, fresh, src = a["detected_at"], _days(a_ts, a["detected_at"]), "movielens"
        elif a["method"] == "isolation_forest":
            s = (a["value"] - 0.5) / 0.3
            observed, fresh, src = a["detected_at"], 0.0, "movielens"
        else:
            s = abs(a["score"]) / 4.0
            observed, fresh, src = a["detected_at"], 0.0, "app"
        direction = "down" if a["kind"] == "series_drop" else "up"
        sigs.append(
            _sig(
                "anomaly",
                a["dedup_key"],
                a["entity_type"],
                a["entity"],
                f"{a['kind'].replace('_', ' ')}: {a['series_id'] or a['entity']}",
                a["value"],
                None,
                s,
                direction,
                src,
                observed,
                None,
                fresh,
                a["evidence"][:2],
                as_of_key,
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
        fc_sigs.append(
            _sig(
                "forecast",
                f"forecast:{f['series_id']}",
                f["entity_type"],
                f["entity"],
                f"{f['entity']} {f['metric']} forecast {'up' if change > 0 else 'down'} {abs(change):.0%}",
                fnum(change, 4),
                "change vs last period",
                min(1.0, abs(change)) * skill,
                "up" if change > 0 else "down",
                "movielens",
                f["points"][0]["t"],
                f"{f['points'][0]['t'][:7]}..{f['points'][-1]['t'][:7]}",
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
            )
        )
    sigs += sorted(fc_sigs, key=lambda x: (-(x["strength"] or 0), x["dedup_key"]))[:FORECAST_SIGNAL_MAX]
    for c in prep.quality["checks"]:
        if c["passed"] or c["severity"] == "info":
            continue
        sigs.append(
            _sig(
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
                iso(prep.as_of),
                None,
                0.0,
                [evidence("test", c["name"], c["value"], c["detail"])],
                as_of_key,
            )
        )
    if live.get("status") == "ok":
        d = live["recent_rate"] - live["prior_rate"]
        sigs.append(
            _sig(
                "live",
                "live:negative_feedback",
                "platform",
                "app",
                "Negative recommendation feedback (7 d)",
                fnum(live["recent_rate"], 4),
                "share",
                1 - live["p_value"],
                "up" if d > 0 else "down" if d < 0 else "flat",
                "app",
                iso(prep.now),
                f"last {cfg.live_recent_days} d vs prior {cfg.live_prior_days} d",
                0.0,
                [evidence("test", "two-proportion z", fnum(live["z"], 3), f"p {live['p_value']:.4f}")],
                as_of_key,
            )
        )
    for r in staleness_risks:
        sigs.append(
            _sig(
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
                iso(prep.now),
                None,
                None,
                r["evidence"],
                as_of_key,
            )
        )
    if stale is not None:
        ratio = stale["new_frac"] / cfg.retrain_new_event_frac
        sigs.append(
            _sig(
                "model",
                "model:new_data",
                "model",
                str(prep.model_version),
                f"{stale['new_events']['total']} ratings since model training",
                stale["new_events"]["total"],
                "ratings",
                ratio,
                "up" if stale["new_events"]["total"] else "flat",
                "model",
                iso(prep.as_of),
                None,
                0.0,
                [
                    evidence(
                        "metric",
                        "new-event share",
                        fnum(stale["new_frac"], 6),
                        f"threshold {cfg.retrain_new_event_frac}",
                    )
                ],
                as_of_key,
            )
        )
    if boot and not boot.get("insufficient") and boot.get("hybrid") and prep.model_replayable:
        h = boot["hybrid"]
        sigs.append(
            _sig(
                "model",
                "model:hybrid_lead",
                "model",
                str(prep.model_version),
                f"Hybrid leads {h['best_single']} by {100 * h['lead']:.1f} % NDCG@10",
                fnum(h["lead"], 4),
                "relative NDCG@10",
                h["p_hybrid_better"],
                "up" if h["lead"] > 0 else "down",
                "model",
                iso(prep.as_of),
                None,
                0.0,
                [
                    evidence(
                        "test",
                        "bootstrap P(hybrid better)",
                        fnum(h["p_hybrid_better"], 4),
                        f"95 % CI of lead {[round(x, 4) for x in h['lead_ci95']]}",
                    )
                ],
                as_of_key,
            )
        )
    best: dict[str, dict[str, Any]] = {}
    for s in sigs:
        cur = best.get(s["dedup_key"])
        if cur is None or (s["strength"] or 0) > (cur["strength"] or 0):
            best[s["dedup_key"]] = s
    return sorted(best.values(), key=lambda s: (-(s["strength"] or 0), s["dedup_key"]))


def series_last_totals(states: dict[str, Any], horizon: int) -> dict[str, float]:
    """Sum of the last horizon complete months per forecast series (the comparison base)."""
    return {k: float(np.nansum(v.history[-horizon:])) for k, v in states.items()}
