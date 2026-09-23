"""RECOMMEND (early warnings): warning candidates from risks and anomalies.

* Risks with level >= warning_min_severity -> key ``risk:<kind>:<entity>``.
* Non-suppressed anomalies with severity >= warning_min_severity detected in the last
  ``warning_anomaly_recent_months`` complete months (older anomalies stay listed in ``anomalies`` but
  do not raise new warnings) -> key = the anomaly's dedup key (one per series/user, not per month).
* Keys an operator dismissed are dropped unless the severity escalated (see PipelineInputs).
* One warning per key: the most severe (then highest observed value) candidate wins.

``confidence`` is carried from the source (risk confidence, or the anomaly's normalised strength)
and is labelled ``confidence_kind: "margin"``: it measures evidence strength, not a probability.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from jev_ml.intel.anomalies import suppression
from jev_ml.intel.common import clip01, evidence
from jev_ml.intel.config import IntelConfig, severity_rank


def _anomaly_conf(a: dict[str, Any], cfg: IntelConfig) -> float:
    if a["method"] == "robust_z":
        return clip01(abs(a["score"]) / (2 * cfg.anomaly_z_threshold))
    if a["method"] == "isolation_forest":
        return clip01((a["value"] - 0.5) / 0.3)
    return clip01(abs(a["score"] or 0.0) / 4.0)  # rate test: z of 4 = full strength


def build_warnings(
    risks: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    suppressed: dict[str, str],
    last_complete: str | None,
    cfg: IntelConfig,
) -> list[dict[str, Any]]:
    min_rank = severity_rank(cfg.warning_min_severity)
    cands: list[dict[str, Any]] = []
    for r in risks:
        if severity_rank(r["level"]) < min_rank:
            continue
        key = f"risk:{r['kind']}:{r['entity']}"
        if suppression(key, r["level"], suppressed):
            continue
        thr = cfg.risk_levels[cfg.warning_min_severity]
        cands.append(
            {
                "key": key,
                "title": r["title"],
                "description": (
                    f"{r['title']}. Risk score {r['score']:.1f}/100 (likelihood {r['likelihood']:.2f} x "
                    "impact "
                    f"{r['impact']:.2f} x confidence {r['confidence']:.2f} x data quality "
                    f"{r['data_quality']:.2f})."
                ),
                "severity": r["level"],
                "confidence": r["confidence"],
                "confidence_kind": "margin",
                "trigger": {
                    "rule": f"risk.level>={cfg.warning_min_severity}",
                    "condition": f"score {r['score']:.1f} >= {thr}",
                    "observed": r["score"],
                    "threshold": thr,
                },
                "evidence": r["evidence"],
                "recommended_action": r["recommended_response"],
                "source": {"type": "risk", "id": r["id"]},
                "entity_type": r["entity_type"],
                "entity": r["entity"],
            }
        )
    recent_from = None
    if last_complete:
        recent_from = (
            pd.Period(last_complete[:7], freq="M") - (cfg.warning_anomaly_recent_months - 1)
        ).strftime("%Y-%m-01")
    for a in anomalies:
        if a["suppressed"] or severity_rank(a["severity"]) < min_rank:
            continue
        if a["method"] == "robust_z" and recent_from and a["detected_at"] < recent_from:
            continue
        if a["method"] == "robust_z":
            cond = f"|robust z| {abs(a['score']):.2f} >= {cfg.anomaly_z_threshold}"
            observed, thr = abs(a["score"]), cfg.anomaly_z_threshold
            what = "spike" if a["kind"] == "series_spike" else "drop"
            title = f"{a['series_id']} {what} in {a['detected_at'][:7]}"
            desc = (
                f"{a['series_id']} was {a['value']} in {a['detected_at'][:7]} vs a trailing median of "
                f"{a['baseline']} (robust z {a['score']:.1f})."
            )
            action = (
                "Check whether the change comes from a few users or a catalogue event before acting on it."
            )
        elif a["method"] == "isolation_forest":
            band = cfg.rater_severity_bands.get(a["severity"])
            cond = f"isolation-forest score {a['value']:.3f} >= {band}"
            observed, thr = a["value"], (band if band is not None else 0.0)
            title = f"Rater {a['entity']} behaves anomalously"
            desc = (
                f"User {a['entity']} is in the top {100 - a['score']:.1f} % most anomalous raters "
                f"(score {a['value']:.3f})."
            )
            action = "Review the rater's history; see the rater_action decision."
        else:
            cond = f"one-sided two-proportion test p < {cfg.live_alpha}"
            observed, thr = a["value"], a["baseline"]
            title = "Negative recommendation feedback rising"
            desc = (
                f"Negative feedback share {a['value']:.2%} in the last 7 days vs {a['baseline']:.2%} before."
            )
            action = "Inspect which reason codes attract dislikes and adjust those rails."
        cands.append(
            {
                "key": a["dedup_key"],
                "title": title,
                "description": desc,
                "severity": a["severity"],
                "confidence": round(_anomaly_conf(a, cfg), 4),
                "confidence_kind": "margin",
                "trigger": {
                    "rule": f"anomaly.severity>={cfg.warning_min_severity}",
                    "condition": cond,
                    "observed": observed,
                    "threshold": thr,
                },
                "evidence": [*a["evidence"], evidence("record", "anomaly", a["id"], a["kind"], a["id"])],
                "recommended_action": action,
                "source": {"type": "anomaly", "id": a["id"]},
                "entity_type": a["entity_type"],
                "entity": a["entity"],
            }
        )
    best: dict[str, dict[str, Any]] = {}
    for c in cands:
        cur = best.get(c["key"])
        rank = (severity_rank(c["severity"]), float(c["trigger"]["observed"] or 0.0))
        if cur is None or rank > (severity_rank(cur["severity"]), float(cur["trigger"]["observed"] or 0.0)):
            best[c["key"]] = c
    out = sorted(
        best.values(), key=lambda w: (-severity_rank(w["severity"]), -(w["confidence"] or 0.0), w["key"])
    )
    return out[: cfg.max_warnings]
