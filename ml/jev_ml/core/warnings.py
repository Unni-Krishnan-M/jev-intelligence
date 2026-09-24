"""RECOMMEND (early warnings): warnings are downstream of the ``early_warning_level`` decisions.

A warning is raised only for a situation whose decision answered WARNING or URGENT_ACTION, and it
carries that decision's id (``decision_id``) and level (``early_warning_level``). Two modes
(``CoreConfig.warning_mode``):

* ``"components"``: one warning per *qualifying component* of the situation, keyed as in v1.1:
  a risk with level >= ``warning_min_severity`` (key ``risk:<kind>:<entity>``) or a non-suppressed
  anomaly with severity >= ``warning_min_severity`` (series anomalies only while they fall in the
  last ``warning_anomaly_recent_months`` complete periods), keyed by the anomaly's dedup key. The
  severity is the component's own severity, which is exactly what put the situation at WARNING
  (medium) or URGENT_ACTION (high, critical). If corroboration alone lifted a situation to WARNING
  without a qualifying component, one situation warning is raised as in the other mode.
* ``"situation"``: one warning per situation, key ``warning:<situation key>``. Severity = the most
  severe qualifying component, but at least the level's floor (WARNING -> medium, URGENT_ACTION ->
  high). Its confidence is the decision's margin.

Keys an operator dismissed are dropped unless the severity escalated. One warning per key (the most
severe, then highest observed value), sorted by severity, confidence, key; at most ``max_warnings``.
``confidence`` is carried from the source (risk confidence, the anomaly's normalised strength, or the
decision's margin) and is labelled ``confidence_kind: "margin"``: evidence strength, not a probability.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jev_ml.core.anomalies import suppression
from jev_ml.core.common import evidence
from jev_ml.core.config import CoreConfig, severity_rank

LEVEL_FLOOR = {"WARNING": "medium", "URGENT_ACTION": "high"}
WARN_LEVELS = ("WARNING", "URGENT_ACTION")
AnomalyText = Callable[[dict[str, Any], CoreConfig], dict[str, Any] | None]


def robust_z_text(a: dict[str, Any], cfg: CoreConfig) -> dict[str, Any]:
    what = "spike" if a["kind"] == "series_spike" else "drop"
    t = a["detected_at"][:7] if a["detected_at"].endswith("-01") else a["detected_at"]
    return {
        "condition": f"|robust z| {abs(a['score']):.2f} >= {cfg.anomaly_z_threshold}",
        "observed": abs(a["score"]),
        "threshold": cfg.anomaly_z_threshold,
        "title": f"{a['series_id']} {what} in {t}",
        "description": (
            f"{a['series_id']} changed by {a['value']:+g} in {t} (to {a.get('level')}) "
            f"vs a typical change of "
            f"{a['baseline']:+g} (robust z {a['score']:.1f})."
            if a.get("anomaly_basis") == "change"
            else f"{a['series_id']} was {a['value']} in {t} vs a trailing median of {a['baseline']} "
            f"(robust z {a['score']:.1f})."
        ),
        "action": "Check what drives the change before acting on it; confirm it in the next period.",
    }


def _risk_warning(r: dict[str, Any], cfg: CoreConfig) -> dict[str, Any]:
    thr = cfg.risk_levels[cfg.warning_min_severity]
    return {
        "key": f"risk:{r['kind']}:{r['entity']}",
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


def _anomaly_warning(a: dict[str, Any], cfg: CoreConfig, text: AnomalyText | None) -> dict[str, Any]:
    t = (text(a, cfg) if text is not None else None) or (
        robust_z_text(a, cfg) if a["method"] == "robust_z" else None
    )
    if t is None:  # an adapter-specific method without a formatter: neutral wording
        t = {
            "condition": f"{a['method']} severity {a['severity']}",
            "observed": a["score"],
            "threshold": None,
            "title": f"{a['kind'].replace('_', ' ')}: {a['entity']}",
            "description": f"{a['kind']} on {a['entity']} (score {a['score']}).",
            "action": "Investigate the anomaly.",
        }
    return {
        "key": a["dedup_key"],
        "title": t["title"],
        "description": t["description"],
        "severity": a["severity"],
        "confidence": round(float(a.get("anomaly_score") or 0.0), 4),
        "confidence_kind": "margin",
        "trigger": {
            "rule": f"anomaly.severity>={cfg.warning_min_severity}",
            "condition": t["condition"],
            "observed": t["observed"],
            "threshold": t["threshold"],
        },
        "evidence": [*a["evidence"], evidence("record", "anomaly", a["id"], a["kind"], a["id"])],
        "recommended_action": t["action"],
        "source": {"type": "anomaly", "id": a["id"]},
        "entity_type": a["entity_type"],
        "entity": a["entity"],
    }


def _situation_warning(
    d: dict[str, Any], comps: list[dict[str, Any]], objs: dict[str, dict[str, Any]], cfg: CoreConfig
) -> dict[str, Any]:
    level = d["answer"]
    warnable = [c for c in comps if c.get("warnable")]
    sev = LEVEL_FLOOR[level]
    for c in warnable:
        if severity_rank(c["severity"]) > severity_rank(sev):
            sev = c["severity"]
    # strongest first; on equal points the risk (which names the consequence) leads the title
    ranked = sorted(comps, key=lambda c: (-float(c["points"] or 0.0), c["stage"] != "risk"))
    top = ranked[0]
    top_obj = objs.get(top["ref"] or "") or {}
    action = top_obj.get("recommended_response") or (
        "Review the situation's evidence; confirm the move in the next period before acting."
    )
    thr = cfg.ewl_level_points["WARNING"]
    p = float(d["state"].get("points") or 0.0)
    parts = [f"{c['title']} ({c['points']:.1f} pts)" for c in ranked[:3] if c["points"]]
    return {
        "key": f"warning:{d['situation']}",
        "title": f"{top['title']}" + (f" (+{len(ranked) - 1} more signals)" if len(ranked) > 1 else ""),
        "description": f"Early-warning level {level} at {p:.2f} points: " + "; ".join(parts) + ".",
        "severity": sev,
        "confidence": d["confidence"],
        "confidence_kind": "margin",
        "trigger": {
            "rule": "early_warning_level>=WARNING",
            "condition": f"points {p:.2f} >= {thr:g}",
            "observed": p,
            "threshold": thr,
        },
        "evidence": list(d["evidence"]),
        "recommended_action": action,
        "source": {"type": "decision", "id": d["id"]},
        "entity_type": d["entity_type"],
        "entity": d["entity"],
    }


def build_warnings(
    ewl_decisions: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    suppressed: dict[str, str],
    cfg: CoreConfig,
    anomaly_text: AnomalyText | None = None,
) -> list[dict[str, Any]]:
    objs: dict[str, dict[str, Any]] = {r["id"]: r for r in risks}
    objs.update({a["id"]: a for a in anomalies})
    cands: list[dict[str, Any]] = []
    for d in ewl_decisions:
        if d["abstained"] or d["answer"] not in WARN_LEVELS:
            continue
        comps = d["state"]["components"]
        made = []
        if cfg.warning_mode == "components":
            for c in comps:
                if not c.get("warnable"):
                    continue
                o = objs.get(c["ref"] or "")
                if o is None:
                    continue
                w = _risk_warning(o, cfg) if c["stage"] == "risk" else _anomaly_warning(o, cfg, anomaly_text)
                if c["stage"] == "risk" and suppression(w["key"], w["severity"], suppressed):
                    continue
                made.append(w)
        if not made and not (cfg.warning_mode == "components" and any(c.get("warnable") for c in comps)):
            w = _situation_warning(d, comps, objs, cfg)
            if suppression(w["key"], w["severity"], suppressed):
                continue
            made.append(w)
        for w in made:
            w["decision_id"] = d["id"]
            w["early_warning_level"] = d["answer"]
            cands.append(w)
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
