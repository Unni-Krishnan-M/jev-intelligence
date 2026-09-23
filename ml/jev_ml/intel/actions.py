"""ACT: action plan from decisions and warnings.

priority_score (0..1) = max(linked risk score / 100, severity weight of the linked warning,
decision confidence x 0.5); P1 >= 0.6, P2 >= 0.35, else P3. Each action type has a *declared*
effort estimate (EFFORT below): it is a planning assumption, not something the data measures.
Warnings already answered by a decision action are not duplicated as "investigate" actions.
"""

from __future__ import annotations

from typing import Any

from jev_ml.intel.common import evidence, fnum, stable_id

# Declared effort estimates per action type (planning assumptions, not measured).
EFFORT = {
    "retrain_model": "medium",
    "serving_model": "low",
    "genre_programming": "low",
    "rater_monitor": "low",
    "rater_quarantine": "medium",
    "reengagement_campaign": "medium",
    "investigate": "low",
}
SEV_WEIGHT = {"low": 0.2, "medium": 0.45, "high": 0.7, "critical": 0.9}


def _priority(score: float) -> str:
    return "P1" if score >= 0.6 else "P2" if score >= 0.35 else "P3"


def build_actions(
    decisions: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    as_of_key: str,
) -> list[dict[str, Any]]:
    risk_by_key = {f"{r['kind']}:{r['entity']}": r for r in risks}
    warn_by_key = {w["key"]: w for w in warnings}
    covered: set[str] = set()
    out: list[dict[str, Any]] = []

    def add(
        kind: str,
        title: str,
        reason: str,
        impact: str,
        risk_text: str,
        next_step: str,
        source: dict[str, str],
        ev: list[dict[str, Any]],
        linked: list[str],
        conf: float | None,
    ) -> None:
        score = (conf or 0.0) * 0.5
        for key in linked:
            r = risk_by_key.get(key)
            if r:
                score = max(score, (r["score"] or 0.0) / 100.0)
            w = warn_by_key.get(f"risk:{key}")
            if w:
                score = max(score, SEV_WEIGHT[w["severity"]])
                covered.add(w["key"])
        out.append(
            {
                "id": stable_id("act", kind, source["id"], as_of_key),
                "title": title,
                "priority": _priority(score),
                "priority_score": fnum(score, 4),
                "reason": reason,
                "expected_impact": impact,
                "effort": EFFORT[kind],
                "effort_basis": "declared estimate per action type",
                "risk": risk_text,
                "evidence": ev,
                "next_step": next_step,
                "source": source,
            }
        )

    current = (manifest or {}).get("model_type")
    for d in decisions:
        if d["abstained"]:
            continue
        src = {"type": "decision", "id": d["id"]}
        st = d["state"]
        if d["key"] == "retrain_model" and d["answer"] == "yes":
            add(
                "retrain_model",
                "Retrain the recommendation model",
                d["rationale"][0],
                f"model catches up with {st['new_events']['total']} new ratings",
                "a retrained model can regress; compare it on the same split before activating",
                "run scripts/train_models.py and review the new experiment report",
                src,
                d["evidence"],
                [f"model_staleness:{d['entity']}"],
                d["confidence"],
            )
        elif d["key"] == "serving_model" and d["answer"] and current and d["answer"] != current:
            add(
                "serving_model",
                f"Switch serving model to {d['answer']}",
                d["rationale"][0],
                f"mean NDCG@10 {st['mean_ndcg10'].get(d['answer'])} vs {st['mean_ndcg10'].get(current)} "
                f"for {current}",
                "offline NDCG may not transfer to live engagement",
                "A/B the candidate on a share of traffic before a full switch",
                src,
                d["evidence"],
                [f"model_quality:{d['entity']}"],
                d["confidence"],
            )
        elif d["key"] == "genre_programming" and d["answer"] in ("boost", "reduce"):
            g = d["entity"]
            add(
                "genre_programming",
                f"{d['answer'].capitalize()} {g} on the home rails",
                d["rationale"][0],
                f"{g} holds {100 * (st['recent_share'] or 0):.1f} % of recent ratings",
                "trend may be driven by a few heavy raters; check before large changes",
                f"adjust the {g} slot share in the next programming review",
                src,
                d["evidence"],
                [f"genre_demand_decline:{g}"],
                d["confidence"],
            )
        elif d["key"] == "rater_action" and d["answer"] in ("monitor", "quarantine"):
            kind = "rater_monitor" if d["answer"] == "monitor" else "rater_quarantine"
            add(
                kind,
                f"{d['answer'].capitalize()} rater {d['entity']}",
                d["rationale"][0],
                f"removing this rater changes {st['top50_changed_if_removed']} of the top-50 trending films",
                "a heavy genuine user can look anomalous; quarantine hides real preferences",
                "review the rater's recent ratings"
                if kind == "rater_monitor"
                else "exclude the rater from popularity/trending inputs pending review",
                src,
                d["evidence"],
                ["rating_manipulation:all"],
                d["confidence"],
            )
        elif d["key"] == "reengagement_campaign" and d["answer"] == "yes":
            add(
                "reengagement_campaign",
                f"Re-engage {st['high_risk']} users at high lapse risk",
                d["rationale"][0],
                f"{st['expected_lapses_high_risk']} expected lapses among targeted users in the next "
                "180 days",
                "messages to users who would have returned anyway add noise",
                "export the high-risk list and schedule a personalised new-releases message",
                src,
                d["evidence"],
                ["audience_lapse:all"],
                d["confidence"],
            )
    for w in warnings:
        if w["key"] in covered:
            continue
        score = SEV_WEIGHT[w["severity"]]
        out.append(
            {
                "id": stable_id("act", "investigate", w["key"], as_of_key),
                "title": f"Investigate: {w['title']}",
                "priority": _priority(score),
                "priority_score": fnum(score, 4),
                "reason": w["description"],
                "expected_impact": "confirms or rules out the warning before any change is made",
                "effort": EFFORT["investigate"],
                "effort_basis": "declared estimate per action type",
                "risk": "none beyond analyst time",
                "evidence": [*w["evidence"][:3], evidence("record", "warning", w["key"], w["severity"])],
                "next_step": w["recommended_action"],
                "source": {"type": "warning", "id": w["key"]},
            }
        )
    order = {"P1": 0, "P2": 1, "P3": 2}
    return sorted(out, key=lambda a: (order[a["priority"]], -(a["priority_score"] or 0), a["id"]))
