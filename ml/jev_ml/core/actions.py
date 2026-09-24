"""ACT: the action plan from decisions, warnings and risks.

priority_score (0..1) = max(linked risk score / 100, severity weight of the linked warning,
decision confidence x 0.5); P1 >= 0.6, P2 >= 0.35, else P3. Each action type has a *declared*
effort estimate: a planning assumption, not something the data measures.

Domain decisions become actions through the adapter's hook (``DomainExtras.decision_actions``),
which calls ``ActionPlan.add``. Every warning not already answered by a decision action becomes an
"investigate" action. The core itself has no domain-specific action types.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from jev_ml.core.common import evidence, fnum, stable_id

SEV_WEIGHT = {"low": 0.2, "medium": 0.45, "high": 0.7, "critical": 0.9}
INVESTIGATE_EFFORT = "low"  # declared


def priority_of(score: float) -> str:
    return "P1" if score >= 0.6 else "P2" if score >= 0.35 else "P3"


class ActionPlan:
    def __init__(self, warnings: list[dict[str, Any]], risks: list[dict[str, Any]], as_of_key: str) -> None:
        self.risk_by_key = {f"{r['kind']}:{r['entity']}": r for r in risks}
        self.warn_by_key = {w["key"]: w for w in warnings}
        self.covered: set[str] = set()
        self.items: list[dict[str, Any]] = []
        self.as_of_key = as_of_key

    def add(
        self,
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
        effort: str,
    ) -> None:
        """``linked`` = risk keys ``<kind>:<entity>``; their warnings count as covered."""
        score = (conf or 0.0) * 0.5
        for key in linked:
            r = self.risk_by_key.get(key)
            if r:
                score = max(score, (r["score"] or 0.0) / 100.0)
            w = self.warn_by_key.get(f"risk:{key}")
            if w:
                score = max(score, SEV_WEIGHT[w["severity"]])
                self.covered.add(w["key"])
        self.items.append(
            {
                "id": stable_id("act", kind, source["id"], self.as_of_key),
                "title": title,
                "priority": priority_of(score),
                "priority_score": fnum(score, 4),
                "reason": reason,
                "expected_impact": impact,
                "effort": effort,
                "effort_basis": "declared estimate per action type",
                "risk": risk_text,
                "evidence": ev,
                "next_step": next_step,
                "source": source,
            }
        )


DecisionActions = Callable[[ActionPlan, list[dict[str, Any]]], None]


def build_actions(
    decisions: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    as_of_key: str,
    decision_actions: DecisionActions | None = None,
) -> list[dict[str, Any]]:
    plan = ActionPlan(warnings, risks, as_of_key)
    if decision_actions is not None:
        decision_actions(plan, decisions)
    out = plan.items
    for w in warnings:
        if w["key"] in plan.covered:
            continue
        score = SEV_WEIGHT[w["severity"]]
        act = {
            "id": stable_id("act", "investigate", w["key"], as_of_key),
            "title": f"Investigate: {w['title']}",
            "priority": priority_of(score),
            "priority_score": fnum(score, 4),
            "reason": w["description"],
            "expected_impact": "confirms or rules out the warning before any change is made",
            "effort": INVESTIGATE_EFFORT,
            "effort_basis": "declared estimate per action type",
            "risk": "none beyond analyst time",
            "evidence": [*w["evidence"][:3], evidence("record", "warning", w["key"], w["severity"])],
            "next_step": w["recommended_action"],
            "source": {"type": "warning", "id": w["key"]},
        }
        if w.get("decision_id"):
            act["decision_id"] = w["decision_id"]
        out.append(act)
    order = {"P1": 0, "P2": 1, "P3": 2}
    return sorted(out, key=lambda a: (order[a["priority"]], -(a["priority_score"] or 0), a["id"]))
