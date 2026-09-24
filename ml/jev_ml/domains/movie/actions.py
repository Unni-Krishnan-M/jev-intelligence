"""ACT (movie domain): movie decisions -> actions (the core adds "investigate" actions for warnings).

Each action type has a *declared* effort estimate (EFFORT): a planning assumption, not something the
data measures. Priority scoring and warning coverage: ``jev_ml.core.actions.ActionPlan``.
"""

from __future__ import annotations

from typing import Any

from jev_ml.core.actions import ActionPlan

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


def decision_actions(
    plan: ActionPlan, decisions: list[dict[str, Any]], manifest: dict[str, Any] | None
) -> None:
    current = (manifest or {}).get("model_type")
    for d in decisions:
        if d["abstained"]:
            continue
        src = {"type": "decision", "id": d["id"]}
        st = d["state"]
        if d["key"] == "retrain_model" and d["answer"] == "yes":
            plan.add(
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
                EFFORT["retrain_model"],
            )
        elif d["key"] == "serving_model" and d["answer"] and current and d["answer"] != current:
            plan.add(
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
                EFFORT["serving_model"],
            )
        elif d["key"] == "genre_programming" and d["answer"] in ("boost", "reduce"):
            g = d["entity"]
            plan.add(
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
                EFFORT["genre_programming"],
            )
        elif d["key"] == "rater_action" and d["answer"] in ("monitor", "quarantine"):
            kind = "rater_monitor" if d["answer"] == "monitor" else "rater_quarantine"
            plan.add(
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
                EFFORT[kind],
            )
        elif d["key"] == "reengagement_campaign" and d["answer"] == "yes":
            plan.add(
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
                EFFORT["reengagement_campaign"],
            )
