"""Promotion rules, kept pure so the API, the CLI and the scheduler apply the same policy.

A version may be promoted without ``force`` only when:
* it is not already active and not rejected;
* no model serves at all (bootstrap), or:
* its latest gate passed; and
* that gate compared it against the model that is active *now* (a gate against an incumbent that has
  since been replaced proves nothing about the current one), or there is no active model at all
  (bootstrap) and the gate ran without an incumbent.

``force`` (admin, with a reason, audited) bypasses the gate but never the artifact check: the
EngineHolder always loads a version before swapping to it, and never swaps to one that fails.
"""

from __future__ import annotations

from typing import Any


def promotion_blockers(
    version: str,
    state: str | None,
    gate: dict[str, Any] | None,
    current_active: str | None,
) -> list[str]:
    """Why ``version`` cannot be promoted without force (empty list = promotable)."""
    if state == "active" or version == current_active:
        return [f"{version} is already the active model"]
    out: list[str] = []
    if state == "rejected":
        out.append(f"{version} was rejected by its gate")
    if current_active is None:  # bootstrap: nothing serves, so there is nothing to protect
        return out
    if gate is None:
        out.append("no gate result: evaluate the version first")
        return out
    if not gate.get("passed"):
        out.extend(gate.get("reasons") or ["the gate did not pass"])
    gated_against = gate.get("incumbent")
    if gated_against != current_active:
        out.append(
            f"the gate compared against {gated_against or 'no model'}, but the active model is "
            f"{current_active or 'none'}: evaluate again"
        )
    return out
