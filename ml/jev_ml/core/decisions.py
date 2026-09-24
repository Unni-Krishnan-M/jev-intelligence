"""The JEV decision framework: bounded, typed decisions with versioned policies.

A decision answers one fixed question with a declared answer type (``boolean``, ``choice``,
``score``), a fixed option set and a versioned policy. Normal code computes the evidence; the policy
only combines it. Every decision records its input ``state``, the answer and a score per option, a
confidence *with its kind*, a rationale, evidence and its policy version, or ``abstained: true``
with a ``fallback_reason``.

Confidence kinds (``CONFIDENCE_KINDS``):

* ``probability``: from a probabilistic computation (bootstrap, calibrated model);
* ``margin``: normalised margin between the best and second-best option scores,
  (s1 - s2) / (s1 + s2); *not* a probability;
* ``rule``: a deterministic threshold rule (1.0 when it fires);
* ``interval``: the nominal coverage of ``answer_interval`` (score decisions);
* ``evidence``: 1 - (adjusted) p-value of the underlying test; strength of evidence, not a
  probability that the answer is right.

``DecisionSpec`` declares a question once; ``spec.decide(...)`` / ``spec.abstain(...)`` build the
wire record. Decisions are produced inside batches (``jev_ml.core.batches``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from jev_ml.core.common import fnum, stable_id

CONFIDENCE_KINDS = ("probability", "margin", "rule", "interval", "evidence")
DECISION_KINDS = ("boolean", "choice", "score")


def decision(
    key: str,
    policy: str,
    question: str,
    kind: str,
    options: list[str],
    answer: str | float | None,
    scores: dict[str, float],
    confidence: float | None,
    confidence_kind: str,
    state: dict[str, Any],
    rationale: list[str],
    ev: list[dict[str, Any]],
    entity_type: str,
    entity: str,
    as_of_key: str,
    fallback: str | None = None,
    scale: dict[str, Any] | None = None,
    answer_interval: list[float | None] | None = None,
) -> dict[str, Any]:
    """The decision wire record (docs/intelligence.md section 4 + 9.1). ``fallback`` = abstention."""
    if kind == "score" and fallback is None and not isinstance(answer, int | float):
        raise ValueError(f"{key}: a score decision needs a numeric answer")
    return {
        "id": stable_id("dec", key, entity, as_of_key),
        "key": key,
        "spec_id": key,
        "policy_version": policy,
        "question": question,
        "kind": kind,
        "options": options,
        "answer": answer,
        "option_scores": {k: fnum(v, 4) for k, v in scores.items()},
        "confidence": fnum(confidence, 4),
        "confidence_kind": confidence_kind,
        "state": state,
        "rationale": rationale,
        "evidence": ev,
        "abstained": fallback is not None,
        "fallback_reason": fallback,
        "entity_type": entity_type,
        "entity": entity,
        "scale": scale,
        "answer_interval": answer_interval,
        "batch_id": None,  # set by core.batches.run_batch
    }


def margin(scores: dict[str, float], answer: str) -> float:
    """(s_answer - best other) / (s_answer + best other), floored at 0."""
    others = [v for k, v in scores.items() if k != answer]
    if not others:
        return 1.0
    s1, s2 = scores[answer], max(others)
    return max(0.0, (s1 - s2) / (s1 + s2)) if s1 + s2 > 0 else 0.0


@dataclass(frozen=True)
class DecisionSpec:
    """A bounded question declared once (key, policy version, answer type, options, confidence kind)."""

    key: str
    policy_version: str
    kind: str
    confidence_kind: str
    question: str
    options: tuple[str, ...] = ()
    scale: dict[str, Any] | None = field(default=None, hash=False)

    def __post_init__(self) -> None:
        if self.kind not in DECISION_KINDS:
            raise ValueError(f"{self.key}: kind must be one of {DECISION_KINDS}")
        if self.confidence_kind not in CONFIDENCE_KINDS:
            raise ValueError(f"{self.key}: confidence_kind must be one of {CONFIDENCE_KINDS}")

    def decide(
        self,
        answer: str | float,
        scores: dict[str, float],
        confidence: float | None,
        state: dict[str, Any],
        rationale: list[str],
        ev: list[dict[str, Any]],
        entity_type: str,
        entity: str,
        as_of_key: str,
        question: str | None = None,
        answer_interval: list[float | None] | None = None,
    ) -> dict[str, Any]:
        if self.kind != "score" and answer not in self.options:
            raise ValueError(f"{self.key}: answer {answer!r} is not one of {self.options}")
        return decision(
            self.key,
            self.policy_version,
            question or self.question,
            self.kind,
            list(self.options),
            answer,
            scores,
            confidence,
            self.confidence_kind,
            state,
            rationale,
            ev,
            entity_type,
            entity,
            as_of_key,
            scale=self.scale,
            answer_interval=answer_interval,
        )

    def abstain(
        self,
        reason: str,
        state: dict[str, Any],
        entity_type: str,
        entity: str,
        as_of_key: str,
        question: str | None = None,
        rationale: list[str] | None = None,
        ev: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return decision(
            self.key,
            self.policy_version,
            question or self.question,
            self.kind,
            list(self.options),
            None,
            {},
            None,
            self.confidence_kind,
            state,
            rationale or [],
            ev or [],
            entity_type,
            entity,
            as_of_key,
            fallback=reason,
            scale=self.scale,
        )
