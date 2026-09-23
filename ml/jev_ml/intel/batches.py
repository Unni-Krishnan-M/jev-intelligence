"""Multi-question decision calls: several bounded questions answered against ONE shared state.

A batch takes a snapshot of every input its questions may read (``state``), deep-copies it once so
upstream objects can never be changed by a policy, fingerprints it (``state_hash``: sha1 over a
canonical encoding; DataFrames and arrays are hashed by value), and runs each question's policy on
that same snapshot. Atomicity:

* all questions see the identical snapshot (one copy, hashed before the first question runs);
* after the last question the snapshot is hashed again. If it changed (a policy mutated shared
  state) or any policy raised, *none* of the batch's answers is kept: every question records an
  abstention with the failure as ``fallback_reason`` and the batch reports ``status: "failed"``;
* otherwise every decision produced gets the batch's ``batch_id``. A decision may still abstain on
  its own evidence (for example insufficient data); that is the question's answer, not a failure.

Ids are stable: ``batch_id = stable_id("batch", name, as_of)``, like decision ids.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.intel.common import stable_id

log = logging.getLogger(__name__)

Policy = Callable[[Mapping[str, Any]], list[dict[str, Any]]]


@dataclass(frozen=True)
class Question:
    """One bounded question of a batch.

    ``policy`` maps the shared snapshot to the decisions it answers (one per entity). ``entities``
    lists the (entity_type, entity) pairs the question covers, used only to record abstentions when
    the batch fails."""

    key: str
    policy_version: str
    kind: str
    confidence_kind: str
    question: str
    policy: Policy
    entities: Callable[[Mapping[str, Any]], list[tuple[str, str]]]


def _canon(obj: Any) -> Any:
    """Canonical, JSON-encodable form used for hashing (not for output)."""
    if isinstance(obj, pd.DataFrame):
        h = pd.util.hash_pandas_object(obj, index=True).to_numpy()
        return {"__df__": [list(map(str, obj.columns)), hashlib.sha1(h.tobytes()).hexdigest()]}  # noqa: S324
    if isinstance(obj, pd.Series):
        h = pd.util.hash_pandas_object(obj, index=True).to_numpy()
        return {"__series__": hashlib.sha1(h.tobytes()).hexdigest()}  # noqa: S324
    if isinstance(obj, np.ndarray):
        return {"__nd__": [str(obj.dtype), list(obj.shape), hashlib.sha1(obj.tobytes()).hexdigest()]}  # noqa: S324
    if isinstance(obj, Mapping):
        return {str(k): _canon(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, list | tuple | set | frozenset):
        items = sorted(obj, key=str) if isinstance(obj, set | frozenset) else obj
        return [_canon(v) for v in items]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return repr(float(obj))  # exact, and NaN-safe
    if obj is None or isinstance(obj, str):
        return obj
    return repr(obj)


def state_hash(state: Mapping[str, Any]) -> str:
    """sha1 of the canonical encoding of a state snapshot (same numbers -> same hash)."""
    raw = json.dumps(_canon(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode()).hexdigest()  # noqa: S324 (fingerprint, not security)


def _abstention(q: Question, entity_type: str, entity: str, reason: str, as_of_key: str) -> dict[str, Any]:
    return {
        "id": stable_id("dec", q.key, entity, as_of_key),
        "key": q.key,
        "spec_id": q.key,
        "policy_version": q.policy_version,
        "question": q.question,
        "kind": q.kind,
        "options": [],
        "answer": None,
        "option_scores": {},
        "confidence": None,
        "confidence_kind": q.confidence_kind,
        "state": {},
        "rationale": [],
        "evidence": [],
        "abstained": True,
        "fallback_reason": reason,
        "entity_type": entity_type,
        "entity": entity,
        "scale": None,
        "answer_interval": None,
        "batch_id": None,
    }


def run_batch(
    name: str,
    question: str,
    state: Mapping[str, Any],
    questions: list[Question],
    as_of_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Answer ``questions`` against one snapshot of ``state``; returns (decisions, batch record)."""
    batch_id = stable_id("batch", name, as_of_key)
    snapshot: Mapping[str, Any] = copy.deepcopy(dict(state))
    h0 = state_hash(snapshot)
    decisions: list[dict[str, Any]] = []
    failure: str | None = None
    for q in questions:
        try:
            produced = q.policy(snapshot)
        except Exception as exc:  # any policy failure fails the whole batch
            failure = f"batch {name} failed in {q.key}: {type(exc).__name__}: {exc}"
            log.warning("decision %s", failure, exc_info=True)
            break
        bad = [d.get("key") for d in produced if d.get("key") != q.key]
        if bad:
            failure = f"batch {name}: policy {q.key} produced decisions for {sorted(set(map(str, bad)))}"
            break
        decisions.extend(produced)
    if failure is None and state_hash(snapshot) != h0:
        failure = f"batch {name}: a policy changed the shared state snapshot"
    if failure is not None:
        decisions = []
        for q in questions:
            try:
                ents = q.entities(snapshot)
            except Exception:  # fall back to one platform-level abstention
                log.warning("batch %s: entities of %s failed", name, q.key, exc_info=True)
                ents = []
            for et, e in ents or [("platform", "all")]:
                decisions.append(_abstention(q, et, e, failure, as_of_key))
    for d in decisions:
        d["batch_id"] = batch_id
    record = {
        "id": batch_id,
        "name": name,
        "question": question,
        "keys": [q.key for q in questions],
        "decision_ids": [d["id"] for d in decisions],
        "state_hash": h0,
        "state_keys": sorted(snapshot),
        "policy_versions": {q.key: q.policy_version for q in questions},
        "n_decisions": len(decisions),
        "n_abstained": sum(bool(d["abstained"]) for d in decisions),
        "status": "failed" if failure else "ok",
        "failure_reason": failure,
    }
    return decisions, record
