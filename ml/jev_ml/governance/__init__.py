"""Retraining and model governance (Phase 2, WS2; docs/RETRAINING_AND_MODEL_GOVERNANCE.md).

* ``snapshot``  versioned, content-hashed training snapshots: MovieLens plus app feedback
* ``retrain``   train a snapshot into a registered ``candidate`` with full lineage
* ``gates``     the promotion gate: candidate vs the active model on the same frozen split
* ``promotion`` pure promotion rules (is a gate result still valid for the current incumbent?)

This package never touches the database: the API (``jev_api.services.governance``) reads app
feedback into a DataFrame, and persists jobs, snapshots and gate results.
"""

from __future__ import annotations

from jev_ml.governance.gates import GateConfig, evaluate_candidate
from jev_ml.governance.promotion import promotion_blockers
from jev_ml.governance.retrain import config_hash, train_candidate
from jev_ml.governance.snapshot import APP_EVENT_COLUMNS, SEMANTICS, build_snapshot, load_snapshot_manifest

__all__ = [
    "APP_EVENT_COLUMNS",
    "SEMANTICS",
    "GateConfig",
    "build_snapshot",
    "config_hash",
    "evaluate_candidate",
    "load_snapshot_manifest",
    "promotion_blockers",
    "train_candidate",
]
