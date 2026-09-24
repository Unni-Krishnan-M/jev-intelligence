"""File-based model registry: models/<version>/manifest.json + models/registry.json (active pointer,
lifecycle states and activation history).

The API mirrors this into the ModelVersion / ModelGovernance tables; the files remain the source of truth
for artifacts and for which version serves, so training never needs a database.

Lifecycle (docs/RETRAINING_AND_MODEL_GOVERNANCE.md):

    candidate --gate fails--> rejected
    candidate --promote-----> active --superseded or rolled back--> retired

Registering a version does NOT activate it: ``register_version(..., activate=False)`` is the default
and records a ``candidate``. Only an explicit ``activate=True`` (the bootstrap model, or an operator
decision) or ``set_active`` makes a version serve.

registry.json keeps its v1 keys (``active``, ``versions``) so older readers keep working, and adds
``states`` (version -> state), ``previous`` (the version that served before the active one) and
``history`` (every activation, promotion and rollback, oldest first). A v1 file without ``states`` is
read as: the active version is ``active``, every other version ``retired``.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jev_ml.paths import MODELS_DIR

REGISTRY_FILE = "registry.json"
MODEL_STATES = ("candidate", "active", "retired", "rejected")
# how many history entries registry.json keeps (the DB audit log keeps everything)
MAX_HISTORY = 200


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
    # mkstemp creates 0600; the API and trainer may run as different users, so make it world-readable
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def read_registry(models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    path = models_dir / REGISTRY_FILE
    if not path.exists():
        return {"active": None, "versions": [], "states": {}, "previous": None, "history": []}
    data: dict[str, Any] = json.loads(path.read_text())
    data.setdefault("active", None)
    data.setdefault("versions", [])
    data.setdefault("previous", None)
    data.setdefault("history", [])
    states: dict[str, str] = data.setdefault("states", {})
    for v in data["versions"]:  # v1 registries: infer the state of versions registered before states
        states.setdefault(v, "active" if v == data["active"] else "retired")
    return data


def _history(reg: dict[str, Any], action: str, version: str, **extra: Any) -> None:
    reg["history"].append({"at": _now(), "action": action, "version": version, **extra})
    del reg["history"][:-MAX_HISTORY]


def _activate(reg: dict[str, Any], version: str, action: str, actor: str | None) -> None:
    old = reg["active"]
    if old == version:
        return
    if old is not None:
        reg["states"][old] = "retired"
    reg["states"][version] = "active"
    reg["previous"] = old
    reg["active"] = version
    _history(reg, action, version, previous=old, actor=actor)


def register_version(manifest: dict[str, Any], models_dir: Path = MODELS_DIR, activate: bool = False) -> None:
    """Record a trained version. It becomes a ``candidate``; only ``activate=True`` makes it serve
    (the bootstrap model of an empty registry, or an explicit operator decision)."""
    reg = read_registry(models_dir)
    version = manifest["version"]
    if version not in reg["versions"]:
        reg["versions"].append(version)
        reg["states"][version] = "candidate"
        _history(reg, "register", version)
    if activate:
        _activate(reg, version, "activate", None)
    _atomic_write(models_dir / REGISTRY_FILE, reg)


def set_active(
    version: str, models_dir: Path = MODELS_DIR, action: str = "activate", actor: str | None = None
) -> None:
    """Make ``version`` the serving model; the previous one becomes ``retired`` and ``previous``.

    ``action`` is recorded in the history (activate | promote | rollback). The caller is responsible
    for the policy (gate, audit); the registry only refuses versions without artifacts."""
    reg = read_registry(models_dir)
    if not (models_dir / version / "manifest.json").exists():
        raise FileNotFoundError(f"unknown model version {version}")
    if version not in reg["versions"]:
        reg["versions"].append(version)
    _activate(reg, version, action, actor)
    if action == "rollback":
        # rolling back to the previous version: the new "previous" is whatever served before *it*,
        # so a second rollback walks further back instead of toggling between two versions
        reg["previous"] = _served_before(reg, version, exclude=reg["previous"])
    _atomic_write(models_dir / REGISTRY_FILE, reg)


def _served_before(reg: dict[str, Any], version: str, exclude: str | None) -> str | None:
    """The version that was active right before ``version`` was last made active by a non-rollback
    action, if it still exists and is not ``exclude``."""
    for entry in reversed(reg["history"]):
        if entry.get("version") == version and entry.get("action") in ("activate", "promote"):
            prev = entry.get("previous")
            if prev and prev != exclude and prev in reg["versions"]:
                return str(prev)
            return None
    return None


def set_state(version: str, state: str, models_dir: Path = MODELS_DIR, reason: str | None = None) -> None:
    """Set a non-serving state (candidate | rejected | retired). The active version cannot be moved
    out of ``active`` here: promote or roll back another version instead."""
    if state not in MODEL_STATES or state == "active":
        raise ValueError(f"set_state cannot set {state!r}; use set_active")
    reg = read_registry(models_dir)
    if version not in reg["versions"]:
        raise FileNotFoundError(f"unknown model version {version}")
    if reg["active"] == version:
        raise ValueError(f"{version} is the active model")
    reg["states"][version] = state
    _history(reg, state, version, reason=reason)
    _atomic_write(models_dir / REGISTRY_FILE, reg)


def version_state(version: str, models_dir: Path = MODELS_DIR) -> str | None:
    state: str | None = read_registry(models_dir)["states"].get(version)
    return state


def rollback_target(models_dir: Path = MODELS_DIR) -> str | None:
    """The version a rollback restores: the one that served before the active version."""
    reg = read_registry(models_dir)
    prev: str | None = reg.get("previous")
    if prev and prev != reg["active"] and (models_dir / prev / "manifest.json").exists():
        return prev
    return None


def load_manifest(version: str, models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((models_dir / version / "manifest.json").read_text())
    return data


def list_manifests(models_dir: Path = MODELS_DIR) -> list[dict[str, Any]]:
    out = []
    for mf in sorted(models_dir.glob("*/manifest.json")):
        out.append(json.loads(mf.read_text()))
    return sorted(out, key=lambda m: m["created_at"])


def active_version(models_dir: Path = MODELS_DIR) -> str | None:
    active: str | None = read_registry(models_dir).get("active")
    return active
