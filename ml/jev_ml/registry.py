"""File-based model registry: models/<version>/manifest.json + models/registry.json (active pointer).

The API mirrors this into the ModelVersion / Experiment / EvaluationMetric tables on startup; the
files remain the source of truth for artifacts so that training never needs a database.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from jev_ml.paths import MODELS_DIR

REGISTRY_FILE = "registry.json"


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
        return {"active": None, "versions": []}
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def register_version(manifest: dict[str, Any], models_dir: Path = MODELS_DIR, activate: bool = True) -> None:
    reg = read_registry(models_dir)
    version = manifest["version"]
    if version not in reg["versions"]:
        reg["versions"].append(version)
    if activate or reg["active"] is None:
        reg["active"] = version
    _atomic_write(models_dir / REGISTRY_FILE, reg)


def set_active(version: str, models_dir: Path = MODELS_DIR) -> None:
    reg = read_registry(models_dir)
    if not (models_dir / version / "manifest.json").exists():
        raise FileNotFoundError(f"unknown model version {version}")
    if version not in reg["versions"]:
        reg["versions"].append(version)
    reg["active"] = version
    _atomic_write(models_dir / REGISTRY_FILE, reg)


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
