"""Train a snapshot into a registered ``candidate`` with full lineage (never activates).

Uses only the training public API (``run_pipeline``): a snapshot directory is shaped like
``data/processed``, so it is passed as ``processed_dir``. The lineage is written next to the
manifest as ``models/<version>/lineage.json`` (the manifest itself stays as training wrote it):

    snapshot id + content hash, config path + hash, git commit, seed, quick flag, trigger, job id,
    the retrain decision id (when the JEV ``retrain_model`` decision triggered the job), the
    experiment run, training metrics and timings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from jev_ml.governance.snapshot import load_snapshot_manifest
from jev_ml.paths import CONFIGS_DIR, EXPERIMENTS_DIR, MODELS_DIR
from jev_ml.registry import load_manifest, read_registry, register_version

log = logging.getLogger(__name__)

LINEAGE_FILE = "lineage.json"


def config_hash(config_path: Path | None) -> str:
    """SHA-256 (first 16 hex) of the parsed experiment config, key-order independent."""
    path = config_path or CONFIGS_DIR / "experiment.yaml"
    cfg = yaml.safe_load(path.read_text())
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_lineage(version: str, lineage: dict[str, Any], models_dir: Path = MODELS_DIR) -> None:
    path = models_dir / version / LINEAGE_FILE
    path.write_text(json.dumps(lineage, indent=2, sort_keys=True, default=str))
    os.chmod(path, 0o644)


def load_lineage(version: str, models_dir: Path = MODELS_DIR) -> dict[str, Any] | None:
    path = models_dir / version / LINEAGE_FILE
    if not path.exists():
        return None
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def train_candidate(
    snapshot_dir: Path,
    config_path: Path | None = None,
    quick: bool = True,
    models_dir: Path = MODELS_DIR,
    experiments_dir: Path = EXPERIMENTS_DIR,
    lineage_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train on ``snapshot_dir`` and register the result as a candidate. Returns the lineage."""
    from jev_ml.training import run_pipeline

    snap = load_snapshot_manifest(snapshot_dir)
    active_before = read_registry(models_dir)["active"]
    t0 = time.perf_counter()
    res = run_pipeline(
        config_path,
        quick=quick,
        processed_dir=snapshot_dir,
        models_dir=models_dir,
        experiments_dir=experiments_dir,
        activate=False,
    )
    version = res["model_version"]
    if not version:
        raise RuntimeError("training produced no model version")
    # run_pipeline(activate=False) registers without activating. Defensive: a registry in which no
    # version was active would, in the v1 semantics, have activated this one. Never leave a
    # candidate serving because of that.
    reg = read_registry(models_dir)
    if reg["active"] == version and active_before is not None:
        raise RuntimeError(f"candidate {version} was activated by training: refusing (governance)")
    if version not in reg["versions"]:
        register_version(load_manifest(version, models_dir), models_dir, activate=False)
    manifest = load_manifest(version, models_dir)
    lineage = {
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "snapshot_id": snap["snapshot_id"],
        "snapshot_content_hash": snap["content_hash"],
        "snapshot_cutoff": snap.get("cutoff"),
        "snapshot_row_counts": snap.get("row_counts"),
        "snapshot_dir": str(snapshot_dir),
        "dataset_version": manifest.get("dataset_version"),
        "config_path": str(config_path or CONFIGS_DIR / "experiment.yaml"),
        "config_hash": config_hash(config_path),
        "git_commit": manifest.get("git_commit"),
        "jev_ml_version": manifest.get("jev_ml_version"),
        "seed": manifest.get("training_seed"),
        "quick": quick,
        "experiment_run": res.get("run_id"),
        "split": res.get("split"),
        "metrics": {
            "test_hybrid": (res.get("metrics") or {}).get("test", {}).get("hybrid"),
            "cold_start_hybrid": (res.get("metrics") or {}).get("cold_start", {}).get("hybrid"),
        },
        "trained_on_rows": manifest.get("trained_on_rows"),
        "train_seconds": round(time.perf_counter() - t0, 2),
        "incumbent_at_training": active_before,
        **(lineage_extra or {}),
    }
    write_lineage(version, lineage, models_dir)
    log.info("registered candidate %s from %s", version, snap["snapshot_id"])
    return lineage
