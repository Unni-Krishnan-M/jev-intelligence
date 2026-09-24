"""A movie replay reads the model active at replay time and records which one; it can be pinned."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev_ml.domains.movie.ingest import load_default_inputs


def _models(tmp: Path) -> Path:
    md = tmp / "models"
    for v in ("jev-a", "jev-b"):
        (md / v).mkdir(parents=True)
        (md / v / "manifest.json").write_text(json.dumps({"version": v}))
    (md / "registry.json").write_text(json.dumps({"active": "jev-a"}))
    return md


def test_replay_uses_active_model_unless_pinned(processed_dir: Path, tmp_path: Path):
    md = _models(tmp_path)
    kw = {"processed_dir": processed_dir, "models_dir": md, "experiments_dir": tmp_path / "exp"}
    assert load_default_inputs(**kw).model_manifest["version"] == "jev-a"
    # a promotion between two replays changes the manifest a replay reads ...
    (md / "registry.json").write_text(json.dumps({"active": "jev-b"}))
    assert load_default_inputs(**kw).model_manifest["version"] == "jev-b"
    # ... unless the replay is pinned to the version the first run recorded
    assert load_default_inputs(**kw, model_version="jev-a").model_manifest["version"] == "jev-a"
    with pytest.raises(ValueError):
        load_default_inputs(**kw, model_version="../jev-a")


def test_run_records_the_model_version(processed_dir: Path, tmp_path: Path):
    from datetime import UTC, datetime

    from jev_ml.core import run_domain
    from jev_ml.domains.movie.adapter import MovieAdapter

    md = _models(tmp_path)
    inputs = load_default_inputs(processed_dir=processed_dir, models_dir=md, experiments_dir=tmp_path)
    res = run_domain(MovieAdapter(inputs=inputs), now=datetime(2026, 9, 24, tzinfo=UTC))
    assert res.data["run"]["model_version"] == "jev-a"
