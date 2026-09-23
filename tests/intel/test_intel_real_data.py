"""Smoke test on the real processed MovieLens data (skipped when it is absent)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from jev_ml.intel import load_default_inputs, run_pipeline, run_scenario

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(
    not (PROCESSED / "interactions.csv").exists(), reason="no processed MovieLens data in data/processed"
)


@pytest.fixture(scope="module")
def real_inputs():
    # explicit paths: tests/conftest.py points JEV_MODELS_DIR/EXPERIMENTS_DIR at a temp dir
    return load_default_inputs(
        processed_dir=PROCESSED, models_dir=ROOT / "models", experiments_dir=ROOT / "experiments"
    )


def test_real_pipeline_runs_fast_and_is_plausible(real_inputs):
    t0 = time.perf_counter()
    res = run_pipeline(real_inputs)
    assert time.perf_counter() - t0 < 10.0
    d = res.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["data"]["sources"][0]["rows"] == len(real_inputs.interactions)
    assert d["data"]["quality"]["score"] == 1.0
    assert d["summary"]["counts"]["warnings"] <= 10  # not spammy
    assert d["predictions"]["lapse"]["status"] == "ok"
    assert d["predictions"]["lapse"]["metrics"]["auc"] > d["predictions"]["lapse"]["metrics"]["baseline_auc"]
    batches = {b["name"]: b for b in d["decision_batches"]}
    assert set(batches) == {"model_governance", "genre_programming", "audience"}
    assert all(b["status"] == "ok" for b in batches.values())
    assert all(x["batch_id"] in {b["id"] for b in batches.values()} for x in d["decisions"])
    slots = [x for x in d["decisions"] if x["key"] == "editorial_slot_share"]
    assert slots and any(not x["abstained"] for x in slots)
    for x in slots:
        if not x["abstained"]:
            lo, hi = x["answer_interval"]
            assert 0 <= lo <= hi <= 100 and 0 <= x["answer"] <= 100 and x["confidence_kind"] == "interval"
    out = run_scenario(res, {"series_id": "volume:all", "horizon_months": 12})
    assert len(out["baseline"]["points"]) == 12


def test_real_replay_2017(real_inputs):
    from dataclasses import replace
    from datetime import UTC, datetime

    d = run_pipeline(replace(real_inputs, as_of=datetime(2017, 7, 1, tzinfo=UTC))).to_dict()
    assert d["run"]["as_of"] == "2017-07-01T00:00:00Z"
    assert all(p["t"] <= "2017-06-01" for s in d["series"] for p in s["points"])
    if real_inputs.model_manifest:
        dec = {x["key"]: x for x in d["decisions"]}
        assert dec["retrain_model"]["abstained"]
