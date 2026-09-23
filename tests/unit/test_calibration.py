"""Recommendation confidence (contract 9.2): isotonic calibration fitted on validation only."""

from __future__ import annotations

import itertools
import json
import shutil

import numpy as np
import pytest
from conftest import MODEL_PARAMS

from jev_ml.calibration import (
    CALIBRATION_FILE,
    Calibrator,
    calibrate,
    calibration_summary,
    ece,
    ece_quantile,
    next_ratings,
    stratum_bounds,
)
from jev_ml.engine import Interaction, RecommendationEngine
from jev_ml.evaluation.split import make_split

TRAINING_CONFIG = {
    "split": {"strategy": "user_temporal", "val_frac": 0.1, "test_frac": 0.2},
    "models": MODEL_PARAMS,
    "evaluation": {"relevance_threshold": 4.0, "cold_start_profile_size": 3},
}


def _run(movies, inter):
    # the tiny users hold ~1 validation row each, so the label horizon is 1 rating
    return calibrate(movies, inter, TRAINING_CONFIG, {}, 42, "m-test", k=10, horizon=1, strata=(3, None))


@pytest.fixture(scope="module")
def cal(tiny_data):
    movies, inter = tiny_data
    return _run(movies, inter)


def test_calibration_payload_contract(cal):
    assert cal["calibration_version"].startswith("cal-1.0.0-")
    assert cal["confidence_kind"] == "probability" and cal["model_version"] == "m-test"
    assert cal["fitted_on"]["split"] == "validation" and cal["fitted_on"]["test_rows_used_for_fit"] == 0
    assert [s["name"] for s in cal["strata"]] == ["profile_3", "profile_full"]
    # strata tile the profile sizes without gaps; the last one is open-ended
    rngs = [s["applies_to"] for s in cal["strata"]]
    assert rngs[0]["min"] == 0 and rngs[-1]["max"] is None
    assert all(a["max"] + 1 == b["min"] for a, b in itertools.pairwise(rngs))
    for s in cal["strata"]:
        assert s["feature"] in ("score", "rank")
        for split in ("validation", "test"):
            m = s[split]
            assert 0 <= m["ece"] <= 1 and 0 <= m["brier"] <= 1 and m["n"] > 0
    assert cal["assumptions"] and cal["headline"]["split"] == "test"
    json.dumps(cal, allow_nan=False)
    summary = calibration_summary(cal)
    assert "mapping" not in json.dumps(summary)
    assert summary["strata"][0]["test"]["ece"] == cal["strata"][0]["test"]["ece"]


def test_calibrator_monotone_and_bounded(cal):
    c = Calibrator.from_dict(cal)
    for st in c.strata:
        grid = np.linspace(st.x.min() - 1, st.x.max() + 1, 200)
        p = np.interp(grid, st.x, st.y)
        assert np.all((p >= 0) & (p <= 1))
        d = np.diff(np.interp(np.arange(1, 11), st.x, st.y)) if st.feature == "rank" else np.diff(p)
        assert np.all(d <= 1e-12) if st.feature == "rank" else np.all(d >= -1e-12)
    for n in (0, 3, 50, 10_000):
        v = c.predict(0.5, 1, n)
        assert v is not None and 0 <= v <= 1
    assert c.predict(0.5, 11, 5) is None, "no extrapolation beyond the calibrated rank"
    assert c.predict(float("nan"), 1, 5) is None


def test_calibrator_rejects_invalid_mapping(cal):
    bad = json.loads(json.dumps(cal))
    m = bad["strata"][0]["mapping"]
    m["y"] = [1.5] * len(m["y"])
    with pytest.raises(ValueError):
        Calibrator.from_dict(bad)
    flipped = json.loads(json.dumps(cal))
    fm = flipped["strata"][0]["mapping"]
    if len(fm["y"]) > 1 and fm["y"][0] != fm["y"][-1]:
        fm["y"] = fm["y"][::-1]
        with pytest.raises(ValueError):
            Calibrator.from_dict(flipped)


def test_fit_uses_no_test_data(tiny_data, cal):
    """Scrambling every test-split rating must leave the fitted calibrator untouched."""
    movies, inter = tiny_data
    split = make_split(inter, **TRAINING_CONFIG["split"])
    key = ["user_id", "movie_id", "timestamp"]
    test_keys = set(map(tuple, split.test[key].to_numpy().tolist()))
    scrambled = inter.copy()
    is_test = np.array([tuple(r) in test_keys for r in scrambled[key].to_numpy().tolist()])
    assert is_test.sum() == len(split.test)
    scrambled.loc[is_test, "rating"] = 6.0 - scrambled.loc[is_test, "rating"]  # invert likes/dislikes
    again = _run(movies, scrambled)
    for a, b in zip(cal["strata"], again["strata"], strict=True):
        assert a["mapping"] == b["mapping"] and a["validation"] == b["validation"]
        assert a["applies_to"] == b["applies_to"]
    assert again["fitted_on"] == cal["fitted_on"]
    assert again["calibration_version"] == cal["calibration_version"]
    # ...while the test evaluation does see the change
    assert any(a["test"] != b["test"] for a, b in zip(cal["strata"], again["strata"], strict=True))


def test_calibration_is_deterministic(tiny_data, cal):
    again = _run(*tiny_data)
    assert again["strata"] == cal["strata"]


def test_metrics_helpers():
    y = np.array([0, 0, 1, 1], dtype=float)
    assert ece(y, y) == 0.0 and ece_quantile(y, y) == 0.0
    assert ece(np.full(4, 0.5), y) == pytest.approx(0.0)
    assert ece(np.full(4, 0.9), np.zeros(4)) == pytest.approx(0.9)
    assert stratum_bounds([0.0, 3.0, 10.0, 70.0]) == [(0, 1), (2, 5), (6, 26), (27, None)]


def test_next_ratings_takes_first_rows_in_time():
    import pandas as pd

    t = pd.DataFrame(
        {"user_id": [1, 1, 1, 2], "movie_id": [5, 6, 7, 8], "rating": [4, 4, 4, 4], "timestamp": [3, 1, 2, 1]}
    )
    out = next_ratings(t, 2)
    assert out["movie_id"].tolist() == [6, 7]  # user 2 has < 2 rows and is dropped


@pytest.fixture()
def model_copy(tiny_model_dir, tmp_path):
    dst = tmp_path / tiny_model_dir.name
    shutil.copytree(tiny_model_dir, dst)
    return dst


def _profile_events():
    return [Interaction(1, "rating", 5.0), Interaction(2, "rating", 4.5), Interaction(3, "favorite")]


def test_engine_without_calibration_has_no_confidence(model_copy):
    assert not (model_copy / CALIBRATION_FILE).exists()
    e = RecommendationEngine(model_copy)
    assert e.calibration is None and e.health()["calibration"] is None
    recs = e.recommend(e.build_profile(_profile_events()), k=5)
    assert recs and all(r.confidence is None and r.confidence_kind is None for r in recs)


def test_engine_attaches_calibrated_confidence(model_copy, cal):
    e0 = RecommendationEngine(model_copy)
    payload = {**cal, "model_version": e0.version}
    (model_copy / CALIBRATION_FILE).write_text(json.dumps(payload))
    e = RecommendationEngine(model_copy)
    assert e.calibration is not None and e.calibration["calibration_version"] == cal["calibration_version"]
    assert e.health()["calibration"] == e.calibration
    profile = e.build_profile(_profile_events())
    recs = e.recommend(profile, k=10)
    c = Calibrator.from_dict(payload)
    for r in recs:
        assert r.confidence_kind == "probability" and 0 <= r.confidence <= 1
        assert r.confidence == pytest.approx(c.predict(r.score, r.rank, profile.n_interactions), abs=1e-4)
    # beyond the calibrated rank range there is no confidence (never guessed)
    page2 = e.recommend(profile, k=5, offset=10)
    assert all(r.confidence is None and r.confidence_kind is None for r in page2)
    # identical rankings with and without calibration
    assert [r.movie_id for r in recs] == [r.movie_id for r in e0.recommend(profile, k=10)]


def test_engine_ignores_calibration_of_another_model(model_copy, cal):
    (model_copy / CALIBRATION_FILE).write_text(json.dumps({**cal, "model_version": "someone-else"}))
    e = RecommendationEngine(model_copy)
    assert e.calibration is None
    assert all(r.confidence is None for r in e.recommend(e.build_profile(_profile_events()), k=3))


def test_engine_survives_corrupt_calibration(model_copy):
    (model_copy / CALIBRATION_FILE).write_text("{not json")
    e = RecommendationEngine(model_copy)
    assert e.calibration is None
