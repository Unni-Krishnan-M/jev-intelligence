"""Checks against the real trained MovieLens model (skipped when it has not been trained yet)."""

import json
from pathlib import Path

import numpy as np
import pytest

from jev_ml.engine import Interaction, RecommendationEngine

REAL_MODELS = Path(__file__).resolve().parents[2] / "models"
reg = REAL_MODELS / "registry.json"
active = json.loads(reg.read_text()).get("active") if reg.exists() else None
pytestmark = pytest.mark.skipif(
    not active or not (REAL_MODELS / str(active)).exists(), reason="no trained MovieLens model in models/"
)


@pytest.fixture(scope="module")
def engine() -> RecommendationEngine:
    return RecommendationEngine(REAL_MODELS / str(active))


def test_real_model_personalizes(engine):
    nolan = engine.build_profile(
        [Interaction(m, "rating", 5.0) for m in (79132, 109487, 58559)]
    )  # Inception, Interstellar, Dark Knight
    kids = engine.build_profile(
        [Interaction(m, "rating", 5.0) for m in (1, 2355, 6377)]
    )  # Toy Story, A Bug's Life, Finding Nemo
    a = [r.movie_id for r in engine.recommend(nolan, k=20)]
    b = [r.movie_id for r in engine.recommend(kids, k=20)]
    assert len(set(a) & set(b)) <= 4, "different tastes must yield different lists"
    assert not set(a) & {79132, 109487, 58559}
    genres = engine.movies.set_index("movie_id")["genres"]
    assert sum("Animation" in genres[m] or "Children" in genres[m] for m in b[:10]) >= 6


def test_real_model_scores_and_ids_valid(engine):
    recs = engine.recommend(engine.build_profile([Interaction(2571, "rating", 4.5)]), k=50)
    assert len(recs) == 50
    assert all(np.isfinite(r.score) for r in recs)
    assert {r.movie_id for r in recs} <= set(engine.movies["movie_id"].tolist())


def test_real_explanations_are_grounded(engine):
    p = engine.build_profile([Interaction(m, "rating", 5.0) for m in (79132, 109487)])
    titles = {79132: "Inception", 109487: "Interstellar"}
    for r in engine.recommend(p, k=10):
        for mid in r.anchor_movie_ids:
            assert mid in titles
        if "Because you liked" in r.reason or "which you rated highly" in r.reason:
            assert any(t in r.reason for t in titles.values())


def test_real_calibrated_confidence(engine):
    """calibration.json is written by scripts/calibrate_recommendations.py (validation-fitted)."""
    if engine.calibration is None:
        pytest.skip("active model has no calibration.json")
    cal = engine.calibration
    assert cal["model_version"] == engine.version and cal["fitted_on"]["split"] == "validation"
    assert cal["fitted_on"]["test_rows_used_for_fit"] == 0
    warm = engine.build_profile(
        [Interaction(m, "rating", 5.0) for m in range(1, 200) if engine.item_index.index_of(m) is not None][
            :40
        ]
    )
    for profile in (engine.build_profile([]), engine.build_profile([Interaction(2571, "rating", 4.5)]), warm):
        recs = engine.recommend(profile, k=20)
        assert all(r.confidence_kind == "probability" and 0.0 <= r.confidence <= 1.0 for r in recs)
    # calibrated probabilities stay near the measured hit rates (a few %), never a raw score
    assert max(r.confidence for r in engine.recommend(warm, k=20)) < 0.25
    assert engine.health()["calibration"]["headline"]["split"] == "test"
