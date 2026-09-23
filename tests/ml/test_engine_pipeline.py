import json

import numpy as np

from jev_ml.engine import Interaction, RecommendationEngine


def test_pipeline_writes_experiment_and_model(pipeline_result):
    r = pipeline_result
    assert set(r["metrics"]["test"]) == {"random", "popularity", "content", "itemknn", "als", "hybrid"}
    for model, m in r["metrics"]["test"].items():
        for k in (5, 10, 20):
            for metric in ("precision", "recall", "f1", "ndcg", "map", "hit_rate", "coverage"):
                v = m[f"{metric}@{k}"]
                assert 0.0 <= v <= 1.0 and np.isfinite(v), (model, metric, k)
        assert np.isfinite(m["novelty@10"]) and 0 <= m["diversity@10"] <= 1
    # the synthetic data has planted genre structure: personalised models must beat random
    test = r["metrics"]["test"]
    assert test["hybrid"]["ndcg@10"] > test["random"]["ndcg@10"]
    assert test["itemknn"]["ndcg@10"] > test["random"]["ndcg@10"]
    assert r["model_version"]


def test_pipeline_is_reproducible(processed_dir, pipeline_result):
    from jev_ml.training import run_pipeline

    again = run_pipeline(quick=True, processed_dir=processed_dir, activate=False, train_production=False)
    for model in ("hybrid", "als", "itemknn", "popularity", "content"):
        assert again["metrics"]["test"][model] == pipeline_result["metrics"]["test"][model], model


def test_engine_loads_and_serves(tiny_model_dir):
    e = RecommendationEngine(tiny_model_dir)
    h = e.health()
    assert h["n_items"] == len(e.movies)
    events = [
        Interaction(1, "rating", 5.0),
        Interaction(2, "rating", 4.5),
        Interaction(3, "favorite"),
        Interaction(4, "watch"),
        Interaction(40, "rating", 1.0),
    ]
    profile = e.build_profile(events, genre_prefs=["Sci-Fi"])
    recs = e.recommend(profile, k=10)
    ids = [r.movie_id for r in recs]
    valid = set(e.movies["movie_id"].tolist())
    assert len(recs) == 10 and len(set(ids)) == 10
    assert set(ids) <= valid
    assert not set(ids) & {1, 2, 3, 4, 40}
    assert all(np.isfinite(r.score) for r in recs)
    assert all(r.reason for r in recs)
    assert [r.rank for r in recs] == list(range(1, 11))
    for r in recs:
        assert set(r.signals) == {"content", "collaborative", "latent", "popularity", "preference", "recency"}
    # manifest records reproducibility metadata
    m = json.loads((tiny_model_dir / "manifest.json").read_text())
    for key in (
        "version",
        "created_at",
        "dataset_version",
        "training_seed",
        "artifact_files",
        "hybrid_config",
    ):
        assert key in m


def test_engine_cold_start_and_similar(tiny_model_dir):
    e = RecommendationEngine(tiny_model_dir)
    cold = e.recommend(e.build_profile([]), k=5)
    assert len(cold) == 5
    genre_only = e.recommend(e.build_profile([], genre_prefs=["Horror"]), k=5)
    assert all("Horror" in e.movies.set_index("movie_id").loc[r.movie_id, "genres"] for r in genre_only)
    sims = e.similar(1, 5)
    assert len(sims) == 5 and all(s["movie_id"] != 1 for s in sims)
    new = e.similar_to_metadata(
        {"title": "Unseen", "genres": "Comedy", "directors": "Lou Punch", "year": 2030}, 5
    )
    assert new and all(0 < s["score"] <= 1.0001 for s in new)


def test_unknown_movies_in_history_are_ignored(tiny_model_dir):
    e = RecommendationEngine(tiny_model_dir)
    p = e.build_profile([Interaction(999_999, "rating", 5.0), Interaction(1, "rating", 5.0)])
    assert p.n_interactions == 1
