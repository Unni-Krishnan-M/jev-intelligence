"""Bootstrap/paired statistics, cold-stage weights, the logistic calibrator and the evaluator
extensions used by scripts/benchmark_recommenders.py."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from conftest import MODEL_PARAMS

from jev_ml.calibration import Calibrator, LogisticMap, calibrate, logistic_design, signal_agreement
from jev_ml.evaluation.benchmark import decide
from jev_ml.evaluation.cold_start import BUCKETS, bucket_of, stage_candidates, with_stage
from jev_ml.evaluation.evaluator import Evaluator
from jev_ml.evaluation.stats import bootstrap_ci, cluster_bootstrap, holm, paired_comparison
from jev_ml.models.hybrid import SIGNALS, HybridConfig
from jev_ml.models.popularity import PopularityRecommender
from jev_ml.signals import UserProfile
from jev_ml.training import build_context, component_fn

# --- statistics ------------------------------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    x = np.random.default_rng(0).normal(1.0, 1.0, 400)
    a, b = bootstrap_ci(x, b=500, seed=3), bootstrap_ci(x, b=500, seed=3)
    assert a == b and a["lo"] < a["mean"] < a["hi"]
    assert a["hi"] - a["lo"] == pytest.approx(2 * 1.96 / np.sqrt(400), rel=0.25)
    assert bootstrap_ci([], b=10)["mean"] is None


def test_paired_comparison_detects_a_shift_and_not_noise():
    rng = np.random.default_rng(1)
    base = rng.uniform(0, 1, 300)
    same = paired_comparison(base, base, b=300, permutations=500)
    assert same["diff"] == 0 and same["p_value"] == 1.0 and not same["significant"]
    better = paired_comparison(base + 0.05 + rng.normal(0, 0.01, 300), base, b=300, permutations=500)
    assert better["significant"] and better["lo"] > 0 and better["p_value"] < 0.01 and better["wins"] > 0.9
    noise = paired_comparison(base + rng.normal(0, 0.2, 300), base, b=300, permutations=500)
    assert noise["lo"] < 0 < noise["hi"]
    with pytest.raises(ValueError):
        paired_comparison([1.0], [1.0, 2.0])


def test_holm_is_monotone_and_capped():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] == pytest.approx(0.03) and adj["b"] == pytest.approx(0.08) and adj["c"] == 0.5
    assert holm({"x": 0.6, "y": 0.7}) == {"x": 1.0, "y": 1.0}
    assert adj["a"] <= adj["b"] <= adj["c"]


def test_cluster_bootstrap_resamples_users():
    users = np.repeat(np.arange(50), 4)
    vals = np.repeat(np.random.default_rng(2).normal(size=50), 4)
    res = cluster_bootstrap(users, lambda idx: float(vals[idx].mean()), b=200)
    assert res["lo"] < res["value"] < res["hi"] and res["b"] == 200


# --- cold stages -----------------------------------------------------------------------------------


def _profile(n: int, genres: bool = True) -> UserProfile:
    events = [(i, 1.5, 1.0, float(i)) for i in range(n)]
    return UserProfile.from_events(events, genre_prefs={"Comedy": 1.0} if genres else None)


def test_cold_stage_weights_apply_within_bounds(tiny_ranker):
    import copy

    r = copy.copy(tiny_ranker)
    stages = [
        {"name": "cold_0", "min_profile": 0, "max_profile": 0, "weights": {"popularity": 1.0}},
        {"name": "cold_1_3", "min_profile": 1, "max_profile": 3, "popularity_blend": 0.5},
    ]
    r.config = HybridConfig.from_dict({**asdict(tiny_ranker.config), "cold_stages": stages})
    w0 = r.effective_weights(_profile(0))
    assert w0["popularity"] == 1.0 and r.weight_strategy(_profile(0)) == "cold_0"
    adaptive = tiny_ranker.effective_weights(_profile(2))
    w2 = r.effective_weights(_profile(2))
    assert w2["popularity"] == pytest.approx(0.5 + 0.5 * adaptive["popularity"])
    assert sum(w2.values()) == pytest.approx(1.0)
    assert r.effective_weights(_profile(8)) == tiny_ranker.effective_weights(_profile(8))
    assert r.weight_strategy(_profile(8)) == "adaptive"
    # a free weight stage drops signals that carry no evidence (no history -> no CF, no genres)
    r.config = with_stage(tiny_ranker.config, {"weights": dict.fromkeys(SIGNALS, 1.0)})
    w = r.effective_weights(_profile(0, genres=False))
    assert w["collaborative"] == w["latent"] == w["content"] == w["preference"] == 0.0
    assert sum(w.values()) == pytest.approx(1.0)


def test_cold_stage_keeps_explanations_grounded(tiny_ranker, tiny_data):
    import copy

    r = copy.copy(tiny_ranker)
    r.config = with_stage(tiny_ranker.config, {"weights": {"content": 0.5, "popularity": 0.5}})
    prof = _profile(3)
    for item in r.rank(prof, k=8):
        total = sum(s["contribution"] for s in item.signals.values())
        assert total == pytest.approx(item.score, abs=1e-9), "score = sum of real contributions"
        for kind, contribs in item.contributions.items():
            if kind in ("content", "collaborative", "latent"):
                assert all(c.item is None or c.item in set(prof.items.tolist()) for c in contribs)


def test_hybrid_config_round_trip_and_default_off():
    assert HybridConfig().cold_stages is None
    cfg = HybridConfig.from_dict({"cold_stages": [{"max_profile": 3, "popularity_blend": 0.2}]})
    again = HybridConfig.from_dict(json.loads(json.dumps(asdict(cfg))))
    assert again == cfg
    assert HybridConfig.from_dict({"weights": {"content": 0.1}}).cold_stages is None


def test_stage_candidates_and_buckets():
    cands = stage_candidates(HybridConfig(), seed=1, n_dirichlet=3, n_profile=2)
    fams = [f for f, _ in cands]
    assert fams[0] == "incumbent" and cands[0][1] is None
    assert {"popularity", "pop_blend", "stage_hybrid", "profile_content"} <= set(fams)
    for f, st in cands:
        if f == "profile_content":
            assert set(st["weights"]) <= {"content", "preference", "popularity", "recency"}
    assert stage_candidates(HybridConfig(), 1, 3, 2) == cands, "seeded"
    assert [bucket_of(n) for n in (0, 1, 3, 4, 10, 11)] == [
        "cold_0",
        "cold_1_3",
        "cold_1_3",
        "cold_4_10",
        "cold_4_10",
        "warm",
    ]
    assert [b["min_profile"] for b in BUCKETS] == [0, 1, 4]


# --- evaluator extensions ------------------------------------------------------------------------


def test_evaluator_per_user_metrics_truncation_and_exclusions(tiny_data, tiny_components):
    movies, inter = tiny_data
    ctx = build_context(movies, inter, 42)
    args = (inter, inter, ctx.item_index, ctx.user_ids, tiny_components.content.pairwise)
    pop = tiny_components.popularity.item_popularity_fraction()
    top = int(np.argmax(tiny_components.popularity.popularity))
    ev = Evaluator(*args, pop, ks=(5, 10), truncate_profile_to=0, exclude_items=np.array([top]))
    assert all(p.n_interactions == 0 for p in ev.profiles.values())
    res = ev.evaluate("pop", component_fn(tiny_components.popularity))
    assert len(res.per_user["ndcg@10"]) == len(res.user_ids) == res.n_users
    assert np.mean(res.per_user["ndcg@10"]) == pytest.approx(res.metrics["ndcg@10"])
    # the excluded film is never recommended even though it is the most popular
    rec = component_fn(tiny_components.popularity)(ev.profiles[ev.users[0]], 10)
    assert top not in rec.tolist()
    ev2 = Evaluator.from_profiles(
        {1: UserProfile(), 2: UserProfile()}, {1: {0, 1}, 2: set()}, ctx.item_index, 10, args[4], pop
    )
    assert ev2.users == [1]


def test_popularity_scoring_params_are_backward_compatible(tiny_data, tmp_path):
    movies, inter = tiny_data
    ctx = build_context(movies, inter, 42)
    default = PopularityRecommender().fit(ctx)
    likes = np.bincount(ctx.item_index.indices_of(inter["movie_id"].to_numpy()), minlength=len(movies))
    assert default.params() == {"trending_half_life_days": 365.0, "bayes_prior_votes": 10.0}
    assert np.all(np.isfinite(default.popularity)) and default.user_counts.sum() == likes.sum()
    decayed = PopularityRecommender(reach_weight=1.0, score_half_life_days=30).fit(ctx)
    decayed.save(tmp_path / "p")
    back = PopularityRecommender.load(tmp_path / "p")
    assert back.reach_weight == 1.0 and back.score_half_life_days == 30
    np.testing.assert_allclose(back.popularity, decayed.popularity)


# --- logistic calibrator -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cal_logistic(tiny_data):
    movies, inter = tiny_data
    cfg = {
        "split": {"strategy": "user_temporal", "val_frac": 0.1, "test_frac": 0.2},
        "models": MODEL_PARAMS,
        "evaluation": {"relevance_threshold": 4.0},
    }
    out = calibrate(movies, inter, cfg, {}, 42, "m-log", k=10, horizon=1, strata=(3, None), method="logistic")
    # force one stratum to serve its logistic model so the serving path is exercised
    st = out["strata"][-1]
    st["serving_method"] = "logistic"
    st["logistic"] = {
        k: st["logistic_candidate"][k] for k in ("features", "mean", "scale", "coef", "intercept")
    }
    return out


def _signals(pop: float = 2.0, norm: float = 0.7) -> dict:
    return {
        s: {"raw": pop if s == "popularity" else 0.1, "normalized": norm, "weight": 0.2, "contribution": 0.1}
        for s in SIGNALS
    }


def test_logistic_calibrator_serving_and_fallback(cal_logistic):
    c = Calibrator.from_dict(cal_logistic)
    warm = c.stratum_for(100)
    assert warm is not None and warm.logistic is not None
    p = c.predict(0.5, 1, 100, signals=_signals())
    assert p is not None and 0.0 < p < 1.0
    iso = c.predict(0.5, 1, 100)  # no signals -> the isotonic map (historical behaviour)
    assert iso == pytest.approx(float(warm.value(np.array([0.5]), np.array([1]))[0]))
    x = logistic_design([0.5], [1], [100], [2.0], [signal_agreement(_signals())])
    assert p == pytest.approx(float(warm.logistic.predict_design(x)[0]))
    assert c.predict(0.5, 11, 100, signals=_signals()) is None, "no extrapolation beyond k"
    assert signal_agreement(_signals(norm=0.2)) == 0.0 and signal_agreement(_signals()) == 1.0


def test_logistic_calibrator_rejects_bad_parameters(cal_logistic):
    bad = json.loads(json.dumps(cal_logistic))
    bad["strata"][-1]["logistic"]["scale"][0] = 0.0
    with pytest.raises(ValueError):
        Calibrator.from_dict(bad)
    bad = json.loads(json.dumps(cal_logistic))
    bad["strata"][-1]["logistic"]["features"] = ["score"]
    with pytest.raises(ValueError):
        Calibrator.from_dict(bad)
    lm = LogisticMap.from_dict(cal_logistic["strata"][-1]["logistic"])
    back = LogisticMap.from_dict(lm.to_dict())
    assert np.allclose(back.coef, lm.coef) and back.intercept == lm.intercept


def test_engine_uses_signals_for_logistic_confidence(tiny_model_dir, tmp_path, cal_logistic):
    import shutil

    from jev_ml.engine import Interaction, RecommendationEngine

    dst = tmp_path / tiny_model_dir.name
    shutil.copytree(tiny_model_dir, dst)
    e0 = RecommendationEngine(dst)
    (dst / "calibration.json").write_text(json.dumps({**cal_logistic, "model_version": e0.version}))
    e = RecommendationEngine(dst)
    events = [Interaction(m, "rating", 5.0) for m in range(1, 9)]
    prof = e.build_profile(events)
    c = Calibrator.from_dict(cal_logistic)
    for r in e.recommend(prof, k=5):
        assert r.weight_strategy == "adaptive"
        assert r.confidence == pytest.approx(
            c.predict(r.score, r.rank, prof.n_interactions, signals=r.signals), abs=1e-4
        )


# --- decision rule ----------------------------------------------------------------------------------


def _res(cold_lo: float, cold_sig: bool, auc_lo: float, brier_lo: float) -> dict:
    d = {"diff": 0.01, "lo": cold_lo, "hi": 0.02, "significant": cold_sig}
    st = {
        "name": "profile_full",
        "auc_diff_served_minus_isotonic": {"lo": auc_lo, "hi": auc_lo + 0.05},
        "brier_diff_served_minus_isotonic": {"lo": brier_lo, "hi": brier_lo + 0.001},
    }
    return {
        "protocols": {
            "user_temporal": {
                "cold_buckets": {"cold_1_3/all": {"paired": {"hybrid_cold_vs_hybrid": d}}},
                "calibration": {"strata": [st]},
            }
        }
    }


def test_decide_requires_non_inferiority_and_a_significant_gain():
    ok = decide(_res(-0.001, True, 0.01, -0.001), 0.005)
    assert ok["adopt_cold_stages"] and ok["adopt_logistic_calibrator"]
    worse = decide(_res(-0.02, True, -0.01, 0.0001), 0.005)
    assert not worse["adopt_cold_stages"] and not worse["adopt_logistic_calibrator"]
    no_gain = decide(_res(-0.001, False, -0.01, -0.001), 0.005)
    assert not no_gain["adopt_cold_stages"] and not no_gain["adopt_logistic_calibrator"]
