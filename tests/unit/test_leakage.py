"""Leakage controls: split ordering, train-only fits (popularity, IDF/tags), validation-only
calibration fits, catalogue statistics unused by the ranking, and the new-user protocol."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import MODEL_PARAMS

from jev_ml.calibration import calibrate
from jev_ml.evaluation.cold_start import HISTORY_PREFIX, new_user_cases, onboarding_genres
from jev_ml.evaluation.leakage import (
    LeakageError,
    check_split,
    future_items,
    movies_with_tags_before,
    tag_cutoffs,
)
from jev_ml.evaluation.split import Split, global_temporal_split, make_split, user_temporal_split
from jev_ml.models.content import ContentRecommender
from jev_ml.models.hybrid import HybridConfig
from jev_ml.training import build_context, fit_components


def _interactions(n_users: int = 30, per_user: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [
        (u, int(m), float(rng.choice([2.0, 4.0, 5.0])), int(rng.integers(0, 100_000)))
        for u in range(1, n_users + 1)
        for m in rng.choice(300, per_user, replace=False)
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def test_global_split_no_test_row_precedes_train_max():
    s = global_temporal_split(_interactions())
    ev = check_split(s)
    assert s.test["timestamp"].min() > s.train["timestamp"].max()
    assert s.test["timestamp"].min() > s.val["timestamp"].max()
    assert ev["train_max_ts"] < ev["val_min_ts"] and ev["val_max_ts"] < ev["test_min_ts"]


def test_check_split_detects_violations():
    s = global_temporal_split(_interactions())
    leaked = Split(
        pd.concat([s.train, s.test.head(1)], ignore_index=True), s.val, s.test.iloc[1:], s.strategy
    )
    with pytest.raises(LeakageError):
        check_split(leaked)  # a test-period row in train breaks the global order
    dup = Split(s.train, s.val, pd.concat([s.test, s.train.head(1)]), s.strategy)
    with pytest.raises(LeakageError):
        check_split(dup)  # the same (user, item) in two parts
    u = user_temporal_split(_interactions())
    assert check_split(u)["users_checked"] > 0
    early = u.test.copy()
    early.loc[early.index[0], "timestamp"] = -1  # a test row before that user's training rows
    with pytest.raises(LeakageError):
        check_split(Split(u.train, u.val, early, u.strategy))


def test_popularity_is_fitted_on_training_rows_only(tiny_data):
    movies, inter = tiny_data
    s = make_split(inter, "global_temporal")
    ctx = build_context(movies, s.train, 42)
    comps = fit_components(ctx, MODEL_PARAMS, 42)
    assert comps.popularity.user_counts.sum() == len(s.train)
    held_out_only = set(s.test["movie_id"]) - set(s.train["movie_id"])
    for mid in held_out_only:
        assert comps.popularity.user_counts[ctx.item_index.index_of(mid)] == 0
    assert comps.popularity.reference_ts == s.train["timestamp"].max()


def _tags(movies: pd.DataFrame) -> pd.DataFrame:
    first, second = int(movies["movie_id"].iloc[0]), int(movies["movie_id"].iloc[1])
    return pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "movieId": [first, second, first, second],
            "tag": ["early bird", "early bird", "zzfuture", "zzfuture"],
            "timestamp": [100, 100, 10_000, 10_000],
        }
    )


def test_tags_and_idf_use_only_tags_before_the_cut(tiny_data):
    movies, _ = tiny_data
    tags = _tags(movies)
    cut = movies_with_tags_before(movies, tags, 5_000.0)
    assert "zzfuture" not in "|".join(cut["tags"]) and "early bird" in "|".join(cut["tags"])
    content = ContentRecommender(min_df=1).fit(build_context(cut, pd.DataFrame(columns=_COLS), 42))
    names = content.featurizer.feature_names()
    assert "t:early bird" in names and not any("zzfuture" in n for n in names)
    # per-user cut: user 2's cut lies after their tag, user 1's before
    per_user = movies_with_tags_before(movies, tags, {1: 50.0, 2: 20_000.0})
    joined = "|".join(per_user["tags"])
    assert "zzfuture" in joined and "early bird" not in joined
    # without a cut the preprocessing aggregation is reproduced; without the file, tags are dropped
    assert movies_with_tags_before(movies, None, None)["tags"].eq("").all()


_COLS = ["user_id", "movie_id", "rating", "timestamp"]


def test_tag_cutoffs_follow_the_protocol():
    inter = _interactions()
    g = make_split(inter, "global_temporal")
    assert tag_cutoffs(g, "fit") == g.val["timestamp"].min()
    assert tag_cutoffs(g, "test") == g.test["timestamp"].min()
    u = make_split(inter, "user_temporal")
    fit = tag_cutoffs(u, "fit")
    assert isinstance(fit, dict)
    held = pd.concat([u.val, u.test])
    for uid, t in fit.items():
        assert t == held[held.user_id == uid]["timestamp"].min()
        assert t >= u.train[u.train.user_id == uid]["timestamp"].max()


def test_future_items_are_unreleased_and_unseen(tiny_data):
    movies, inter = tiny_data
    cutoff = pd.Timestamp("2000-06-01").timestamp()  # tiny catalogue years run 1990-2004
    fut = future_items(movies, inter.iloc[0:0], cutoff)
    years = movies["year"].to_numpy()[fut]
    assert len(fut) and (years > 2000).all()
    assert len(future_items(movies, inter, cutoff)) == 0  # every tiny film was rated before


def test_ranking_ignores_catalogue_statistics(tiny_data, tiny_components):
    """movies.n_ratings / mean_rating cover all ratings (test included); the unfiltered ranking
    that is evaluated must not depend on them."""
    movies, inter = tiny_data
    scrambled = movies.copy()
    rng = np.random.default_rng(1)
    scrambled["n_ratings"] = rng.integers(0, 1000, len(movies))
    scrambled["mean_rating"] = rng.uniform(0.5, 5.0, len(movies))
    a = tiny_components.ranker(movies, HybridConfig())
    b = tiny_components.ranker(scrambled, HybridConfig())
    ctx = build_context(movies, inter, 42)
    from jev_ml.data.dataset import profiles_from_interactions

    profiles = profiles_from_interactions(inter, ctx.item_index, np.asarray([1, 2, 3]))
    for p in profiles.values():
        assert [r.item for r in a.rank(p, 10, explain=False)] == [
            r.item for r in b.rank(p, 10, explain=False)
        ]


def test_logistic_calibration_fit_uses_no_test_data(tiny_data):
    movies, inter = tiny_data
    cfg = {
        "split": {"strategy": "user_temporal", "val_frac": 0.1, "test_frac": 0.2},
        "models": MODEL_PARAMS,
        "evaluation": {"relevance_threshold": 4.0},
    }

    def run(df: pd.DataFrame) -> dict:
        return calibrate(movies, df, cfg, {}, 42, "m", k=10, horizon=1, strata=(3, None), method="logistic")

    base = run(inter)
    split = make_split(inter, **cfg["split"])
    key = ["user_id", "movie_id", "timestamp"]
    test_keys = set(map(tuple, split.test[key].to_numpy().tolist()))
    scr = inter.copy()
    is_test = np.array([tuple(r) in test_keys for r in scr[key].to_numpy().tolist()])
    scr.loc[is_test, "rating"] = 6.0 - scr.loc[is_test, "rating"]
    again = run(scr)
    for a, b in zip(base["strata"], again["strata"], strict=True):
        fit_keys = ("mean", "scale", "coef", "intercept", "C", "crossfit_log_loss", "validation")
        assert {k: a["logistic_candidate"][k] for k in fit_keys} == {
            k: b["logistic_candidate"][k] for k in fit_keys
        }
        assert a["serving_method"] == b["serving_method"] and a["mapping"] == b["mapping"]
    assert again["fitted_on"]["test_rows_used_for_fit"] == 0
    assert again["calibration_version"] == base["calibration_version"]
    assert base["calibration_version"].startswith("cal-1.1.0-")
    json.dumps(base, allow_nan=False)


def test_new_user_protocol_is_leak_free():
    rows = []
    for u in (1, 2):  # known before the cut
        rows += [(u, 100 + i, 4.0, i) for i in range(5)]
    fit = pd.DataFrame(rows, columns=_COLS)
    win = []
    for u in (1, 7, 8):  # user 1 is known; 7 and 8 are new; 8 has too few window ratings
        n = 15 if u != 8 else HISTORY_PREFIX
        win += [(u, 200 + i, 4.0 if i % 2 == 0 else 2.0, 1000 + i) for i in range(n)]
    window = pd.DataFrame(win, columns=_COLS)
    from jev_ml.data.dataset import ItemIndex

    idx = ItemIndex(np.arange(0, 400))
    profiles, relevant, prefix = new_user_cases(fit, window, idx, 3, 4.0)
    assert set(relevant) == {7}, "known users and users without post-prefix targets are excluded"
    prof_items = set(profiles[7].items.tolist())
    assert prof_items == {200, 201, 202} and profiles[7].train_user_index is None
    assert all(i >= 200 + HISTORY_PREFIX for i in relevant[7]), "targets come after the prefix"
    assert set(prefix["movie_id"]) == set(range(200, 200 + HISTORY_PREFIX))
    movies = pd.DataFrame({"movie_id": np.arange(0, 400), "genres": ["Drama|Comedy"] * 400})
    assert onboarding_genres(prefix, movies)[7] == {"Comedy": 1.0, "Drama": 1.0}
