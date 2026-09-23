import copy

import numpy as np
import pytest
from unit.test_models import genre_block, scifi_fan

from jev_ml.explain import explain
from jev_ml.models.hybrid import SIGNALS, HybridConfig, RecommendationFilters
from jev_ml.signals import UserProfile


def test_weights_sum_to_one_and_adapt(tiny_ranker):
    cold = tiny_ranker.effective_weights(UserProfile(genre_prefs={"Horror": 1.0}))
    warm_profile = UserProfile.from_events([(i, 1.0, 1.0, 0.0) for i in range(12)])
    warm = tiny_ranker.effective_weights(warm_profile)
    for w in (cold, warm):
        assert set(w) == set(SIGNALS)
        assert abs(sum(w.values()) - 1.0) < 1e-9
    assert cold["collaborative"] == 0 and cold["latent"] == 0 and cold["content"] == 0
    assert cold["popularity"] > warm["popularity"]
    assert warm["collaborative"] > 0 and warm["latent"] > 0


def test_minmax_normalization(tiny_ranker):
    out = tiny_ranker._normalize(np.array([2.0, 4.0, 6.0]))
    assert out.tolist() == [0.0, 0.5, 1.0]
    assert tiny_ranker._normalize(np.array([3.0, 3.0])).tolist() == [1.0, 1.0]
    assert tiny_ranker._normalize(np.array([0.0, 0.0])).tolist() == [0.0, 0.0]


def test_rank_excludes_consumed_and_respects_k(tiny_ranker):
    prof = scifi_fan()
    recs = tiny_ranker.rank(prof, k=7)
    assert len(recs) == 7
    ids = [r.item for r in recs]
    assert not set(ids) & set(prof.consumed.tolist())
    assert len(set(ids)) == len(ids)
    assert all(np.isfinite(r.score) for r in recs)


def test_rank_is_personalized_and_deterministic(tiny_ranker):
    prof = scifi_fan()
    a = [r.item for r in tiny_ranker.rank(prof, k=6)]
    b = [r.item for r in tiny_ranker.rank(prof, k=6)]
    assert a == b
    assert sum(i in genre_block("Sci-Fi") for i in a) >= 5
    horror = UserProfile.from_events([(i, 2.0, 1.0, 0.0) for i in list(genre_block("Horror"))[:4]])
    c = [r.item for r in tiny_ranker.rank(horror, k=6)]
    assert a != c


def test_pagination_is_consistent(tiny_ranker):
    prof = scifi_fan()
    full = [r.item for r in tiny_ranker.rank(prof, k=12)]
    page2 = [r.item for r in tiny_ranker.rank(prof, k=6, offset=6)]
    assert page2 == full[6:12]


def test_cold_start_uses_genre_preferences(tiny_ranker):
    prof = UserProfile(genre_prefs={"Comedy": 1.0})
    recs = tiny_ranker.rank(prof, k=5)
    assert len(recs) == 5
    assert sum(r.item in genre_block("Comedy") for r in recs) >= 4
    for r in recs:
        assert r.signals["collaborative"]["weight"] == 0
        assert r.signals["latent"]["weight"] == 0


def test_brand_new_user_gets_popular_items(tiny_ranker):
    recs = tiny_ranker.rank(UserProfile(), k=5)
    assert len(recs) == 5
    assert all(
        r.signals["popularity"]["weight"] == pytest.approx(1 - r.signals["recency"]["weight"]) for r in recs
    )


def test_filters(tiny_ranker):
    prof = scifi_fan()
    recs = tiny_ranker.rank(prof, k=5, filters=RecommendationFilters(genres=["Horror"]))
    assert recs and all(r.item in genre_block("Horror") for r in recs)
    recs = tiny_ranker.rank(prof, k=5, filters=RecommendationFilters(year_min=2000))
    assert all(tiny_ranker._years[r.item] >= 2000 for r in recs)


def test_mmr_increases_diversity(tiny_ranker):
    prof = UserProfile.from_events([(i, 1.0, 1.0, 0.0) for i in (0, 15, 30)])
    focused = copy.copy(tiny_ranker)
    focused.config = HybridConfig(diversity_lambda=1.0)
    diverse = copy.copy(tiny_ranker)
    diverse.config = HybridConfig(diversity_lambda=0.3)

    def ild(ranker):
        items = np.array([r.item for r in ranker.rank(prof, k=8, explain=False)])
        sims = tiny_ranker.content.pairwise(items)
        n = len(items)
        return 1 - (sims.sum() - np.trace(sims)) / (n * (n - 1))

    assert ild(diverse) >= ild(focused)


def test_explanations_only_cite_profile_items(tiny_ranker, tiny_data):
    movies = tiny_data[0]
    prof = scifi_fan()
    titles = movies["title"].tolist()
    for r in tiny_ranker.rank(prof, k=5):
        ex = explain(r, prof, lambda i: titles[i], lambda i: int(movies["movie_id"].iloc[i]))
        assert ex.reason
        profile_ids = {int(movies["movie_id"].iloc[i]) for i in prof.items}
        assert set(ex.anchor_movie_ids) <= profile_ids
        mentioned = [t for t in titles if t in ex.reason]
        for t in mentioned:  # any film named in the reason is one the user actually interacted with
            assert titles.index(t) in prof.items.tolist() or t == titles[r.item]
