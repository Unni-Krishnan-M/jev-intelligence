import numpy as np
from conftest import GENRES, N_PER_GENRE

from jev_ml.models.base import topk_indices
from jev_ml.signals import UserProfile


def genre_block(g: str) -> range:
    i = GENRES.index(g)
    return range(i * N_PER_GENRE, (i + 1) * N_PER_GENRE)


def scifi_fan(n: int = 4) -> UserProfile:
    items = list(genre_block("Sci-Fi"))[:n]
    return UserProfile.from_events([(i, 2.0, 1.0, float(k)) for k, i in enumerate(items)])


def test_topk_is_deterministic_with_ties():
    scores = np.array([1.0, 3.0, 3.0, 2.0, 3.0])
    assert topk_indices(scores, 2).tolist() == [1, 2]  # ties broken by index
    assert topk_indices(scores, 3, mask=np.array([1])).tolist() == [2, 4, 3]
    assert topk_indices(np.array([-np.inf, 1.0]), 5).tolist() == [1]


def test_popularity_counts_likes(tiny_components):
    pop = tiny_components.popularity
    assert pop.user_counts.sum() > 0
    assert np.all(np.isfinite(pop.popularity))
    top = int(np.argmax(pop.popularity))
    assert pop.like_counts[top] == pop.like_counts.max() or pop.user_counts[top] > 0


def test_content_neighbours_share_genre(tiny_components):
    content = tiny_components.content
    idx, sims = content.similar(0, 5)
    assert 0 not in idx.tolist()
    assert all(i in genre_block("Sci-Fi") for i in idx)
    assert np.all(np.diff(sims) <= 1e-9)


def test_content_profile_prefers_liked_genre(tiny_components):
    scores = tiny_components.content.score(scifi_fan())
    top = topk_indices(scores, 5, scifi_fan().consumed)
    assert all(i in genre_block("Sci-Fi") for i in top)


def test_content_explanation_features_are_real(tiny_components, tiny_data):
    movies = tiny_data[0]
    prof = scifi_fan()
    contribs = tiny_components.content.explain(prof, 10)
    assert contribs
    anchors = [c.item for c in contribs if c.kind == "item"]
    assert anchors and anchors[0] in prof.items.tolist()
    directors = {c.label for c in contribs if c.kind == "director"}
    assert directors <= set(movies["directors"])


def test_itemknn_and_als_personalize(tiny_components):
    prof = scifi_fan()
    for model in (tiny_components.itemknn, tiny_components.als):
        top = topk_indices(model.score(prof), 5, prof.consumed)
        in_genre = sum(i in genre_block("Sci-Fi") for i in top)
        assert in_genre >= 4, (model.name, top)


def test_als_loss_decreases_and_factors_finite(tiny_components):
    als = tiny_components.als
    loss = als.loss_history
    assert loss[-1] < loss[0]
    assert np.isfinite(als.item_factors).all() and np.isfinite(als.user_factors).all()


def test_als_fold_in_matches_known_user_ranking(tiny_components, tiny_data):
    """Folding in a training user's own history ranks items like their learned factor does."""
    from jev_ml.data.dataset import ItemIndex, profiles_from_interactions

    movies, inter = tiny_data
    idx = ItemIndex(movies["movie_id"].to_numpy())
    uid = 5
    known = profiles_from_interactions(inter, idx, np.array([uid]), {uid: uid - 1})[uid]
    unseen = profiles_from_interactions(inter, idx, np.array([uid]))[uid]
    a = topk_indices(tiny_components.als.score(known), 10, known.consumed)
    b = topk_indices(tiny_components.als.score(unseen), 10, unseen.consumed)
    assert len(set(a) & set(b)) >= 7


def test_model_save_load_roundtrip(tmp_path, tiny_components):
    from jev_ml.models.als import ALSRecommender
    from jev_ml.models.content import ContentRecommender
    from jev_ml.models.itemknn import ItemKNNRecommender
    from jev_ml.models.popularity import PopularityRecommender

    prof = scifi_fan()
    for cls, model in (
        (PopularityRecommender, tiny_components.popularity),
        (ContentRecommender, tiny_components.content),
        (ItemKNNRecommender, tiny_components.itemknn),
        (ALSRecommender, tiny_components.als),
    ):
        model.save(tmp_path / model.name)
        loaded = cls.load(tmp_path / model.name)
        assert np.allclose(loaded.score(prof), model.score(prof), atol=1e-4), model.name


def test_registry_file_is_world_readable(tmp_path):
    import stat

    from jev_ml.registry import REGISTRY_FILE, register_version

    register_version({"version": "v-test"}, models_dir=tmp_path)
    mode = stat.S_IMODE((tmp_path / REGISTRY_FILE).stat().st_mode)
    assert mode & 0o044 == 0o044  # group and others can read it
