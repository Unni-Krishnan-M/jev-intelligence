import numpy as np
import pandas as pd
import pytest

from jev_ml.data.preprocess import clean_title, normalize_tag
from jev_ml.features import ContentFeaturizer, movie_field_tokens, normalize_genre
from jev_ml.signals import UserProfile, preference_weight, rating_weight, rating_weights


def test_rating_weight_is_monotonic_and_positive():
    ratings = np.arange(0.5, 5.01, 0.5)
    w = rating_weights(ratings)
    assert np.all(w > 0)
    assert np.all(np.diff(w) >= 0)
    assert rating_weight(5.0) == 2.0 and rating_weight(4.0) == 1.0
    assert np.allclose(w, [rating_weight(r) for r in ratings])


def test_preference_weight_sign():
    assert preference_weight(5.0) == 1.0
    assert preference_weight(3.0) == 0.0
    assert preference_weight(1.0) == -1.0


def test_profile_merges_duplicate_items():
    p = UserProfile.from_events([(3, 1.0, 0.5, 10.0), (3, 2.0, -0.9, 20.0), (1, 0.6, 0.3, 5.0)])
    assert p.items.tolist() == [1, 3]
    assert p.weights.tolist() == [0.6, 3.0]
    assert p.pref_weights.tolist() == [0.3, -0.9]  # max-magnitude signal wins
    assert p.timestamps.tolist() == [5.0, 20.0]
    assert p.n_interactions == 2


def test_profile_consumed_includes_exclusions():
    p = UserProfile.from_events([(2, 1.0, 1.0, 0.0)], exclude=[7, 2])
    assert p.consumed.tolist() == [2, 7]


@pytest.mark.parametrize(
    ("raw", "title", "year"),
    [
        ("Matrix, The (1999)", "The Matrix", 1999),
        ("Toy Story (1995)", "Toy Story", 1995),
        (
            "City of Lost Children, The (Cité des enfants perdus, La) (1995)",
            "The City of Lost Children",
            1995,
        ),
        ("Babylon 5", "Babylon 5", None),
    ],
)
def test_clean_title(raw, title, year):
    assert clean_title(raw) == (title, year)


def test_text_normalization():
    assert normalize_tag("  Dark   COMEDY ") == "dark comedy"
    assert normalize_genre("science fiction film") == "science fiction"
    assert normalize_genre("Sci-Fi") == "science fiction"  # MovieLens and Wikidata share a vocabulary


def test_field_tokens_are_prefixed():
    toks = movie_field_tokens(
        {
            "title": "Inception",
            "genres": "Sci-Fi|Action",
            "directors": "Christopher Nolan",
            "cast": "Leonardo DiCaprio",
            "keywords": "dream",
            "tags": "mind-bending",
            "year": 2010,
            "description": "2010 film by Christopher Nolan",
            "wd_genres": "",
        }
    )
    assert "d:christopher nolan" in toks["directors"]
    assert "g:science fiction" in toks["genres"]
    assert toks["decade"] == ["y:2010s"]
    assert "w:inception" in toks["title"]
    assert "o:film" not in toks["description"]  # domain stop word removed


def test_featurizer_state_roundtrip_is_exact(tiny_data):
    movies = tiny_data[0]
    f = ContentFeaturizer(min_df=1)
    x = f.fit_transform(movies)
    g = ContentFeaturizer.from_state(f.state())
    y = g.transform(movies)
    assert x.shape == y.shape
    assert np.allclose(x.toarray(), y.toarray(), atol=1e-6)
    norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
    assert np.allclose(norms[norms > 0], 1.0, atol=1e-5)


def test_featurizer_handles_new_movie(tiny_data):
    f = ContentFeaturizer(min_df=1)
    f.fit_transform(tiny_data[0])
    vec = f.transform(
        pd.DataFrame([{"title": "Brand New", "genres": "Horror", "directors": "Rex Dread", "year": 2030}])
    )
    assert vec.shape[0] == 1 and vec.nnz > 0
