"""Shared fixtures.

Environment variables are set here, at import time, so the application under test uses a
throwaway SQLite database and a model trained on a small synthetic catalogue. The real data and
model directories are never touched. Tests that need the real MovieLens artifacts live in
tests/ml/test_real_artifacts.py and skip when those artifacts are absent.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="jev-tests-"))
os.environ.setdefault("JEV_DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ["JEV_MODELS_DIR"] = str(_TMP / "models")
os.environ["JEV_EXPERIMENTS_DIR"] = str(_TMP / "experiments")
os.environ["JEV_JWT_SECRET"] = "test-secret-not-for-production-0123456789abcdef"
os.environ["JEV_ADMIN_EMAIL"] = "admin@example.com"
os.environ["JEV_ADMIN_PASSWORD"] = "admin-pass-123"
os.environ["JEV_RATE_LIMIT_PER_MINUTE"] = "100000"
os.environ["JEV_AUTH_RATE_LIMIT_PER_MINUTE"] = "100000"
os.environ.pop("JEV_REDIS_URL", None)
os.environ["JEV_INTEL_RUN_ON_STARTUP"] = "false"  # intel tests trigger runs explicitly

import json  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from jev_ml.models.hybrid import HybridConfig, HybridRanker  # noqa: E402
from jev_ml.training import Components, build_context, fit_components, save_model_version  # noqa: E402

TMP_ROOT = _TMP
GENRES = ["Sci-Fi", "Comedy", "Drama", "Horror"]
DIRECTORS = {
    "Sci-Fi": ["Ava Orbit", "Kai Nebula"],
    "Comedy": ["Lou Punch", "Mia Giggle"],
    "Drama": ["Ira Tear", "Noa Stage"],
    "Horror": ["Rex Dread", "Zoe Night"],
}
N_PER_GENRE = 15

MODEL_PARAMS = {
    "popularity": {"trending_half_life_days": 365, "bayes_prior_votes": 5},
    "content": {"min_df": 1, "n_neighbors": 20},
    "itemknn": {"k": 20, "shrinkage": 2, "binary": False},
    "als": {"factors": 8, "regularization": 0.05, "alpha": 10, "iterations": 8},
}


def make_movies() -> pd.DataFrame:
    rows = []
    mid = 1
    for g in GENRES:
        for j in range(N_PER_GENRE):
            director = DIRECTORS[g][j % 2]
            rows.append(
                {
                    "movie_id": mid,
                    "title": f"{g} Story {j + 1}",
                    "raw_title": f"{g} Story {j + 1} (20{j:02d})",
                    "year": 1990 + j,
                    "genres": g if j % 5 else f"{g}|Drama" if g != "Drama" else g,
                    "tags": f"{g.lower()} vibes",
                    "directors": director,
                    "director_source": "wikidata",
                    "cast": f"{g} Actor A|{g} Actor {j % 3}",
                    "keywords": f"{g.lower()} theme",
                    "wd_genres": f"{g.lower()} film",
                    "countries": "Testland",
                    "description": f"a {g.lower()} film by {director}",
                    "runtime_min": 100 + j,
                    "imdb_id": f"tt{mid:07d}",
                    "tmdb_id": mid,
                    "n_ratings": 0,
                    "mean_rating": None,
                }
            )
            mid += 1
    return pd.DataFrame(rows)


def make_interactions(movies: pd.DataFrame, n_users: int = 80, seed: int = 7) -> pd.DataFrame:
    """Each user loves one genre (high ratings) and samples a few others (low ratings)."""
    rng = np.random.default_rng(seed)
    by_genre = {g: movies[movies["genres"].str.startswith(g)]["movie_id"].to_numpy() for g in GENRES}
    rows = []
    t0 = 1_500_000_000
    for u in range(1, n_users + 1):
        fav = GENRES[u % len(GENRES)]
        liked = rng.choice(by_genre[fav], size=10, replace=False)
        others = rng.choice(np.concatenate([by_genre[g] for g in GENRES if g != fav]), size=4, replace=False)
        ts = t0 + u * 1000
        order = rng.permutation(len(liked) + len(others))  # likes and dislikes interleave in time
        events = [(m, float(rng.choice([4.0, 4.5, 5.0]))) for m in liked]
        events += [(m, float(rng.choice([1.5, 2.0, 2.5]))) for m in others]
        for slot, (m, r) in zip(order, events, strict=True):
            rows.append((u, int(m), r, ts + int(slot) * 60))
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def with_stats(movies: pd.DataFrame, inter: pd.DataFrame) -> pd.DataFrame:
    stats = inter.groupby("movie_id")["rating"].agg(n_ratings="size", mean_rating="mean")
    out = movies.drop(columns=["n_ratings", "mean_rating"]).merge(
        stats, left_on="movie_id", right_index=True, how="left"
    )
    out["n_ratings"] = out["n_ratings"].fillna(0).astype(int)
    return out


@pytest.fixture(scope="session")
def tiny_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = make_movies()
    inter = make_interactions(movies)
    return with_stats(movies, inter), inter


@pytest.fixture(scope="session")
def tiny_components(tiny_data: tuple[pd.DataFrame, pd.DataFrame]) -> Components:
    movies, inter = tiny_data
    ctx = build_context(movies, inter, seed=42)
    return fit_components(ctx, MODEL_PARAMS, seed=42)


@pytest.fixture(scope="session")
def tiny_ranker(tiny_data: tuple[pd.DataFrame, pd.DataFrame], tiny_components: Components) -> HybridRanker:
    return tiny_components.ranker(tiny_data[0], HybridConfig())


@pytest.fixture(scope="session")
def processed_dir(tiny_data: tuple[pd.DataFrame, pd.DataFrame]) -> Path:
    """A data/processed-style directory holding the synthetic dataset (for pipeline tests)."""
    movies, inter = tiny_data
    d = TMP_ROOT / "processed"
    d.mkdir(exist_ok=True)
    movies.to_csv(d / "movies.csv", index=False)
    inter.to_csv(d / "interactions.csv", index=False)
    (d / "dataset_meta.json").write_text(
        json.dumps({"dataset_version": "synthetic-test-v1", "n_movies": len(movies)})
    )
    return d


@pytest.fixture(scope="session")
def tiny_model_dir(tiny_data: tuple[pd.DataFrame, pd.DataFrame], tiny_components: Components) -> Path:
    """Registered + active synthetic model version inside JEV_MODELS_DIR (used by the API)."""
    from jev_ml.registry import register_version

    movies, inter = tiny_data
    models_dir = Path(os.environ["JEV_MODELS_DIR"])
    out = save_model_version(
        tiny_components,
        movies,
        HybridConfig(),
        {
            "dataset_version": "synthetic-test-v1",
            "training_seed": 42,
            "training_config": {"models": MODEL_PARAMS},
            "metrics": {"ndcg@10": 0.5},
            "trained_on_rows": len(inter),
            "split": {"strategy": "none"},
        },
        models_dir,
    )
    register_version(json.loads((out / "manifest.json").read_text()), models_dir, activate=True)
    return out


@pytest.fixture(scope="session")
def pipeline_result(processed_dir: Path, tiny_model_dir: Path) -> dict:
    """Full split → evaluate → train → register → report run on the synthetic dataset (no tuning)."""
    from jev_ml.training import run_pipeline

    return run_pipeline(quick=True, processed_dir=processed_dir, activate=False)


@pytest.fixture(scope="session")
def client(tiny_model_dir: Path, pipeline_result: dict):
    from fastapi.testclient import TestClient

    from jev_api.main import create_app

    with TestClient(create_app()) as c:
        yield c


def register(client, email: str, name: str = "Tester", password: str = "password-123") -> dict:
    r = client.post("/auth/register", json={"email": email, "password": password, "display_name": name})
    assert r.status_code == 201, r.text
    client.cookies.clear()  # tests authenticate with explicit bearer tokens
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def admin_headers(client) -> dict:
    r = client.post("/auth/login", json={"email": "admin@example.com", "password": "admin-pass-123"})
    assert r.status_code == 200, r.text
    client.cookies.clear()
    return {"Authorization": f"Bearer {r.json()['access_token']}"}
