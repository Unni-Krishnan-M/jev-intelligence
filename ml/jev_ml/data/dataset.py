"""Loading processed data and building sparse interaction matrices."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from jev_ml.data.preprocess import LIST_SEP
from jev_ml.paths import PROCESSED_DIR
from jev_ml.signals import UserProfile, preference_weight, rating_weights

LIST_COLUMNS = ("genres", "tags", "directors", "cast", "keywords", "wd_genres", "countries")


def split_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value:
        return []
    return [v for v in value.split(LIST_SEP) if v]


def load_movies(path: Path = PROCESSED_DIR / "movies.csv") -> pd.DataFrame:
    movies = pd.read_csv(path, dtype={"imdb_id": str, "description": str}, keep_default_na=False,
                         na_values={"year": [""], "runtime_min": [""], "tmdb_id": [""], "mean_rating": [""]})
    movies["year"] = pd.to_numeric(movies["year"], errors="coerce").astype("Int64")
    movies["runtime_min"] = pd.to_numeric(movies["runtime_min"], errors="coerce").astype("Int64")
    movies["mean_rating"] = pd.to_numeric(movies["mean_rating"], errors="coerce")
    return movies.sort_values("movie_id").reset_index(drop=True)


def load_interactions(path: Path = PROCESSED_DIR / "interactions.csv") -> pd.DataFrame:
    return pd.read_csv(path, dtype={"user_id": np.int64, "movie_id": np.int64, "rating": np.float64,
                                    "timestamp": np.int64})


def load_dataset_meta(path: Path = PROCESSED_DIR / "dataset_meta.json") -> dict:
    return json.loads(path.read_text())


@dataclass
class ItemIndex:
    """Bijection between external movie ids and dense item indices 0..n-1 (sorted by movie id)."""

    movie_ids: np.ndarray

    def __post_init__(self) -> None:
        self.movie_ids = np.asarray(self.movie_ids, dtype=np.int64)
        self._pos = {int(m): i for i, m in enumerate(self.movie_ids)}

    def __len__(self) -> int:
        return len(self.movie_ids)

    def index_of(self, movie_id: int) -> int | None:
        return self._pos.get(int(movie_id))

    def indices_of(self, movie_ids: np.ndarray) -> np.ndarray:
        return np.asarray([self._pos[int(m)] for m in movie_ids], dtype=np.int64)


def build_user_item_matrix(
    interactions: pd.DataFrame, item_index: ItemIndex, user_ids: np.ndarray | None = None
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Users × items CSR matrix of implicit confidence weights (see signals.rating_weight)."""
    if user_ids is None:
        user_ids = np.sort(interactions["user_id"].unique())
    upos = {int(u): i for i, u in enumerate(user_ids)}
    rows = interactions["user_id"].map(upos).to_numpy()
    cols = item_index.indices_of(interactions["movie_id"].to_numpy())
    vals = rating_weights(interactions["rating"].to_numpy())
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_index)), dtype=np.float64)
    mat.sum_duplicates()
    return mat, np.asarray(user_ids, dtype=np.int64)


def profiles_from_interactions(
    interactions: pd.DataFrame,
    item_index: ItemIndex,
    user_ids: np.ndarray,
    train_user_positions: dict[int, int] | None = None,
) -> dict[int, UserProfile]:
    """Build a UserProfile per user from rating rows (used by evaluation and batch scripts)."""
    profiles: dict[int, UserProfile] = {}
    grouped = interactions.groupby("user_id")
    for uid in user_ids:
        uid = int(uid)
        if uid not in grouped.groups:
            profiles[uid] = UserProfile()
            continue
        g = grouped.get_group(uid)
        items = item_index.indices_of(g["movie_id"].to_numpy())
        ratings = g["rating"].to_numpy()
        w = rating_weights(ratings)
        p = np.asarray([preference_weight(r) for r in ratings])
        ts = g["timestamp"].to_numpy(dtype=np.float64)
        profiles[uid] = UserProfile.from_events(
            zip(items.tolist(), w.tolist(), p.tolist(), ts.tolist(), strict=True),
            train_user_index=(train_user_positions or {}).get(uid),
        )
    return profiles
