"""Synthetic data with known structure for the intelligence-layer tests.

Not named conftest.py: tests/unit import helpers via `from conftest import ...`, which a second
conftest module would shadow.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from jev_ml.intel.ingest import PipelineInputs
from jev_ml.intel.series import Series

GENRES = ["Drama", "Comedy", "Horror", "Sci-Fi"]
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def make_ratings(
    seed: int = 0, n_users: int = 160, n_movies: int = 240, months: int = 84
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    movies = pd.DataFrame(
        {
            "movie_id": np.arange(1, n_movies + 1),
            "title": [f"Film {i}" for i in range(1, n_movies + 1)],
            "genres": [
                "|".join(sorted(set(rng.choice(GENRES, size=rng.integers(1, 3))))) for _ in range(n_movies)
            ],
        }
    )
    quality = rng.normal(3.5, 0.5, n_movies)
    pop = 1.0 / np.arange(1, n_movies + 1) ** 0.8
    pop /= pop.sum()
    start = pd.Timestamp("2008-01-01", tz="UTC")
    rows = []
    for u in range(1, n_users + 1):
        first = int(rng.integers(0, months - 6))
        dur = int(min(months - first, 1 + rng.exponential(12)))
        bias = rng.normal(0, 0.4)
        seen: set[int] = set()
        for m in range(first, first + dur):
            k = int(rng.integers(3, 15))
            m0 = (start + pd.DateOffset(months=m)).timestamp()
            m1 = (start + pd.DateOffset(months=m + 1)).timestamp()
            for _ in range(k):
                i = int(rng.choice(n_movies, p=pop))
                if i in seen:
                    continue
                seen.add(i)
                r = float(np.clip(np.round((quality[i] + bias + rng.normal(0, 0.7)) * 2) / 2, 0.5, 5.0))
                rows.append((u, i + 1, r, int(rng.uniform(m0, m1 - 1))))
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True), movies


def make_inputs(seed: int = 0, **kw) -> PipelineInputs:
    ratings, movies = make_ratings(seed)
    return PipelineInputs(
        interactions=ratings,
        movies=movies,
        dataset_meta={"dataset_version": "synthetic-1"},
        now=NOW,
        **kw,
    )


def make_series(values: np.ndarray, metric: str = "volume", sid: str | None = None) -> Series:
    months = pd.period_range("2010-01", periods=len(values), freq="M")
    vals = np.asarray(values, dtype=float)
    return Series(
        sid or f"{metric}:all",
        metric,
        "all",
        "platform",
        "ratings/month",
        months,
        vals,
        False,
        vals.copy() if metric == "volume" else np.full(len(vals), 1000.0),
    )
