"""UNDERSTAND (movie domain): MovieLens ratings as generic observations + the movie series specs.

Observations: one row per rating <= as_of (``timestamp``, ``entity_id`` = ``user:<id>``,
``entity_type`` = user, ``event_type`` = rating, ``value`` = the rating, ``source`` = movielens,
plus ``user_id``, ``movie_id`` and the group column ``genre`` = the film's genres as a tuple).

Specs (the v1.1 series, same ids and order):
* ``volume:all`` (count), ``active_users:all`` (distinct raters), adverse direction down;
* per genre: ``volume:genre:<G>`` (count; forecast, not scanned), ``share:genre:<G>`` (share of
  ratings; scanned, share-forecast, adverse down) and ``rating:genre:<G>`` (mean rating, only months
  with >= ``rating_min_count`` ratings; scanned). A film counts once per genre;
  ``(no genres listed)`` is excluded. Genres come from the processed catalogue.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from jev_ml.core.series import Series, SeriesSpec
from jev_ml.core.series import build_series as core_build_series
from jev_ml.domains.movie.config import IntelConfig
from jev_ml.domains.movie.ingest import Prepared

GROUP = "genre"


def movie_observations(prep: Prepared) -> pd.DataFrame:
    r = prep.ratings
    genres = {m: tuple(dict.fromkeys(g)) for m, g in prep.genre_of.items()}
    uid = r["user_id"].to_numpy()
    return pd.DataFrame(
        {
            "timestamp": r["timestamp"].to_numpy(dtype=float),
            "entity_id": pd.Series(uid).map(lambda u: f"user:{u}").to_numpy(),
            "entity_type": "user",
            "event_type": "rating",
            "value": r["rating"].to_numpy(dtype=float),
            "source": "movielens",
            "user_id": uid,
            "movie_id": r["movie_id"].to_numpy(),
            GROUP: r["movie_id"].map(genres).to_numpy(),
        }
    )


def movie_specs(prep: Prepared, cfg: IntelConfig) -> list[SeriesSpec]:
    groups = tuple(prep.genres)
    return [
        SeriesSpec(
            "volume:all",
            "count",
            "ratings/month",
            name="volume",
            adverse_direction="down",
            forecast=True,
            trend_unit="volume/month",
            source="movielens",
            count_noun="ratings",
        ),
        SeriesSpec(
            "active_users:all",
            "nunique",
            "users/month",
            name="active_users",
            value_column="user_id",
            adverse_direction="down",
            forecast=True,
            trend_unit="active_users/month",
            source="movielens",
            count_noun="ratings",
        ),
        SeriesSpec(
            "volume:genre:{group}",
            "count",
            "ratings/month",
            name="volume",
            group_by=GROUP,
            groups=groups,
            entity_type="genre",
            scan_anomalies=False,  # moves with platform volume: one burst would be reported ~15 times
            forecast=True,
            trend_unit="volume/month",
            source="movielens",
            count_noun="ratings",
        ),
        SeriesSpec(
            "share:genre:{group}",
            "share",
            "share of ratings",
            group_by=GROUP,
            groups=groups,
            entity_type="genre",
            adverse_direction="down",
            share_forecast=True,
            trend_unit="share/month",
            source="movielens",
            count_noun="ratings",
        ),
        SeriesSpec(
            "rating:genre:{group}",
            "mean",
            "mean rating",
            name="rating",
            group_by=GROUP,
            groups=groups,
            entity_type="genre",
            min_count=cfg.rating_min_count,
            trend_unit="stars/month",
            source="movielens",
            count_noun="ratings",
        ),
    ]


def build_series(prep: Prepared, cfg: IntelConfig) -> list[Series]:
    """The movie series from validated frames (v1.1 signature, used by evaluation and scenarios)."""
    return core_build_series(movie_observations(prep), movie_specs(prep, cfg), prep.as_of, domain="movie")


def month_range(prep: Prepared) -> tuple[pd.PeriodIndex, bool]:
    from jev_ml.core.series import period_grid

    return period_grid(prep.ratings["timestamp"].to_numpy(dtype=np.float64), prep.as_of)
