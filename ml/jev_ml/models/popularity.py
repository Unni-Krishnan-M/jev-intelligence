"""Non-personalized popularity baseline + trending and Bayesian-average quality signals.

Ranking score: ``log1p(likes) + reach_weight * log1p(users)``. With ``score_half_life_days`` set,
both counts are exponentially time-decayed relative to the newest training interaction, which
turns the baseline into a "recently popular" ranking (a strong baseline under a global time split).
The defaults (reach 0.25, no decay) are the historical formula, so saved models are unchanged."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jev_ml.models.base import Recommender, TrainContext
from jev_ml.signals import LIKE_THRESHOLD, UserProfile


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(
        self,
        trending_half_life_days: float = 365.0,
        bayes_prior_votes: float = 10.0,
        reach_weight: float = 0.25,
        score_half_life_days: float | None = None,
    ) -> None:
        self.trending_half_life_days = trending_half_life_days
        self.bayes_prior_votes = bayes_prior_votes
        self.reach_weight = float(reach_weight)
        self.score_half_life_days = None if score_half_life_days is None else float(score_half_life_days)
        self.user_counts = np.zeros(0)
        self.like_counts = np.zeros(0)
        self.trending = np.zeros(0)
        self.bayes_rating = np.zeros(0)
        self.popularity = np.zeros(0)
        self.n_users = 0
        self.reference_ts = 0.0

    def fit(self, ctx: TrainContext) -> PopularityRecommender:
        n = ctx.n_items
        df = ctx.interactions
        items = ctx.item_index.indices_of(df["movie_id"].to_numpy())
        ratings = df["rating"].to_numpy(dtype=np.float64)
        ts = df["timestamp"].to_numpy(dtype=np.float64)

        self.n_users = int(df["user_id"].nunique())
        self.user_counts = np.bincount(items, minlength=n).astype(np.float64)
        self.like_counts = np.bincount(items, weights=(ratings >= LIKE_THRESHOLD).astype(float), minlength=n)

        self.reference_ts = float(ts.max()) if len(ts) else 0.0
        age_days = (self.reference_ts - ts) / 86400.0
        decay = np.power(0.5, age_days / self.trending_half_life_days)
        self.trending = np.bincount(items, weights=decay, minlength=n)

        sums = np.bincount(items, weights=ratings, minlength=n)
        global_mean = float(ratings.mean()) if len(ratings) else 3.5
        m = self.bayes_prior_votes
        self.bayes_rating = (sums + m * global_mean) / (self.user_counts + m)

        # ranking score: log-scaled count of *liked* interactions, blended with raw reach
        likes, reach = self.like_counts, self.user_counts
        if self.score_half_life_days:
            d = np.power(0.5, age_days / self.score_half_life_days)
            likes = np.bincount(items, weights=(ratings >= LIKE_THRESHOLD) * d, minlength=n)
            reach = np.bincount(items, weights=d, minlength=n)
        self.popularity = np.log1p(likes) + self.reach_weight * np.log1p(reach)
        return self

    def score(self, profile: UserProfile) -> np.ndarray:
        return self.popularity.copy()

    def item_popularity_fraction(self) -> np.ndarray:
        """Share of training users who interacted with each item (for novelty)."""
        return self.user_counts / max(self.n_users, 1)

    def params(self) -> dict[str, Any]:
        out: dict[str, Any] = {"trending_half_life_days": self.trending_half_life_days,
                               "bayes_prior_votes": self.bayes_prior_votes}
        # only non-default scoring params are recorded, so existing manifests stay byte-identical
        if self.reach_weight != 0.25:
            out["reach_weight"] = self.reach_weight
        if self.score_half_life_days is not None:
            out["score_half_life_days"] = self.score_half_life_days
        return out

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "popularity.npz", user_counts=self.user_counts,
                            like_counts=self.like_counts, trending=self.trending,
                            bayes_rating=self.bayes_rating, popularity=self.popularity)
        self._write_json(path / "popularity.json", {**self.params(), "n_users": self.n_users,
                                                    "reference_ts": self.reference_ts})

    @classmethod
    def load(cls, path: Path) -> PopularityRecommender:
        meta = cls._read_json(path / "popularity.json")
        obj = cls(meta["trending_half_life_days"], meta["bayes_prior_votes"],
                  meta.get("reach_weight", 0.25), meta.get("score_half_life_days"))
        arrs = np.load(path / "popularity.npz")
        for k in ("user_counts", "like_counts", "trending", "bayes_rating", "popularity"):
            setattr(obj, k, arrs[k])
        obj.n_users = meta["n_users"]
        obj.reference_ts = meta["reference_ts"]
        return obj
