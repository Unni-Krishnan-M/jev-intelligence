"""Inference engine: loads one registered model version from disk and serves recommendations.

It performs no training. Users who were never seen in training (every app user) are scored
directly from their interaction events:
  - item-kNN and content work from the events as they are,
  - ALS folds the user in against the frozen item factors,
  - popularity, genre preference and recency need no history at all.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.data.dataset import ItemIndex, load_movies, split_list
from jev_ml.explain import explain
from jev_ml.models.als import ALSRecommender
from jev_ml.models.content import ContentRecommender
from jev_ml.models.hybrid import HybridConfig, HybridRanker, RankedItem, RecommendationFilters
from jev_ml.models.itemknn import ItemKNNRecommender
from jev_ml.models.popularity import PopularityRecommender
from jev_ml.paths import MODELS_DIR
from jev_ml.registry import active_version
from jev_ml.signals import (
    FAVORITE_WEIGHT,
    ONBOARDING_PICK_WEIGHT,
    WATCH_WEIGHT,
    UserProfile,
    preference_weight,
    rating_weight,
)

log = logging.getLogger(__name__)


@dataclass
class Interaction:
    """One user event from the application database."""

    movie_id: int
    kind: str  # "rating" | "favorite" | "watch" | "onboarding"
    value: float | None = None  # rating value for kind == "rating"
    timestamp: float = 0.0


@dataclass
class Recommendation:
    movie_id: int
    title: str
    score: float
    reason: str
    reason_code: str
    secondary_reasons: list[str]
    anchor_movie_ids: list[int]
    signals: dict[str, dict[str, float]]
    rank: int


class RecommendationEngine:
    def __init__(self, model_dir: Path) -> None:
        t0 = time.perf_counter()
        self.model_dir = model_dir
        self.manifest: dict[str, Any] = json.loads((model_dir / "manifest.json").read_text())
        self.version: str = self.manifest["version"]
        self.movies: pd.DataFrame = load_movies(model_dir / "movies.csv")
        self.item_index = ItemIndex(self.movies["movie_id"].to_numpy())
        self.popularity = PopularityRecommender.load(model_dir / "popularity")
        self.content = ContentRecommender.load(model_dir / "content")
        self.itemknn = ItemKNNRecommender.load(model_dir / "itemknn")
        self.als = ALSRecommender.load(model_dir / "als")
        self.config = HybridConfig.from_dict(json.loads((model_dir / "hybrid.json").read_text()))
        self.ranker = HybridRanker(
            self.movies, self.popularity, self.content, self.itemknn, self.als, self.config
        )
        self._titles = self.movies["title"].to_numpy()
        self._ids = self.movies["movie_id"].to_numpy()
        self._validate()
        self.load_seconds = time.perf_counter() - t0
        log.info("loaded model %s (%d items) in %.2fs", self.version, len(self.movies), self.load_seconds)

    @classmethod
    def load_active(cls, models_dir: Path = MODELS_DIR) -> RecommendationEngine:
        version = active_version(models_dir)
        if version is None:
            raise FileNotFoundError(f"no active model in {models_dir}; run scripts/train_models.py")
        return cls(models_dir / version)

    def _validate(self) -> None:
        n = len(self.movies)
        checks = {
            "popularity": len(self.popularity.popularity) == n,
            "content": self.content.item_vectors.shape[0] == n,
            "itemknn": self.itemknn.sim.shape == (n, n),
            "als": self.als.item_factors.shape[0] == n,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ValueError(f"model artifacts inconsistent with item catalog: {bad}")
        if not np.isfinite(self.als.item_factors).all():
            raise ValueError("ALS item factors contain non-finite values")

    # --- profiles --------------------------------------------------------------------------------
    def build_profile(
        self,
        interactions: Iterable[Interaction],
        genre_prefs: Iterable[str] = (),
        excluded_movie_ids: Iterable[int] = (),
    ) -> UserProfile:
        events = []
        for it in interactions:
            idx = self.item_index.index_of(it.movie_id)
            if idx is None:
                continue  # movie unknown to this model version (added after training)
            if it.kind == "rating" and it.value is not None:
                events.append((idx, rating_weight(it.value), preference_weight(it.value), it.timestamp))
            elif it.kind == "favorite":
                events.append((idx, FAVORITE_WEIGHT, 1.0, it.timestamp))
            elif it.kind == "watch":
                events.append((idx, WATCH_WEIGHT, 0.3, it.timestamp))
            elif it.kind == "onboarding":
                events.append((idx, ONBOARDING_PICK_WEIGHT, 1.0, it.timestamp))
        exclude = [i for m in excluded_movie_ids if (i := self.item_index.index_of(m)) is not None]
        return UserProfile.from_events(events, genre_prefs={g: 1.0 for g in genre_prefs}, exclude=exclude)

    # --- recommendations ------------------------------------------------------------------------
    def _to_rec(self, r: RankedItem, profile: UserProfile, rank: int) -> Recommendation:
        ex = explain(r, profile, lambda i: str(self._titles[i]), lambda i: int(self._ids[i]))
        return Recommendation(
            movie_id=int(self._ids[r.item]),
            title=str(self._titles[r.item]),
            score=round(float(r.score), 6),
            reason=ex.reason,
            reason_code=ex.reason_code,
            secondary_reasons=ex.secondary,
            anchor_movie_ids=ex.anchor_movie_ids,
            signals={k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in r.signals.items()},
            rank=rank,
        )

    def recommend(
        self, profile: UserProfile, k: int = 20, offset: int = 0, filters: RecommendationFilters | None = None
    ) -> list[Recommendation]:
        ranked = self.ranker.rank(profile, k=k, offset=offset, filters=filters, explain=True)
        return [self._to_rec(r, profile, offset + i + 1) for i, r in enumerate(ranked)]

    def effective_weights(self, profile: UserProfile) -> dict[str, float]:
        return self.ranker.effective_weights(profile)

    def similar(self, movie_id: int, k: int = 12) -> list[dict[str, Any]]:
        idx = self.item_index.index_of(movie_id)
        if idx is None:
            raise KeyError(movie_id)
        out = []
        src_title = str(self._titles[idx])
        for i, score, parts in self.ranker.similar_items(idx, k):
            shared = self.shared_features(idx, i)
            if shared:
                reason = f"Shares {shared[0]} with {src_title}"
            elif parts["collaborative"] > parts["content"]:
                reason = f"Often watched by viewers of {src_title}"
            else:
                reason = f"Similar in style to {src_title}"
            out.append(
                {
                    "movie_id": int(self._ids[i]),
                    "title": str(self._titles[i]),
                    "score": round(float(score), 6),
                    "reason": reason,
                    "signals": {k2: round(v, 6) for k2, v in parts.items()},
                }
            )
        return out

    def similar_to_metadata(self, movie: dict[str, Any], k: int = 12) -> list[dict[str, Any]]:
        """Item cold start: neighbours of a movie the model has never seen, from metadata alone."""
        vec = self.content.vectorize_new([movie])
        out = []
        for i, score in self.ranker.similar_to_vector(vec, k):
            if score <= 0:
                continue
            out.append(
                {
                    "movie_id": int(self._ids[i]),
                    "title": str(self._titles[i]),
                    "score": round(score, 6),
                    "reason": "Similar metadata (genres, people, themes)",
                    "signals": {"content": round(score, 6), "collaborative": 0.0},
                }
            )
        return out

    def shared_features(self, a: int, b: int) -> list[str]:
        """Human-readable shared director/cast/keywords between two catalog items (from metadata)."""
        ra, rb = self.movies.iloc[a], self.movies.iloc[b]
        out: list[str] = []
        for col, fmt in (("directors", "director {}"), ("cast", "{}"), ("keywords", "the theme “{}”")):
            common = sorted(set(split_list(ra[col])) & set(split_list(rb[col])))
            out.extend(fmt.format(c) for c in common[:1])
        return out

    def trending(
        self,
        k: int = 20,
        offset: int = 0,
        exclude_movie_ids: Iterable[int] = (),
        live_counts: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        exclude = np.asarray(
            [i for m in exclude_movie_ids if (i := self.item_index.index_of(m)) is not None], dtype=np.int64
        )
        boost = None
        if live_counts:
            boost = {i: v for m, v in live_counts.items() if (i := self.item_index.index_of(m)) is not None}
        return [
            {"movie_id": int(self._ids[i]), "title": str(self._titles[i]), "score": round(s, 6)}
            for i, s in self.ranker.trending(k, offset, exclude, boost)
        ]

    def popular(self, k: int = 20, offset: int = 0, min_votes: int = 20) -> list[dict[str, Any]]:
        """Top Bayesian-average rating among movies with at least `min_votes` ratings."""
        score = np.where(self.popularity.user_counts >= min_votes, self.popularity.bayes_rating, -np.inf)
        order = np.lexsort((np.arange(len(score)), -score))
        order = [i for i in order if np.isfinite(score[i])][offset : offset + k]
        return [
            {"movie_id": int(self._ids[i]), "title": str(self._titles[i]), "score": round(float(score[i]), 4)}
            for i in order
        ]

    def health(self) -> dict[str, Any]:
        return {
            "model_version": self.version,
            "n_items": len(self.movies),
            "components": ["popularity", "content", "itemknn", "als", "hybrid"],
            "als_factors": int(self.als.factors),
            "content_features": int(self.content.item_vectors.shape[1]),
            "itemknn_nnz": int(self.itemknn.sim.nnz),
            "load_seconds": round(self.load_seconds, 3),
            "dataset_version": self.manifest.get("dataset_version"),
        }
