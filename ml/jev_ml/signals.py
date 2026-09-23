"""User interaction signals → numeric weights. Training and inference both use this module, so
there is one definition of how a rating, a favorite or a watch counts as evidence of taste.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

# Explicit ratings on the MovieLens 0.5-5 scale.
LIKE_THRESHOLD = 4.0  # rating >= this is a "like"; also the relevance threshold in evaluation
DISLIKE_THRESHOLD = 2.5  # rating <= this is a "dislike"

FAVORITE_WEIGHT = 2.0
WATCH_WEIGHT = 0.6  # watched, not rated: weak positive evidence
ONBOARDING_PICK_WEIGHT = 1.5  # movie picked as a favorite during onboarding


def rating_weight(rating: float) -> float:
    """Implicit-feedback confidence weight of an explicit rating (always > 0).

    CF and MF treat every consumed item as a signal, because having watched something predicts
    what else a user will watch. The strength of that signal grows with the rating:
    5.0 -> 2.0, 4.0 -> 1.0, 3.0 -> 0.5, <=2.5 -> 0.1.
    """
    if rating >= LIKE_THRESHOLD:
        return 1.0 + (rating - LIKE_THRESHOLD)
    if rating > DISLIKE_THRESHOLD:
        return 0.5
    return 0.1


def rating_weights(ratings: np.ndarray) -> np.ndarray:
    r = np.asarray(ratings, dtype=np.float64)
    return np.where(
        r >= LIKE_THRESHOLD, 1.0 + (r - LIKE_THRESHOLD), np.where(r > DISLIKE_THRESHOLD, 0.5, 0.1)
    )


def preference_weight(rating: float) -> float:
    """Signed taste weight for content/genre profiles: liked items pull, disliked items push."""
    return float(np.clip((rating - 3.0) / 2.0, -1.0, 1.0))


@dataclass
class UserProfile:
    """Everything a recommender may use about one user, in *item-index* space.

    `items` / `weights` are positive implicit evidence (ratings of any value, favorites, watches).
    `pref_weights` carries the signed per-item taste used by content and genre profiles.
    `train_user_index` is set only for users seen during training (dataset users).
    """

    items: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    pref_weights: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    timestamps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    genre_prefs: dict[str, float] = field(default_factory=dict)
    exclude: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    train_user_index: int | None = None

    @property
    def n_interactions(self) -> int:
        return len(self.items)

    @property
    def consumed(self) -> np.ndarray:
        return np.union1d(self.items, self.exclude).astype(np.int64)

    @classmethod
    def from_events(
        cls,
        events: Iterable[tuple[int, float, float, float]],
        genre_prefs: dict[str, float] | None = None,
        exclude: Iterable[int] = (),
        train_user_index: int | None = None,
        recency_half_life_days: float | None = None,
    ) -> UserProfile:
        """Merge (item_idx, implicit_weight, preference_weight, timestamp) events per item.

        An item can appear multiple times (rated + favorited + watched): weights are summed,
        preference is the max-magnitude signal, timestamp is the latest. Optionally decays
        weights so recent tastes dominate.
        """
        merged: dict[int, list[float]] = {}
        for item, w, p, ts in events:
            cur = merged.get(item)
            if cur is None:
                merged[item] = [w, p, ts]
            else:
                cur[0] += w
                if abs(p) > abs(cur[1]):
                    cur[1] = p
                cur[2] = max(cur[2], ts)
        if not merged:
            return cls(
                genre_prefs=dict(genre_prefs or {}),
                exclude=np.asarray(sorted(set(exclude)), dtype=np.int64),
                train_user_index=train_user_index,
            )
        keys = np.asarray(sorted(merged), dtype=np.int64)
        vals = np.asarray([merged[int(k)] for k in keys], dtype=np.float64)
        weights, prefs, ts = vals[:, 0], vals[:, 1], vals[:, 2]
        if recency_half_life_days and ts.max() > 0:
            age_days = (ts.max() - ts) / 86400.0
            decay = np.power(0.5, age_days / recency_half_life_days)
            weights = weights * (0.5 + 0.5 * decay)
            prefs = prefs * (0.5 + 0.5 * decay)
        return cls(
            items=keys,
            weights=weights,
            pref_weights=prefs,
            timestamps=ts,
            genre_prefs=dict(genre_prefs or {}),
            exclude=np.asarray(sorted(set(exclude)), dtype=np.int64),
            train_user_index=train_user_index,
        )
