"""One evaluation protocol for every model.

For each evaluable user (history in the training data and at least one relevant target item):
  1. build the profile from *training* rows only,
  2. ask the model for top-max(K) items, excluding everything in the profile,
  3. score the list against the target items rated >= relevance threshold.
All models see the same users, profiles, candidate exclusions and K values.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from jev_ml.data.dataset import ItemIndex, profiles_from_interactions
from jev_ml.evaluation.metrics import (
    ACCURACY_METRICS,
    catalog_coverage,
    intra_list_diversity,
    novelty,
)
from jev_ml.signals import LIKE_THRESHOLD, UserProfile

log = logging.getLogger(__name__)

RecommendFn = Callable[[UserProfile, int], np.ndarray]


@dataclass
class EvalResult:
    model: str
    metrics: dict[str, float]
    n_users: int
    seconds: float
    per_user_ndcg: list[float] = field(default_factory=list)
    #: per-user values of the metrics in PER_USER_METRICS, aligned with ``user_ids``
    per_user: dict[str, list[float]] = field(default_factory=dict)
    user_ids: list[int] = field(default_factory=list)


# per-user values kept for bootstrap CIs and paired tests (jev_ml.evaluation.stats)
PER_USER_METRICS = ("ndcg@10", "recall@10", "precision@10", "hit_rate@10")


class Evaluator:
    def __init__(
        self,
        train: pd.DataFrame,
        target: pd.DataFrame,
        item_index: ItemIndex,
        train_user_ids: np.ndarray,
        pairwise_sim: Callable[[np.ndarray], np.ndarray],
        pop_fraction: np.ndarray,
        ks: tuple[int, ...] = (5, 10, 20),
        relevance_threshold: float = LIKE_THRESHOLD,
        max_users: int | None = None,
        seed: int = 42,
        truncate_profile_to: int | None = None,
        genre_prefs: dict[int, dict[str, float]] | None = None,
        exclude_items: np.ndarray | None = None,
    ) -> None:
        """``truncate_profile_to`` keeps each user's first N training interactions (0 = an empty
        profile) and folds the users in as unseen. ``genre_prefs`` sets simulated onboarding genres
        per user. ``exclude_items`` are never recommendable to anyone (e.g. films released after a
        global time cut); they are added to every profile's exclusion list."""
        self.ks = tuple(sorted(ks))
        self.item_index = item_index
        self.pairwise_sim = pairwise_sim
        self.pop_fraction = pop_fraction
        self.n_train_users = len(train_user_ids)
        rel = target[target["rating"] >= relevance_threshold]
        train_users = set(train["user_id"].unique().tolist())
        users = sorted(u for u in rel["user_id"].unique().tolist() if u in train_users)
        if max_users and len(users) > max_users:
            rng = np.random.default_rng(seed)
            users = sorted(rng.choice(users, size=max_users, replace=False).tolist())
        self.users = users
        self.relevant = {
            int(g["user_id"].iloc[0]): set(item_index.indices_of(g["movie_id"].to_numpy()).tolist())
            for _, g in rel[rel["user_id"].isin(users)].groupby("user_id")
        }
        positions = {int(u): i for i, u in enumerate(train_user_ids)}
        src = train[train["user_id"].isin(users)]
        if truncate_profile_to is not None:
            src = (
                src.sort_values(["user_id", "timestamp", "movie_id"])
                .groupby("user_id")
                .head(truncate_profile_to)
            )
            positions = {}  # truncated users are "unseen": models must fold them in
        self.profiles = profiles_from_interactions(src, item_index, np.asarray(users), positions)
        _decorate_profiles(self.profiles, genre_prefs, exclude_items)

    @classmethod
    def from_profiles(
        cls,
        profiles: dict[int, UserProfile],
        relevant: dict[int, set[int]],
        item_index: ItemIndex,
        n_train_users: int,
        pairwise_sim: Callable[[np.ndarray], np.ndarray],
        pop_fraction: np.ndarray,
        ks: tuple[int, ...] = (5, 10, 20),
        exclude_items: np.ndarray | None = None,
    ) -> Evaluator:
        """An evaluator over prebuilt profiles and relevant sets (e.g. the global-split new-user
        protocol, where the profile comes from the user's first ratings after the time cut).
        Users without any relevant item are dropped, as in the default constructor."""
        ev = cls.__new__(cls)
        ev.ks = tuple(sorted(ks))
        ev.item_index = item_index
        ev.pairwise_sim = pairwise_sim
        ev.pop_fraction = pop_fraction
        ev.n_train_users = n_train_users
        ev.users = sorted(u for u, rel in relevant.items() if rel and u in profiles)
        ev.relevant = {u: relevant[u] for u in ev.users}
        ev.profiles = {u: profiles[u] for u in ev.users}
        _decorate_profiles(ev.profiles, None, exclude_items)
        return ev

    def evaluate(self, name: str, recommend: RecommendFn) -> EvalResult:
        t0 = time.perf_counter()
        kmax = max(self.ks)
        sums = {f"{m}@{k}": 0.0 for m in ACCURACY_METRICS for k in self.ks}
        rec_lists: list[list[int]] = []
        per_user_ndcg: list[float] = []
        per_user: dict[str, list[float]] = {m: [] for m in PER_USER_METRICS}
        div_sum, nov_sum = 0.0, 0.0
        for u in self.users:
            profile = self.profiles[u]
            recs = [int(i) for i in recommend(profile, kmax)][:kmax]
            consumed = set(profile.consumed.tolist())
            if consumed.intersection(recs):
                raise AssertionError(f"{name} recommended already-consumed items for user {u}")
            rel = self.relevant[u]
            for m, fn in ACCURACY_METRICS.items():
                for k in self.ks:
                    v = fn(recs, rel, k)
                    sums[f"{m}@{k}"] += v
                    if f"{m}@{k}" in per_user:
                        per_user[f"{m}@{k}"].append(v)
            for key in PER_USER_METRICS:  # K=10 metrics even when 10 is not in ks
                m_, k_ = key.split("@")
                if int(k_) not in self.ks:
                    per_user[key].append(ACCURACY_METRICS[m_](recs, rel, int(k_)))
            per_user_ndcg.append(ACCURACY_METRICS["ndcg"](recs, rel, 10))
            rec_lists.append(recs)
            top10 = recs[:10]
            div_sum += intra_list_diversity(top10, self.pairwise_sim)
            nov_sum += novelty(top10, self.pop_fraction, self.n_train_users)
        n = max(len(self.users), 1)
        metrics = {k: v / n for k, v in sums.items()}
        for k in self.ks:
            metrics[f"coverage@{k}"] = catalog_coverage(rec_lists, len(self.item_index), k)
        metrics["diversity@10"] = div_sum / n
        metrics["novelty@10"] = nov_sum / n
        secs = time.perf_counter() - t0
        log.info(
            "eval %-14s users=%d ndcg@10=%.4f recall@10=%.4f (%.1fs)",
            name,
            len(self.users),
            metrics["ndcg@10"],
            metrics["recall@10"],
            secs,
        )
        return EvalResult(
            name, metrics, len(self.users), secs, per_user_ndcg, per_user, [int(u) for u in self.users]
        )


def _decorate_profiles(
    profiles: dict[int, UserProfile],
    genre_prefs: dict[int, dict[str, float]] | None,
    exclude_items: np.ndarray | None,
) -> None:
    extra = None if exclude_items is None else np.asarray(exclude_items, dtype=np.int64)
    for u, prof in profiles.items():
        if genre_prefs and u in genre_prefs:
            prof.genre_prefs = dict(genre_prefs[u])
        if extra is not None and len(extra):
            prof.exclude = np.union1d(prof.exclude, extra).astype(np.int64)
