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
    ) -> None:
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
            int(u): set(item_index.indices_of(g["movie_id"].to_numpy()).tolist())
            for u, g in rel[rel["user_id"].isin(users)].groupby("user_id")
        }
        positions = {int(u): i for i, u in enumerate(train_user_ids)}
        src = train[train["user_id"].isin(users)]
        if truncate_profile_to:
            src = (
                src.sort_values(["user_id", "timestamp", "movie_id"])
                .groupby("user_id")
                .head(truncate_profile_to)
            )
            positions = {}  # truncated users are "unseen": models must fold them in
        self.profiles = profiles_from_interactions(src, item_index, np.asarray(users), positions)

    def evaluate(self, name: str, recommend: RecommendFn) -> EvalResult:
        t0 = time.perf_counter()
        kmax = max(self.ks)
        sums = {f"{m}@{k}": 0.0 for m in ACCURACY_METRICS for k in self.ks}
        rec_lists: list[list[int]] = []
        per_user_ndcg: list[float] = []
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
                    sums[f"{m}@{k}"] += fn(recs, rel, k)
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
        return EvalResult(name, metrics, len(self.users), secs, per_user_ndcg)
