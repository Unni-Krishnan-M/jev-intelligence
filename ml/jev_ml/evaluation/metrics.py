"""Top-K ranking metrics (binary relevance) and beyond-accuracy metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for i in ranked[:k] if i in relevant) / k


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for i in ranked[:k] if i in relevant) / len(relevant)


def f1_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    p, r = precision_at_k(ranked, relevant, k), recall_at_k(ranked, relevant, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(pos + 2) for pos, i in enumerate(ranked[:k]) if i in relevant)
    ideal = sum(1.0 / np.log2(pos + 2) for pos in range(min(len(relevant), k)))
    return float(dcg / ideal)


def average_precision_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for pos, i in enumerate(ranked[:k]):
        if i in relevant:
            hits += 1
            total += hits / (pos + 1)
    return total / min(len(relevant), k)


def hit_rate_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(i in relevant for i in ranked[:k]) else 0.0


ACCURACY_METRICS: dict[str, Callable[[Sequence[int], set[int], int], float]] = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "f1": f1_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "hit_rate": hit_rate_at_k,
}


def catalog_coverage(rec_lists: Sequence[Sequence[int]], n_items: int, k: int) -> float:
    seen = {i for recs in rec_lists for i in recs[:k]}
    return len(seen) / n_items if n_items else 0.0


def intra_list_diversity(recs: Sequence[int], pairwise_sim: Callable[[np.ndarray], np.ndarray]) -> float:
    """Mean pairwise dissimilarity (1 - cosine) of a recommendation list."""
    if len(recs) < 2:
        return 0.0
    sims = pairwise_sim(np.asarray(recs, dtype=np.int64))
    n = len(recs)
    off = (sims.sum() - np.trace(sims)) / (n * (n - 1))
    return float(1.0 - off)


def novelty(recs: Sequence[int], pop_fraction: np.ndarray, n_users: int) -> float:
    """Mean self-information −log2 p(i), where p(i) is the share of training users who saw i."""
    if not recs:
        return 0.0
    floor = 1.0 / max(n_users, 1)
    p = np.maximum(pop_fraction[np.asarray(recs, dtype=np.int64)], floor)
    return float(np.mean(-np.log2(p)))
