import math

import numpy as np
import pandas as pd

from jev_ml.evaluation.metrics import (
    average_precision_at_k,
    catalog_coverage,
    f1_at_k,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)
from jev_ml.evaluation.split import global_temporal_split, user_temporal_split


def test_accuracy_metrics_known_values():
    ranked, rel = [1, 2, 3, 4], {2, 4, 9}
    assert precision_at_k(ranked, rel, 4) == 0.5
    assert math.isclose(recall_at_k(ranked, rel, 4), 2 / 3)
    assert math.isclose(f1_at_k(ranked, rel, 4), 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3))
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert math.isclose(ndcg_at_k(ranked, rel, 4), dcg / idcg)
    assert math.isclose(average_precision_at_k(ranked, rel, 4), (1 / 2 + 2 / 4) / 3)
    assert hit_rate_at_k(ranked, rel, 1) == 0.0 and hit_rate_at_k(ranked, rel, 2) == 1.0


def test_perfect_ranking_scores_one():
    assert ndcg_at_k([5, 6], {5, 6}, 2) == 1.0
    assert average_precision_at_k([5, 6], {5, 6}, 2) == 1.0


def test_beyond_accuracy_metrics():
    assert catalog_coverage([[0, 1], [1, 2]], n_items=10, k=2) == 0.3
    ident = lambda items: np.ones((len(items), len(items)))  # noqa: E731
    orth = lambda items: np.eye(len(items))  # noqa: E731
    assert intra_list_diversity([1, 2, 3], ident) == 0.0
    assert intra_list_diversity([1, 2, 3], orth) == 1.0
    pop = np.array([0.5, 0.25, 0.0])
    assert math.isclose(novelty([0, 1], pop, n_users=100), (1 + 2) / 2)
    assert novelty([2], pop, n_users=4) == 2.0  # floor at 1/n_users


def _interactions():
    rng = np.random.default_rng(0)
    rows = [
        (u, m, 4.0, int(rng.integers(0, 10_000)))
        for u in range(1, 21)
        for m in rng.choice(500, 20, replace=False)
    ]
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def test_user_temporal_split_has_no_per_user_leakage():
    df = _interactions()
    s = user_temporal_split(df, val_frac=0.1, test_frac=0.2)
    assert len(s.train) + len(s.val) + len(s.test) == len(df)
    for u in df["user_id"].unique():
        tr = s.train[s.train.user_id == u]
        va = s.val[s.val.user_id == u]
        te = s.test[s.test.user_id == u]
        assert len(te) == 4 and len(va) == 2
        assert tr["timestamp"].max() <= va["timestamp"].min() <= te["timestamp"].min()
    keys = lambda d: set(zip(d.user_id, d.movie_id, strict=True))  # noqa: E731
    assert not (keys(s.train) & keys(s.test)) and not (keys(s.val) & keys(s.test))


def test_global_temporal_split_orders_time():
    s = global_temporal_split(_interactions())
    assert s.train["timestamp"].max() < s.val["timestamp"].min()
    assert s.val["timestamp"].max() < s.test["timestamp"].min()
