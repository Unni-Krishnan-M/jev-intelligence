"""Model-side evidence: paired bootstrap over per-user NDCG@10, new data since training, influence.

Paired bootstrap: users are resampled with replacement (seeded) and every model's mean NDCG@10 is
recomputed on the *same* resample, so model differences keep their per-user pairing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import sub_seed
from jev_ml.domains.movie.config import IntelConfig

EXCLUDED_MODELS = ("random",)  # a random ranker is a sanity floor, never a serving candidate


def bootstrap_models(per_user: dict[str, list[float]] | None, cfg: IntelConfig) -> dict[str, Any] | None:
    if not per_user:
        return None
    models = sorted(m for m in per_user if m not in EXCLUDED_MODELS)
    lengths = {len(per_user[m]) for m in models}
    if len(models) < 2 or len(lengths) != 1:
        return None
    n = lengths.pop()
    if n < cfg.bootstrap_min_users:
        return {"n_users": n, "insufficient": True}
    X = np.array([per_user[m] for m in models], dtype=float).T  # (users, models)
    rng = np.random.default_rng(sub_seed(cfg.seed, "bootstrap:ndcg"))
    idx = rng.integers(0, n, size=(cfg.bootstrap_samples, n))
    counts = np.zeros((cfg.bootstrap_samples, n))
    np.add.at(counts, (np.repeat(np.arange(cfg.bootstrap_samples), n), idx.ravel()), 1.0)
    means = counts @ X / n  # (B, models): resample means without materialising X[idx]
    point = X.mean(axis=0)
    order = np.argsort(-point, kind="mergesort")
    best, runner = int(order[0]), int(order[1])
    wins = np.bincount(np.argmax(means, axis=1), minlength=len(models)) / cfg.bootstrap_samples
    out: dict[str, Any] = {
        "n_users": n,
        "insufficient": False,
        "models": models,
        "mean_ndcg": {m: float(point[i]) for i, m in enumerate(models)},
        "p_best": {m: float(wins[i]) for i, m in enumerate(models)},
        "best": models[best],
        "runner_up": models[runner],
        "p_best_beats_runner_up": float((means[:, best] > means[:, runner]).mean()),
    }
    if "hybrid" in models:
        h = models.index("hybrid")
        singles = [i for i, m in enumerate(models) if m != "hybrid"]
        best_single = max(singles, key=lambda i: point[i])
        lead = (means[:, h] - means[:, best_single]) / np.maximum(means[:, best_single], 1e-12)
        out["hybrid"] = {
            "best_single": models[best_single],
            "lead": float((point[h] - point[best_single]) / max(point[best_single], 1e-12)),
            "lead_ci95": [float(np.quantile(lead, 0.025)), float(np.quantile(lead, 0.975))],
            "p_lead_below_min": float((lead < cfg.model_min_lead).mean()),
            "p_hybrid_better": float((lead > 0).mean()),
        }
    return out


def trending_top(
    ratings: pd.DataFrame, as_of_ts: float, cfg: IntelConfig, exclude_users: set[int] | None = None
) -> list[int]:
    """Top-N films by decay-weighted rating count at as_of (the recommender's trending signal)."""
    r = ratings if not exclude_users else ratings[~ratings["user_id"].isin(list(exclude_users))]
    age = (as_of_ts - r["timestamp"].to_numpy(dtype=float)) / 86400.0
    w = np.power(0.5, age / cfg.influence_half_life_days)
    s = pd.Series(w).groupby(r["movie_id"].to_numpy()).sum()
    s = s.sort_index()
    top = s.sort_values(ascending=False, kind="mergesort").head(cfg.influence_top_n)
    return [int(m) for m in top.index]


def influence(ratings: pd.DataFrame, as_of_ts: float, cfg: IntelConfig, users: list[int]) -> dict[str, Any]:
    """Exact influence: how many of the top-N trending films change when `users`' ratings are
    removed (jointly), plus the same count per individual user."""
    base = trending_top(ratings, as_of_ts, cfg)
    joint = trending_top(ratings, as_of_ts, cfg, set(users))
    per_user = {}
    for u in users:
        alt = trending_top(ratings, as_of_ts, cfg, {u})
        per_user[int(u)] = len(set(base) - set(alt))
    return {
        "top_n": cfg.influence_top_n,
        "changed": len(set(base) - set(joint)),
        "entered": sorted(set(joint) - set(base)),
        "left": sorted(set(base) - set(joint)),
        "per_user": per_user,
    }


def new_events_since_training(
    ratings: pd.DataFrame,
    app_ratings: pd.DataFrame,
    cutoff_ts: float | None,
    created_ts: float | None,
) -> dict[str, Any]:
    ml_new = int((ratings["timestamp"] > cutoff_ts).sum()) if cutoff_ts is not None else 0
    app_new = (
        int((app_ratings["timestamp"] > created_ts).sum())
        if created_ts is not None and len(app_ratings)
        else 0
    )
    return {"movielens": ml_new, "app": app_new, "total": ml_new + app_new}
