"""Uncertainty for offline metrics: per-user bootstrap CIs and paired model comparisons.

Every ranking metric in this package is a mean over evaluated users, so the unit of resampling is
the user (a cluster of candidate rows, never a single row). Comparisons between two models are
*paired*: both models are scored on the same users, and the statistic is the per-user difference.

* ``bootstrap_ci``        percentile bootstrap of a mean (B resamples of users, with replacement)
* ``paired_comparison``   bootstrap CI of the mean difference, a paired sign-flip permutation
                          p-value (exact null of "no difference" under exchangeability), and the
                          share of users where A beats, ties or loses to B
* ``holm``                Holm-Bonferroni adjustment of a family of p-values
* ``cluster_bootstrap``   generic user-cluster bootstrap for statistics that are not per-user
                          means (AUC, Brier, ECE of a calibrator)

All functions are deterministic for a fixed ``seed``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

DEFAULT_B = 2000
DEFAULT_PERMUTATIONS = 10_000


def _as_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError("expected a 1-D sequence of per-user values")
    return a


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    b: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Mean and percentile bootstrap (1 - alpha) CI of per-user values."""
    x = _as_array(values)
    n = len(x)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0, "b": b}
    rng = np.random.default_rng(seed)
    means = np.empty(b)
    chunk = max(1, 2_000_000 // max(n, 1))  # bounded memory for large n
    for start in range(0, b, chunk):
        stop = min(b, start + chunk)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": n, "b": b}


def paired_comparison(
    a: Sequence[float] | np.ndarray,
    b_values: Sequence[float] | np.ndarray,
    b: int = DEFAULT_B,
    permutations: int = DEFAULT_PERMUTATIONS,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired comparison of model A vs model B on the same users (A - B).

    Returns the mean difference with its bootstrap CI, a two-sided sign-flip permutation p-value
    and the win/tie/loss shares. ``significant`` means the CI excludes 0."""
    x, y = _as_array(a), _as_array(b_values)
    if len(x) != len(y):
        raise ValueError("paired comparison needs the same users in the same order")
    d = x - y
    n = len(d)
    if n == 0:
        return {"diff": None, "lo": None, "hi": None, "p_value": None, "n": 0}
    ci = bootstrap_ci(d, b=b, alpha=alpha, seed=seed)
    rng = np.random.default_rng(seed + 1)
    observed = abs(float(d.mean()))
    nz = d[d != 0]
    if len(nz) == 0:
        p = 1.0
    else:
        extreme = 0
        chunk = max(1, 2_000_000 // max(len(nz), 1))
        for start in range(0, permutations, chunk):
            stop = min(permutations, start + chunk)
            signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, len(nz)))
            stat = np.abs((signs * nz).sum(axis=1) / n)
            extreme += int(np.sum(stat >= observed - 1e-15))
        p = (extreme + 1) / (permutations + 1)
    return {
        "diff": float(d.mean()),
        "lo": ci["lo"],
        "hi": ci["hi"],
        "p_value": float(p),
        "n": n,
        "wins": float(np.mean(d > 0)),
        "ties": float(np.mean(d == 0)),
        "losses": float(np.mean(d < 0)),
        "significant": bool(ci["lo"] > 0 or ci["hi"] < 0),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (monotone, capped at 1)."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def cluster_bootstrap(
    clusters: np.ndarray,
    statistic: Callable[[np.ndarray], float | None],
    b: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap a row-level statistic by resampling whole clusters (users) with replacement.

    ``statistic`` receives an array of row indices (rows of resampled clusters, repeated as drawn)
    and returns a float, or None when it is undefined on that resample (skipped)."""
    clusters = np.asarray(clusters)
    uniq, inverse = np.unique(clusters, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == c) for c in range(len(uniq))]
    point = statistic(np.arange(len(clusters)))
    rng = np.random.default_rng(seed)
    vals: list[float] = []
    for _ in range(b):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([rows_of[c] for c in pick])
        v = statistic(idx)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals or point is None:
        return {"value": point, "lo": None, "hi": None, "b": len(vals)}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"value": float(point), "lo": float(lo), "hi": float(hi), "b": len(vals)}
