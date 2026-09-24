"""Statistics for online experiments (docs/EXPERIMENTATION.md, section 5). Pure functions, no I/O.

* ``two_proportion`` : two-proportion z-test (pooled SE for the p-value) with a Wald CI (unpooled SE)
  for the difference ``p_treatment - p_control``.
* ``bootstrap_diff`` : percentile bootstrap CI and a two-sided bootstrap p-value for the difference in
  means of two independent samples (NDCG, diversity, novelty). Seeded, so a report is reproducible.
* ``srm``            : sample-ratio-mismatch chi-square goodness-of-fit test of observed arm sizes
  against the configured weights.
* ``required_sample_size`` : users per arm to detect a relative lift ``mde_relative`` on a baseline
  rate with a two-sided test at ``alpha`` and ``power``.

The unit of analysis is the member (the unit of randomisation), never the exposure: exposures of one
member are correlated, and treating them as independent trials would overstate significance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats as sps


def _r(x: float | None, nd: int = 6) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return round(float(x), nd)


def z_crit(alpha: float) -> float:
    return float(sps.norm.ppf(1.0 - alpha / 2.0))


def two_proportion(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> dict[str, Any]:
    """Treatment vs control. With an empty arm, or no variance in either arm, the test is undefined:
    the p-value is 1 and ``significant`` is False (never a fabricated effect)."""
    out: dict[str, Any] = {
        "control": {"successes": x_c, "n": n_c, "rate": _r(x_c / n_c) if n_c else None},
        "treatment": {"successes": x_t, "n": n_t, "rate": _r(x_t / n_t) if n_t else None},
        "diff": None,
        "relative_lift": None,
        "ci": [None, None],
        "z": None,
        "p_value": 1.0,
        "significant": False,
        "test": "two-proportion z-test",
    }
    if n_c == 0 or n_t == 0:
        return out
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c
    pooled = (x_c + x_t) / (n_c + n_t)
    se_pool = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_c + 1.0 / n_t))
    se = math.sqrt(p_c * (1.0 - p_c) / n_c + p_t * (1.0 - p_t) / n_t)
    zc = z_crit(alpha)
    out["diff"] = _r(diff)
    out["relative_lift"] = _r(diff / p_c) if p_c > 0 else None
    out["ci"] = [_r(diff - zc * se), _r(diff + zc * se)]
    if se_pool == 0.0:
        return out
    z = diff / se_pool
    p = math.erfc(abs(z) / math.sqrt(2.0))
    out["z"] = _r(z, 4)
    out["p_value"] = _r(p)
    out["significant"] = bool(p < alpha)
    return out


def bootstrap_diff(
    control: Sequence[float],
    treatment: Sequence[float],
    alpha: float = 0.05,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Difference in means (treatment - control) with a percentile bootstrap CI. The p-value is the
    two-sided bootstrap tail probability of the difference crossing 0 (a percentile-inversion p)."""
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    out: dict[str, Any] = {
        "control": {"mean": _r(float(c.mean())) if len(c) else None, "n": len(c)},
        "treatment": {"mean": _r(float(t.mean())) if len(t) else None, "n": len(t)},
        "diff": None,
        "relative_lift": None,
        "ci": [None, None],
        "p_value": 1.0,
        "significant": False,
        "test": f"bootstrap ({resamples} resamples, percentile CI)",
    }
    if len(c) < 2 or len(t) < 2:
        return out
    diff = float(t.mean() - c.mean())
    out["diff"] = _r(diff)
    out["relative_lift"] = _r(diff / float(c.mean())) if c.mean() != 0 else None
    rng = np.random.default_rng(seed)
    bc = c[rng.integers(0, len(c), size=(resamples, len(c)))].mean(axis=1)
    bt = t[rng.integers(0, len(t), size=(resamples, len(t)))].mean(axis=1)
    d = bt - bc
    lo, hi = np.quantile(d, [alpha / 2.0, 1.0 - alpha / 2.0])
    out["ci"] = [_r(float(lo)), _r(float(hi))]
    if np.all(d == d[0]):  # no variance at all: nothing to test
        return out
    tail = min(float(np.mean(d <= 0.0)), float(np.mean(d >= 0.0)))
    p = min(1.0, 2.0 * tail)
    out["p_value"] = _r(p)
    out["significant"] = bool(p < alpha and (lo > 0.0 or hi < 0.0))
    return out


def srm(observed: Sequence[int], weights: Sequence[float], alpha: float = 0.001) -> dict[str, Any]:
    """Sample-ratio mismatch: chi-square goodness of fit of the arm sizes to the configured weights.
    A tiny p (default alpha 0.001, the usual SRM threshold) means assignment or logging is broken and
    the comparison must not be trusted."""
    obs = np.asarray(observed, dtype=float)
    w = np.asarray(weights, dtype=float)
    total = float(obs.sum())
    out: dict[str, Any] = {
        "observed": [int(o) for o in obs],
        "expected": None,
        "chi2": None,
        "p_value": None,
        "alpha": alpha,
        "detected": False,
        "test": "chi-square goodness of fit",
    }
    if total == 0 or len(obs) < 2 or w.sum() <= 0:
        return out
    exp = total * w / w.sum()
    chi2, p = sps.chisquare(obs, exp)
    out["expected"] = [_r(float(e), 2) for e in exp]
    out["chi2"] = _r(float(chi2), 4)
    out["p_value"] = _r(float(p))
    out["detected"] = bool(p < alpha)
    return out


def required_sample_size(
    baseline: float | None, mde_relative: float, alpha: float = 0.05, power: float = 0.8
) -> int | None:
    """Users per arm for a two-sided two-proportion test to detect ``baseline * (1 + mde_relative)``.
    None when the baseline is unknown or degenerate (0 or 1)."""
    if baseline is None or not 0.0 < baseline < 1.0 or mde_relative <= 0:
        return None
    p1 = baseline
    p2 = min(0.999999, baseline * (1.0 + mde_relative))
    if p2 == p1:
        return None
    za, zb = z_crit(alpha), float(sps.norm.ppf(power))
    n = (za + zb) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
    return math.ceil(n)


def ndcg_at_k(ranks: Sequence[int], n_relevant: int, k: int = 10) -> float:
    """Binary-relevance NDCG@k: ``ranks`` are the 1-based served ranks of the positively interacted
    items, ``n_relevant`` every distinct positive item in the window (listed or not; the ideal list
    puts min(n_relevant, k) of them on top). 0 when there are none."""
    if n_relevant <= 0:
        return 0.0
    dcg = sum(1.0 / math.log2(r + 1) for r in set(ranks) if 1 <= r <= k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(n_relevant, k) + 1))
    return dcg / idcg


def intra_list_diversity(genre_sets: Sequence[frozenset[str]]) -> float | None:
    """Mean pairwise (1 - Jaccard) of the items' genre sets; None below two items."""
    n = len(genre_sets)
    if n < 2:
        return None
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = genre_sets[i], genre_sets[j]
            union = len(a | b)
            total += 1.0 - (len(a & b) / union if union else 1.0)
            pairs += 1
    return total / pairs


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else None
