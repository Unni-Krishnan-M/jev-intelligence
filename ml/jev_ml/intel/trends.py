"""DETECT (trends): Mann–Kendall + Theil–Sen over the recent window, and a mean-shift change point.

* Mann–Kendall: Kendall's tau-b between the values and time (scipy handles ties); its p-value
  tests "no monotonic trend".
* Theil–Sen: median pairwise slope with a 95 % confidence interval (scipy.stats.theilslopes).
* direction = up/down only when p < alpha AND the Theil–Sen CI excludes zero; otherwise flat.
* q_value: Benjamini–Hochberg adjusted p over all series tested in the run (added field). Later
  stages (signals, risks, decisions) require q <= trend_fdr in addition to ``direction``.
* Change point: single mean shift. Statistic = max over admissible splits (>= min_segment on each
  side) of the pooled two-sample t statistic; p-value from the same max statistic under a seeded
  null (so the search over splits is accounted for): permutations when the segment residuals are
  not autocorrelated, else simulated AR(1) series with the estimated lag-1 coefficient. Reported
  when p < change_point_alpha.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from jev_ml.intel.common import fnum, month_str, stable_id, sub_seed
from jev_ml.intel.config import IntelConfig
from jev_ml.intel.series import Series


def mann_kendall(x: np.ndarray, t: np.ndarray | None = None) -> tuple[float, float]:
    t = np.arange(len(x), dtype=float) if t is None else t
    if len(x) < 3 or np.all(x == x[0]):
        return 0.0, 1.0
    res = stats.kendalltau(t, x)
    tau = float(res.statistic)
    p = float(res.pvalue)
    return (tau if np.isfinite(tau) else 0.0), (p if np.isfinite(p) else 1.0)


def theil_sen(x: np.ndarray, t: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """Returns (slope, intercept, lo95, hi95)."""
    t = np.arange(len(x), dtype=float) if t is None else t
    res = stats.theilslopes(x, t, alpha=0.95)
    return float(res.slope), float(res.intercept), float(res.low_slope), float(res.high_slope)


def _max_shift_stat(mat: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Rows of mat are series of equal length n. Returns (max |t| per row, argmax split k per row)."""
    n = mat.shape[1]
    S = np.cumsum(mat, axis=1)
    total = S[:, -1:]
    mu = total / n
    ss_tot = ((mat - mu) ** 2).sum(axis=1, keepdims=True)
    k = np.arange(m, n - m + 1)
    m1 = S[:, k - 1] / k
    m2 = (total - S[:, k - 1]) / (n - k)
    between = k * (m1 - mu) ** 2 + (n - k) * (m2 - mu) ** 2
    s2 = np.maximum(ss_tot - between, 1e-12) / (n - 2)
    tstat = np.abs(m1 - m2) / np.sqrt(s2 * (1.0 / k + 1.0 / (n - k)))
    idx = np.argmax(tstat, axis=1)
    return tstat[np.arange(len(mat)), idx], k[idx]


def _ar1_phi(x: np.ndarray, k: int) -> float:
    """Lag-1 autocorrelation of the residuals around the two segment means (split at k).

    Estimated under the alternative so that a genuine shift is not mistaken for autocorrelation,
    then Kendall's small-sample bias correction phi + (1 + 3 phi) / n (the raw estimate is biased
    towards 0 on short series, which would make the null too narrow)."""
    r = x.astype(float).copy()
    r[:k] -= r[:k].mean()
    r[k:] -= r[k:].mean()
    den = float(np.dot(r, r))
    if den <= 0:
        return 0.0
    phi = float(np.dot(r[:-1], r[1:]) / den)
    return float(np.clip(phi + (1 + 3 * phi) / len(x), 0.0, 0.95))


def change_point(x: np.ndarray, min_segment: int, permutations: int, seed: int) -> dict[str, Any] | None:
    """Single mean-shift test. Returns {index, before_mean, after_mean, stat, p_value, null, phi}.

    Null distribution of the max-t statistic: a seeded permutation test when the residuals show no
    lag-1 autocorrelation (phi <= 0.1); otherwise simulated stationary AR(1) series with the
    estimated phi, because permutations assume exchangeable months and are badly anti-conservative
    for autocorrelated monthly data (the statistic is scale/location invariant, so only phi matters).
    """
    n = len(x)
    if n < 2 * min_segment + 1 or np.allclose(x, x[0]):
        return None
    rng = np.random.default_rng(seed)
    obs, k = _max_shift_stat(x[None, :], min_segment)
    kk = int(k[0])
    phi = _ar1_phi(x, kk)
    if phi <= 0.1:
        sims = rng.permuted(np.tile(x, (permutations, 1)), axis=1)
        null_kind = "permutation"
    else:
        e = rng.standard_normal((permutations, n))
        sims = np.empty_like(e)
        sims[:, 0] = e[:, 0] / np.sqrt(1 - phi**2)
        for t in range(1, n):
            sims[:, t] = phi * sims[:, t - 1] + e[:, t]
        null_kind = "ar1"
    null, _ = _max_shift_stat(sims, min_segment)
    p = (1.0 + float((null >= obs[0] - 1e-12).sum())) / (permutations + 1.0)
    return {
        "index": kk,
        "before_mean": float(x[:kk].mean()),
        "after_mean": float(x[kk:].mean()),
        "stat": float(obs[0]),
        "p_value": p,
        "null": null_kind,
        "phi": phi,
    }


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values."""
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def trend_of(s: Series, cfg: IntelConfig, as_of_key: str, window: int | None = None) -> dict[str, Any] | None:
    months, values, _ = s.complete()
    w = window or cfg.trend_window_months
    if len(months) < cfg.trend_min_points:
        return None
    wm = months[-w:]
    wv = values[-w:]
    ok = np.isfinite(wv)
    if ok.sum() < cfg.trend_min_points:
        return None
    t = np.arange(len(wv), dtype=float)[ok]
    x = wv[ok].astype(float)
    tau, p = mann_kendall(x, t)
    if np.allclose(x, x[0]):
        slope = lo = hi = 0.0
    else:
        slope, _, lo, hi = theil_sen(x, t)
    if p < cfg.trend_alpha and lo > 0:
        direction = "up"
    elif p < cfg.trend_alpha and hi < 0:
        direction = "down"
    else:
        direction = "flat"
    mean = float(x.mean())
    prior_vals = values[-2 * w : -w] if len(values) > w else np.array([])
    prior_ok = prior_vals[np.isfinite(prior_vals)]
    prior_mean = float(prior_ok.mean()) if len(prior_ok) >= max(3, cfg.trend_min_points // 2) else None
    cp = change_point(
        x, cfg.change_point_min_segment, cfg.change_point_permutations, sub_seed(cfg.seed, "cp:" + s.id)
    )
    cp_out = None
    if cp is not None and cp["p_value"] < cfg.change_point_alpha:
        pos = np.flatnonzero(ok)[cp["index"]]
        cp_out = {
            "date": month_str(wm[pos]),
            "before_mean": fnum(cp["before_mean"]),
            "after_mean": fnum(cp["after_mean"]),
            "p_value": fnum(cp["p_value"]),
            "null": cp["null"],
            "phi": fnum(cp["phi"], 3),
        }
    return {
        "id": stable_id("trend", "trend", s.id, as_of_key),
        "series_id": s.id,
        "entity": s.entity,
        "entity_type": s.entity_type,
        "metric": s.metric,
        "window": {"start": month_str(wm[0]), "end": month_str(wm[-1]), "months": len(wm)},
        "n_points": int(ok.sum()),
        "direction": direction,
        "slope": fnum(slope),
        "slope_ci": [fnum(lo), fnum(hi)],
        "change_rate_pct_per_month": fnum(100.0 * slope / mean, 4) if abs(mean) > 1e-12 else None,
        "kendall_tau": fnum(tau, 4),
        "p_value": fnum(p),
        "q_value": None,
        "evidence_strength": fnum(1.0 - p, 6),
        "recent_mean": fnum(mean),
        "prior_mean": fnum(prior_mean),
        "ratio": fnum(mean / prior_mean, 4) if prior_mean else None,
        "change_point": cp_out,
    }


def analyse_trends(series: list[Series], cfg: IntelConfig, as_of_key: str) -> list[dict[str, Any]]:
    out = [t for s in series if (t := trend_of(s, cfg, as_of_key)) is not None]
    if out:
        q = bh_qvalues(np.array([t["p_value"] for t in out], dtype=float))
        for t, qv in zip(out, q, strict=True):
            t["q_value"] = fnum(qv)
    return out


def window_label(trend: dict[str, Any]) -> str:
    return f"{trend['window']['start'][:7]}..{trend['window']['end'][:7]}"
