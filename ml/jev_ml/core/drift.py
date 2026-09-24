"""Domain-independent drift tests between an entity's *historical* and *recent* windows.

The windows are split by time (``split_windows``): the recent window holds the entity's last
``recent_n`` events (or, when ``recent_days`` is set, the events of the last ``recent_days`` days
before ``as_of``); the historical window holds every earlier event.

Tests (one per aspect; ``statistic`` is always the aspect's headline effect measure, the raw test
statistic sits in ``effect``):

* ``categorical_shift``: Jensen–Shannon distance (base 2, in [0, 1]) between the category shares
  of the two windows; p from a seeded permutation test that reassigns events to windows (window
  sizes fixed). Each event contributes a total weight of 1, split over its categories.
* ``level_shift``: two-sided Mann–Whitney U; statistic = mean(recent) - mean(historical) with a
  percentile bootstrap CI (windows resampled independently).
* ``rate_shift``: Poisson rate-ratio test, exact conditional form: given the total count
  ``n = c_h + c_r``, ``c_r ~ Binomial(n, e_r / (e_h + e_r))`` under equal rates. statistic = rate
  ratio recent / historical, CI from the Clopper–Pearson interval of the binomial proportion. With
  an overdispersion factor phi > 1 (quasi-Poisson) a Wald test on log(ratio), variance scaled by phi.
* ``median_shift``: statistic = median(recent) - median(historical), bootstrap CI, p from a seeded
  permutation test of the median difference.
* ``centroid_shift``: cosine between the window centroids of (L2-normalised) item vectors; p =
  share of random recent-sized subsets of the pooled history whose centroid is at least as
  dissimilar (a within-history permutation null). Computed from the Gram matrix of the pooled
  events, so each permutation costs O(recent_n²).
* ``proportion_shift``: pooled two-proportion z test; statistic = rate(recent) - rate(historical).

**Session clusters.** Events of one session (e.g. one day) are not independent: a user who rates
twenty horror films in one sitting would make an event-level permutation test flag "drift" whenever
the recent window is a single session. Every permutation test therefore accepts optional cluster
labels; with them, whole clusters are permuted (a cluster permutation test: the recent window's
clusters vs the same number of clusters drawn at random from the pooled history; exact under
exchangeable clusters whatever their sizes), ``split_windows`` extends the
recent window to whole clusters, and a test needs >= ``min_historical_clusters`` historical
clusters. The level test then takes its p from the same cluster permutation of |rank-biserial r|.

Every test has an explicit minimum sample size and returns ``status: "insufficient_data"`` below
it. ``summarize`` applies a Holm correction across the aspects that ran; ``drift_detected`` means at
least one adjusted p <= alpha; ``confidence`` = 1 - (smallest adjusted p), kind ``"evidence"``.
It is a strength-of-evidence score, not a probability that the preferences changed.

Permutation and bootstrap tests use at most ``max_events`` pooled events (all recent events plus a
seeded uniform sample of the historical ones); rank and binomial tests use every event. Every
random draw comes from a generator seeded with ``seed`` and the aspect name, so results are
deterministic and do not depend on the order the aspects are run in.

This module imports only numpy/scipy: nothing domain-specific.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy import stats

OK = "ok"
INSUFFICIENT = "insufficient_data"
CONFIDENCE_KIND = "evidence"
DAY = 86400.0


@dataclass(frozen=True)
class DriftConfig:
    """Window split, sample-size guards and resampling parameters of the drift tests."""

    recent_n: int = 20  # recent window = the last N events ...
    recent_days: float | None = None  # ... or, when set, the events of the last D days
    min_recent: int = 10  # fewer recent events -> the whole report is insufficient_data
    min_historical: int = 20  # fewer historical events -> insufficient_data
    min_span_days: float = 14.0  # all events within fewer days -> drift over time is not measurable
    min_category_events: int = 10  # categorical test: events with >= 1 category, per window
    min_values: int = 8  # level / median tests: values per window
    min_rate_exposure_days: float = 14.0  # rate test: historical exposure
    min_rate_recent_exposure_days: float = 3.0  # rate test: recent exposure
    min_rate_events: int = 5  # rate test: total count
    min_proportion_n: int = 10  # proportion test: trials per window
    min_historical_clusters: int = 8  # session-cluster permutation: historical sessions
    alpha: float = 0.05
    n_permutations: int = 499
    n_bootstrap: int = 499
    ci_level: float = 0.95
    max_events: int = 400
    seed: int = 0


@dataclass
class AspectResult:
    aspect: str
    status: str
    test: str
    statistic: float | None = None
    p_value: float | None = None
    p_adjusted: float | None = None
    significant: bool = False
    effect: dict[str, Any] = field(default_factory=dict)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "status": self.status,
            "test": self.test,
            "statistic": _f(self.statistic),
            "p_value": _f(self.p_value),
            "p_adjusted": _f(self.p_adjusted),
            "significant": bool(self.significant),
            "effect": _clean(self.effect),
            "detail": self.detail,
        }


@dataclass
class WindowSplit:
    """Boolean ``recent`` mask aligned with the input timestamps (historical = ~recent)."""

    recent: np.ndarray
    status: str
    detail: str
    as_of: float
    t_split: float | None  # first recent timestamp
    first: float | None
    last: float | None

    @property
    def historical(self) -> np.ndarray:
        return ~self.recent

    @property
    def n_recent(self) -> int:
        return int(self.recent.sum())

    @property
    def n_historical(self) -> int:
        return int((~self.recent).sum())


# --- helpers ---------------------------------------------------------------------------------------
def _f(x: Any, nd: int = 6) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return _f(obj)
    return obj


def rng_for(seed: int, key: str) -> np.random.Generator:
    """Generator seeded by (seed, key): independent of processing order."""
    return np.random.default_rng((seed * 1_000_003 + zlib.crc32(key.encode())) % (2**32))


def _perm_p(null: np.ndarray, observed: float, greater: bool = True) -> float:
    hits = np.sum(null >= observed - 1e-12) if greater else np.sum(null <= observed + 1e-12)
    return float((1 + hits) / (len(null) + 1))


def _partitions(rng: np.random.Generator, n: int, m: int, b: int) -> np.ndarray:
    """b random permutations of range(n) (rows); the first m columns of a row are a random subset
    of size m and the remaining columns its complement."""
    if m >= n:
        return np.tile(np.arange(n), (b, 1))
    return np.argpartition(rng.random((b, n)), m - 1, axis=1)


def _ci_bounds(cfg: DriftConfig) -> tuple[float, float]:
    tail = (1.0 - cfg.ci_level) / 2.0
    return 100.0 * tail, 100.0 * (1.0 - tail)


def _insufficient(aspect: str, test: str, detail: str, **effect: Any) -> AspectResult:
    return AspectResult(aspect, INSUFFICIENT, test, effect=effect, detail=detail)


@dataclass
class _Null:
    """Pooled events (kept historical ones first, then all recent ones) and the permutation units.

    A *unit* is one event (event-level permutation) or one cluster (e.g. one active day: events of
    a session are not independent, so whole sessions are permuted). ``sel`` holds, per permutation,
    the units that play the recent window; ``obs`` the units of the observed recent window."""

    hist_kept: np.ndarray  # indices into the historical input
    capped: bool
    inv: np.ndarray  # pooled event -> unit id
    n_units: int
    obs: np.ndarray  # unit ids of the observed recent window
    sel: np.ndarray  # (B, width) unit ids; padded with the dummy id n_units (a unit with no events)
    mode: str  # "event" | "cluster"
    n_hist_units: int

    def unit_matrix(self) -> sp.csr_matrix:
        n = len(self.inv)
        return sp.csr_matrix((np.ones(n), (np.arange(n), self.inv)), shape=(n, self.n_units))

    def selection(self) -> np.ndarray:
        cm = np.zeros((len(self.sel), self.n_units + 1), dtype=np.float64)
        np.put_along_axis(cm, self.sel, 1.0, axis=1)
        return cm[:, : self.n_units]

    def observed_selection(self) -> np.ndarray:
        cm = np.zeros(self.n_units, dtype=np.float64)
        cm[self.obs] = 1.0
        return cm

    def event_masks(self) -> np.ndarray:
        return self.selection()[:, self.inv] > 0

    def observed_event_mask(self) -> np.ndarray:
        return self.observed_selection()[self.inv] > 0

    @property
    def label(self) -> str:
        return "session-cluster permutation" if self.mode == "cluster" else "permutation"


def _null(
    n_hist: int,
    n_recent: int,
    cfg: DriftConfig,
    rng: np.random.Generator,
    hist_clusters: np.ndarray | None = None,
    recent_clusters: np.ndarray | None = None,
) -> _Null:
    keep = max(cfg.max_events - n_recent, cfg.min_historical)
    hist_idx = np.arange(n_hist)
    capped = n_hist > keep
    if capped:
        hist_idx = np.sort(rng.choice(hist_idx, size=keep, replace=False))
    n = len(hist_idx) + n_recent
    if hist_clusters is None or recent_clusters is None:
        inv = np.arange(n)
        obs = np.arange(len(hist_idx), n)
        sel = _partitions(rng, n, n_recent, cfg.n_permutations)[:, :n_recent]
        return _Null(hist_idx, capped, inv, n, obs, sel, "event", len(hist_idx))
    hc = np.asarray(hist_clusters)[hist_idx]
    rc = np.asarray(recent_clusters)
    _, h_inv = np.unique(hc, return_inverse=True)
    _, r_inv = np.unique(rc, return_inverse=True)
    k_h = int(h_inv.max()) + 1 if len(h_inv) else 0
    k_r = int(r_inv.max()) + 1 if len(r_inv) else 0
    inv = np.concatenate([h_inv, r_inv + k_h])  # a cluster split by the window boundary counts twice
    k = k_h + k_r
    obs = np.arange(k_h, k)
    # exact cluster permutation: the recent window's k_r clusters vs k_r clusters drawn at random from
    # all k (under H0 clusters are exchangeable, so this null is exact whatever the cluster sizes)
    sel = _partitions(rng, k, k_r, cfg.n_permutations)[:, :k_r]
    return _Null(hist_idx, capped, inv, k, obs, sel, "cluster", k_h)


def _too_few_units(nl: _Null, cfg: DriftConfig, aspect: str, test: str) -> AspectResult | None:
    if nl.mode == "cluster" and nl.n_hist_units < cfg.min_historical_clusters:
        return _insufficient(
            aspect,
            test,
            f"{nl.n_hist_units} historical sessions (need >= {cfg.min_historical_clusters}): session-to-"
            "session variation cannot be separated from drift",
            historical_sessions=nl.n_hist_units,
        )
    return None


def _as_clusters(c: np.ndarray | None, keep: np.ndarray) -> np.ndarray | None:
    return None if c is None else np.asarray(c)[keep]


# --- windows ---------------------------------------------------------------------------------------
def split_windows(
    timestamps: np.ndarray,
    cfg: DriftConfig,
    as_of: float | None = None,
    clusters: np.ndarray | None = None,
) -> WindowSplit:
    """Historical vs recent split by time. Ties are ordered by input position (stable). With
    ``clusters`` (e.g. day ids), a cluster touched by the recent window joins it entirely, so no
    session is split between the windows."""
    ts = np.asarray(timestamps, dtype=np.float64)
    n = len(ts)
    recent = np.zeros(n, dtype=bool)
    if n == 0:
        return WindowSplit(recent, INSUFFICIENT, "no events", float(as_of or 0.0), None, None, None)
    ref = float(as_of) if as_of is not None else float(ts.max())
    order = np.argsort(ts, kind="stable")
    if cfg.recent_days is not None:
        recent = ts > ref - cfg.recent_days * DAY
        rule = f"events of the last {cfg.recent_days:g} days"
    else:
        recent[order[max(0, n - cfg.recent_n) :]] = True
        rule = f"last {cfg.recent_n} events"
    if clusters is not None and recent.any():
        cl = np.asarray(clusters)
        recent |= np.isin(cl, np.unique(cl[recent]))
        rule += ", extended to whole sessions"
    first, last = float(ts.min()), float(ts.max())
    t_split = float(ts[recent].min()) if recent.any() else None
    n_r, n_h = int(recent.sum()), int((~recent).sum())
    span_days = (last - first) / DAY
    problems = []
    if n_r < cfg.min_recent:
        problems.append(f"recent window ({rule}) has {n_r} events (need >= {cfg.min_recent})")
    if n_h < cfg.min_historical:
        problems.append(f"historical window has {n_h} events (need >= {cfg.min_historical})")
    if span_days < cfg.min_span_days:
        problems.append(
            f"all events fall within {span_days:.1f} days (need >= {cfg.min_span_days:g}): "
            "a change over time cannot be separated from the order of one session"
        )
    status = INSUFFICIENT if problems else OK
    detail = "; ".join(problems) if problems else f"recent = {rule}; historical = everything before"
    return WindowSplit(recent, status, detail, ref, t_split, first, last)


# --- tests -----------------------------------------------------------------------------------------
def _shares(w: np.ndarray) -> np.ndarray:
    s = w.sum(axis=-1, keepdims=True)
    return np.divide(w, s, out=np.zeros_like(w), where=s > 0)


def _js(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen–Shannon distance (base 2) along the last axis."""
    m = 0.5 * (p + q)

    def kl(a: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(a > 0, a * np.log2(np.where(a > 0, a, 1.0) / np.where(m > 0, m, 1.0)), 0.0)
        return np.asarray(t.sum(axis=-1))

    return np.sqrt(np.clip(0.5 * kl(p) + 0.5 * kl(q), 0.0, 1.0))


def categorical_shift(
    hist: np.ndarray,
    recent: np.ndarray,
    categories: list[str],
    cfg: DriftConfig,
    aspect: str = "categorical_distribution",
    top: int = 5,
    hist_clusters: np.ndarray | None = None,
    recent_clusters: np.ndarray | None = None,
) -> AspectResult:
    """``hist`` / ``recent``: (events x categories) non-negative weights."""
    h = np.asarray(hist, dtype=np.float64).reshape(-1, len(categories))
    r = np.asarray(recent, dtype=np.float64).reshape(-1, len(categories))
    kh, kr = h.sum(axis=1) > 0, r.sum(axis=1) > 0
    hc, rc = _as_clusters(hist_clusters, kh), _as_clusters(recent_clusters, kr)
    h, r = _shares(h[kh]), _shares(r[kr])
    test = f"Jensen–Shannon distance, {'session-cluster ' if hc is not None else ''}permutation test"
    if len(h) < cfg.min_category_events or len(r) < cfg.min_category_events:
        return _insufficient(
            aspect,
            test,
            f"{len(h)} historical / {len(r)} recent events with a category "
            f"(need >= {cfg.min_category_events} each)",
            n_historical=len(h),
            n_recent=len(r),
        )
    rng = rng_for(cfg.seed, aspect)
    nl = _null(len(h), len(r), cfg, rng, hc, rc)
    if (bad := _too_few_units(nl, cfg, aspect, test)) is not None:
        return bad
    p_h, p_r = _shares(h.sum(axis=0)), _shares(r.sum(axis=0))
    js = float(_js(p_h, p_r))
    pool = np.vstack([h[nl.hist_kept], r])
    units = nl.unit_matrix().T @ pool  # (units x categories)
    total = pool.sum(axis=0)
    rs = nl.selection() @ units
    null = _js(_shares(total - rs), _shares(rs))
    ro = nl.observed_selection() @ units
    p = _perm_p(null, float(_js(_shares(total - ro), _shares(ro))))  # same (capped) pool

    def top_of(v: np.ndarray) -> list[dict[str, Any]]:
        order = np.lexsort((np.arange(len(v)), -v))[:top]
        return [{"category": categories[i], "share": _f(v[i], 4)} for i in order if v[i] > 0]

    diff = p_r - p_h
    movers = np.lexsort((np.arange(len(diff)), -np.abs(diff)))[:3]
    return AspectResult(
        aspect,
        OK,
        test,
        statistic=js,
        p_value=p,
        effect={
            "js_distance": _f(js, 4),
            "historical_top": top_of(p_h),
            "recent_top": top_of(p_r),
            "largest_changes": [
                {
                    "category": categories[i],
                    "historical": _f(p_h[i], 4),
                    "recent": _f(p_r[i], 4),
                    "change": _f(diff[i], 4),
                }
                for i in movers
                if abs(diff[i]) > 0
            ],
            "n_historical": len(h),
            "n_recent": len(r),
            **_null_effect(nl, cfg),
        },
        detail=f"JS distance {js:.3f} between historical and recent category shares",
    )


def _null_effect(nl: _Null, cfg: DriftConfig) -> dict[str, Any]:
    return {
        "permutations": cfg.n_permutations,
        "permutation_unit": "session" if nl.mode == "cluster" else "event",
        "historical_units": nl.n_hist_units,
        "recent_units": len(nl.obs),
        "historical_subsampled_to": len(nl.hist_kept) if nl.capped else None,
    }


def _boot_diff(
    h: np.ndarray, r: np.ndarray, fn: Any, cfg: DriftConfig, rng: np.random.Generator
) -> tuple[float, float]:
    bh = h[rng.integers(0, len(h), (cfg.n_bootstrap, len(h)))]
    br = r[rng.integers(0, len(r), (cfg.n_bootstrap, len(r)))]
    d = fn(br, axis=1) - fn(bh, axis=1)
    lo, hi = np.percentile(d, _ci_bounds(cfg))
    return float(lo), float(hi)


def _finite(
    hist: np.ndarray, recent: np.ndarray, hc: np.ndarray | None, rc: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    h = np.asarray(hist, dtype=np.float64)
    r = np.asarray(recent, dtype=np.float64)
    fh, fr = np.isfinite(h), np.isfinite(r)
    return h[fh], r[fr], _as_clusters(hc, fh), _as_clusters(rc, fr)


def level_shift(
    hist: np.ndarray,
    recent: np.ndarray,
    cfg: DriftConfig,
    aspect: str = "numeric_level",
    hist_clusters: np.ndarray | None = None,
    recent_clusters: np.ndarray | None = None,
) -> AspectResult:
    """Mann–Whitney U. Without clusters: asymptotic two-sided p. With clusters: p from permuting
    whole sessions, on |rank-biserial correlation| (U normalised by the window sizes, which vary
    between permutations)."""
    h, r, hc, rc = _finite(hist, recent, hist_clusters, recent_clusters)
    test = (
        "Mann–Whitney U, session-cluster permutation p, bootstrap CI of the mean shift"
        if hc is not None
        else "Mann–Whitney U (two-sided), bootstrap CI of the mean shift"
    )
    if len(h) < cfg.min_values or len(r) < cfg.min_values:
        return _insufficient(
            aspect,
            test,
            f"{len(h)} historical / {len(r)} recent values (need >= {cfg.min_values} each)",
            n_historical=len(h),
            n_recent=len(r),
        )
    rng = rng_for(cfg.seed, aspect)
    nl = _null(len(h), len(r), cfg, rng, hc, rc)
    if (bad := _too_few_units(nl, cfg, aspect, test)) is not None:
        return bad
    shift = float(r.mean() - h.mean())
    constant = np.ptp(np.concatenate([h, r])) == 0
    if constant:
        u, p = len(h) * len(r) / 2.0, 1.0
    else:
        u = float(stats.mannwhitneyu(r, h, alternative="two-sided", method="asymptotic").statistic)
        if nl.mode == "event":
            p = float(stats.mannwhitneyu(r, h, alternative="two-sided", method="asymptotic").pvalue)
        else:
            pool = np.concatenate([h[nl.hist_kept], r])
            ranks = stats.rankdata(pool)
            a = nl.unit_matrix().T
            unit_ranks, unit_n = a @ ranks, a @ np.ones(len(pool))

            def rb(rank_sum: np.ndarray, m: np.ndarray) -> np.ndarray:
                n = len(pool)
                denom = m * (n - m)
                uu = rank_sum - m * (m + 1) / 2.0
                return np.asarray(np.divide(2.0 * uu, denom, out=np.zeros_like(uu), where=denom > 0) - 1.0)

            sel = nl.selection()
            null = np.abs(rb(sel @ unit_ranks, sel @ unit_n))
            o = nl.observed_selection()
            p = _perm_p(null, float(abs(rb(np.atleast_1d(o @ unit_ranks), np.atleast_1d(o @ unit_n))[0])))
    lo, hi = _boot_diff(h[nl.hist_kept], r, np.mean, cfg, rng)
    return AspectResult(
        aspect,
        OK,
        test,
        statistic=shift,
        p_value=p,
        effect={
            "historical_mean": _f(h.mean(), 4),
            "recent_mean": _f(r.mean(), 4),
            "shift": _f(shift, 4),
            "ci": [_f(lo, 4), _f(hi, 4)],
            "ci_level": cfg.ci_level,
            "u_statistic": _f(u, 2),
            "rank_biserial": _f(2.0 * u / (len(h) * len(r)) - 1.0, 4),
            "n_historical": len(h),
            "n_recent": len(r),
            **(_null_effect(nl, cfg) if nl.mode == "cluster" else {}),
        },
        detail=f"mean {h.mean():.2f} -> {r.mean():.2f} (shift {shift:+.2f}, "
        f"{100 * cfg.ci_level:.0f}% CI {lo:+.2f}..{hi:+.2f})",
    )


def rate_shift(
    hist_count: float,
    hist_exposure: float,
    recent_count: float,
    recent_exposure: float,
    cfg: DriftConfig,
    aspect: str = "event_rate",
    unit: str = "events per day",
    dispersion: float = 1.0,
) -> AspectResult:
    """Counts over exposures (in days). Exact conditional (binomial) Poisson rate-ratio test; with
    ``dispersion`` phi > 1 (quasi-Poisson, estimated by the caller from the historical window), a
    Wald test on log(ratio) with variance phi * (1/c_h + 1/c_r) instead (0.5 added to zero counts),
    and the CI widened the same way."""
    phi = max(1.0, float(dispersion)) if math.isfinite(dispersion) else 1.0
    test = (
        "Poisson rate ratio (exact conditional binomial test)"
        if phi == 1.0
        else "quasi-Poisson rate ratio (Wald test on log ratio, overdispersion from history)"
    )
    ch, cr = round(hist_count), round(recent_count)
    eh, er = float(hist_exposure), float(recent_exposure)
    n = ch + cr
    if eh < cfg.min_rate_exposure_days or er < cfg.min_rate_recent_exposure_days or n < cfg.min_rate_events:
        return _insufficient(
            aspect,
            test,
            f"exposure {eh:.1f} historical / {er:.1f} recent days, {n} events "
            f"(need >= {cfg.min_rate_exposure_days:g} / {cfg.min_rate_recent_exposure_days:g} days "
            f"and >= {cfg.min_rate_events} events)",
            historical_count=ch,
            recent_count=cr,
            historical_exposure_days=eh,
            recent_exposure_days=er,
        )
    rate_h, rate_r = ch / eh, cr / er
    rr = rate_r / rate_h if rate_h > 0 else math.inf
    tail = (1.0 - cfg.ci_level) / 2.0
    if phi == 1.0:
        p_value = float(stats.binomtest(cr, n, er / (eh + er)).pvalue)
        # Clopper–Pearson bounds of the binomial proportion via the beta quantiles
        ci_low = float(stats.beta.ppf(tail, cr, n - cr + 1)) if cr > 0 else 0.0
        ci_high = float(stats.beta.ppf(1.0 - tail, cr + 1, n - cr)) if cr < n else 1.0
        odds = eh / er

        def ratio(pp: float) -> float | None:
            return None if pp >= 1.0 else pp / (1.0 - pp) * odds

        ci = [ratio(ci_low), ratio(ci_high)]
    else:
        a, b = max(ch, 0.5), max(cr, 0.5)
        log_rr = math.log((b / er) / (a / eh))
        se = math.sqrt(phi * (1.0 / a + 1.0 / b))
        p_value = float(2.0 * stats.norm.sf(abs(log_rr) / se))
        zc = float(stats.norm.ppf(1.0 - tail))
        ci = [math.exp(log_rr - zc * se), math.exp(log_rr + zc * se)]
    return AspectResult(
        aspect,
        OK,
        test,
        statistic=rr,
        p_value=p_value,
        effect={
            "historical_rate": _f(rate_h, 5),
            "recent_rate": _f(rate_r, 5),
            "unit": unit,
            "rate_ratio": _f(rr, 4),
            "ci": [_f(ci[0], 4), _f(ci[1], 4)],
            "ci_level": cfg.ci_level,
            "dispersion": _f(phi, 4),
            "historical_count": ch,
            "recent_count": cr,
            "historical_exposure_days": _f(eh, 2),
            "recent_exposure_days": _f(er, 2),
        },
        detail=f"{unit}: {rate_h:.3f} -> {rate_r:.3f} (ratio {rr:.2f})",
    )


def _masked_medians(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Median of values[mask_row] for every row, in O(B n) via a single sort of the values."""
    order = np.argsort(values, kind="stable")
    v = values[order]
    cs = np.cumsum(masks[:, order], axis=1)
    m = cs[:, -1]
    lo_k = (m + 1) // 2  # 1-based rank of the lower middle element
    hi_k = m // 2 + 1
    lo = v[np.argmax(cs >= lo_k[:, None], axis=1)]
    hi = v[np.argmax(cs >= hi_k[:, None], axis=1)]
    return np.asarray(np.where(m > 0, 0.5 * (lo + hi), np.nan))


def median_shift(
    hist: np.ndarray,
    recent: np.ndarray,
    cfg: DriftConfig,
    aspect: str = "numeric_attribute",
    hist_clusters: np.ndarray | None = None,
    recent_clusters: np.ndarray | None = None,
) -> AspectResult:
    h, r, hc, rc = _finite(hist, recent, hist_clusters, recent_clusters)
    test = f"median shift, bootstrap CI + {'session-cluster ' if hc is not None else ''}permutation test"
    if len(h) < cfg.min_values or len(r) < cfg.min_values:
        return _insufficient(
            aspect,
            test,
            f"{len(h)} historical / {len(r)} recent values (need >= {cfg.min_values} each)",
            n_historical=len(h),
            n_recent=len(r),
        )
    rng = rng_for(cfg.seed, aspect)
    nl = _null(len(h), len(r), cfg, rng, hc, rc)
    if (bad := _too_few_units(nl, cfg, aspect, test)) is not None:
        return bad
    shift = float(np.median(r) - np.median(h))
    hb = h[nl.hist_kept]
    pool = np.concatenate([hb, r])
    masks = nl.event_masks()
    null = np.abs(_masked_medians(pool, masks) - _masked_medians(pool, ~masks))
    p = _perm_p(null, abs(float(np.median(r) - np.median(hb))))
    lo, hi = _boot_diff(hb, r, np.median, cfg, rng)
    return AspectResult(
        aspect,
        OK,
        test,
        statistic=shift,
        p_value=p,
        effect={
            "historical_median": _f(np.median(h), 4),
            "recent_median": _f(np.median(r), 4),
            "shift": _f(shift, 4),
            "ci": [_f(lo, 4), _f(hi, 4)],
            "ci_level": cfg.ci_level,
            "n_historical": len(h),
            "n_recent": len(r),
            **_null_effect(nl, cfg),
        },
        detail=f"median {np.median(h):.5g} -> {np.median(r):.5g} (shift {shift:+.3g}, "
        f"{100 * cfg.ci_level:.0f}% CI {lo:+.3g}..{hi:+.3g})",
    )


def _unit_rows(x: Any) -> Any:
    if sp.issparse(x):
        x = sp.csr_matrix(x, dtype=np.float64)
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel())
        return sp.diags(np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)) @ x
    x = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


def centroid_shift(
    hist_vectors: Any,
    recent_vectors: Any,
    cfg: DriftConfig,
    aspect: str = "content_similarity",
    hist_clusters: np.ndarray | None = None,
    recent_clusters: np.ndarray | None = None,
) -> AspectResult:
    """Rows = events (dense array or scipy sparse); rows are L2-normalised here. The null draws
    recent-sized sets of units (events or sessions) from the pooled history, via the unit-level
    Gram matrix, so each permutation costs O(recent_units²)."""
    test = (
        "centroid cosine vs within-history "
        f"{'session-cluster ' if hist_clusters is not None else ''}permutation null"
    )
    nh, nr = hist_vectors.shape[0], recent_vectors.shape[0]
    if nh < cfg.min_historical or nr < cfg.min_recent:
        return _insufficient(
            aspect,
            test,
            f"{nh} historical / {nr} recent vectors (need >= {cfg.min_historical} / {cfg.min_recent})",
            n_historical=nh,
            n_recent=nr,
        )
    rng = rng_for(cfg.seed, aspect)
    nl = _null(nh, nr, cfg, rng, hist_clusters, recent_clusters)
    if (bad := _too_few_units(nl, cfg, aspect, test)) is not None:
        return bad
    hi = nl.hist_kept
    if sp.issparse(hist_vectors):
        pool = _unit_rows(sp.vstack([sp.csr_matrix(hist_vectors)[hi], sp.csr_matrix(recent_vectors)]).tocsr())
        g = np.asarray((pool @ pool.T).toarray())
    else:
        pool_d = _unit_rows(np.vstack([np.asarray(hist_vectors)[hi], np.asarray(recent_vectors)]))
        g = pool_d @ pool_d.T
    if nl.mode == "cluster":
        a = nl.unit_matrix()
        g = np.asarray(a.T @ (a.T @ g).T)  # unit Gram A^T G A (G symmetric)
    col = g.sum(axis=0)
    total = float(col.sum())
    g = np.pad(g, ((0, 1), (0, 1)))  # the dummy padding unit contributes nothing
    col = np.append(col, 0.0)

    def cosine(idx: np.ndarray) -> np.ndarray:
        """Centroid cosine for each row of recent unit sets (complement = historical)."""
        s_rr = g[idx[..., :, None], idx[..., None, :]].sum(axis=(-1, -2))
        c_r = col[idx].sum(axis=-1)
        s_hr = c_r - s_rr
        s_hh = total - 2.0 * c_r + s_rr
        denom = np.sqrt(np.maximum(s_hh, 0.0) * np.maximum(s_rr, 0.0))
        return np.asarray(np.divide(s_hr, denom, out=np.zeros_like(s_hr), where=denom > 0))

    obs = float(cosine(nl.obs[None, :])[0])
    null = cosine(nl.sel)
    p = _perm_p(null, obs, greater=False)
    return AspectResult(
        aspect,
        OK,
        test,
        statistic=obs,
        p_value=p,
        effect={
            "cosine": _f(obs, 4),
            "null_mean_cosine": _f(null.mean(), 4),
            "null_p05_cosine": _f(np.percentile(null, 5), 4),
            "n_historical": nh,
            "n_recent": nr,
            **_null_effect(nl, cfg),
        },
        detail=f"centroid cosine {obs:.3f} (random recent-sized sets of the history: {null.mean():.3f})",
    )


def proportion_shift(
    hist_success: int,
    hist_n: int,
    recent_success: int,
    recent_n: int,
    cfg: DriftConfig,
    aspect: str = "proportion",
) -> AspectResult:
    test = "two-proportion z test (pooled)"
    if hist_n < cfg.min_proportion_n or recent_n < cfg.min_proportion_n:
        return _insufficient(
            aspect,
            test,
            f"{hist_n} historical / {recent_n} recent trials (need >= {cfg.min_proportion_n} each)",
            n_historical=hist_n,
            n_recent=recent_n,
        )
    ph, pr = hist_success / hist_n, recent_success / recent_n
    pooled = (hist_success + recent_success) / (hist_n + recent_n)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / hist_n + 1 / recent_n))
    z = (pr - ph) / se0 if se0 > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z))) if se0 > 0 else 1.0
    se = math.sqrt(ph * (1 - ph) / hist_n + pr * (1 - pr) / recent_n)
    zc = float(stats.norm.ppf(0.5 + cfg.ci_level / 2))
    return AspectResult(
        aspect,
        OK,
        test,
        statistic=pr - ph,
        p_value=p,
        effect={
            "historical_rate": _f(ph, 4),
            "recent_rate": _f(pr, 4),
            "difference": _f(pr - ph, 4),
            "ci": [_f(pr - ph - zc * se, 4), _f(pr - ph + zc * se, 4)],
            "ci_level": cfg.ci_level,
            "z": _f(z, 4),
            "n_historical": hist_n,
            "n_recent": recent_n,
        },
        detail=f"rate {ph:.2f} -> {pr:.2f} ({recent_n} recent / {hist_n} historical trials)",
    )


# --- multiplicity + summary -----------------------------------------------------------------------
def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values (None entries are skipped and stay None)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    out: list[float | None] = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(sorted(idx, key=lambda j: (pvalues[j], j))):
        p = pvalues[i]
        assert p is not None
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def summarize(aspects: list[AspectResult], cfg: DriftConfig) -> dict[str, Any]:
    """Holm-correct the aspects that ran (in place) and return the report-level fields."""
    ran = [a for a in aspects if a.status == OK and a.p_value is not None]
    adj = holm([a.p_value for a in ran])
    for a, pa in zip(ran, adj, strict=True):
        a.p_adjusted = pa
        a.significant = pa is not None and pa <= cfg.alpha
    if not ran:
        return {
            "status": INSUFFICIENT,
            "drift_detected": False,
            "confidence": None,
            "confidence_kind": CONFIDENCE_KIND,
        }
    min_adj = min(a.p_adjusted for a in ran if a.p_adjusted is not None)
    return {
        "status": OK,
        "drift_detected": any(a.significant for a in ran),
        "confidence": _f(1.0 - min_adj, 4),
        "confidence_kind": CONFIDENCE_KIND,
    }
