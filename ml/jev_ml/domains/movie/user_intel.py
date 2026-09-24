"""Movie user intelligence: preference drift, the recommendation-strategy decision, and
recommendations served downstream of that decision (``GET /me/intelligence``, docs/platform.md
§4, §5, §10).

Events
------
Every interaction (rating, favourite, watch, onboarding pick) of a movie known to the model counts
as one *event* per movie: the movie's first engagement time is its timestamp and its latest rating
(if any) its rating. Events after ``as_of`` are ignored; ``as_of`` defaults to the latest event.

Drift aspects (``jev_ml.core.drift``)
-------------------------------------
genre_distribution (JS distance of consumption genre shares, permutation test), rating_level
(Mann–Whitney), activity_rate (Poisson rate ratio of *active days*; single-day bursts of dozens of
ratings would violate the Poisson assumption for raw event counts), release_year (median shift),
content_similarity (cosine of TF-IDF centroids of the content model vs a within-history permutation
null) and acceptance (two-proportion test on recommendation feedback, when provided). Holm across
the aspects that ran.

Strategy decision (policy ``strategy-1.0.0``, margin confidence)
--------------------------------------------------------------
Evidence is measured in "bans": e(p) = min(cap, -log10 p). ``standard`` scores the constant
c = -log10(alpha) (the significance threshold). The preference-drift evidence
e_pref = e(smallest Holm-adjusted p over the preference aspects genre_distribution,
content_similarity, release_year, rating_level) goes to exactly one alternative:

* ``adapt_to_recent`` if the offline evaluation (scripts/evaluate_drift.py, recorded in
  ``EVALUATED_EFFECTS``) showed that adapting *helps* drifting users: the lower end of the paired
  bootstrap CI of the NDCG@10 difference vs standard is > ``adapt_min_gain``;
* otherwise ``explore`` if the evaluation showed that exploring costs at most
  ``explore_max_ndcg_cost`` NDCG@10 (CI lower end >= -cost) for drifting users;
* otherwise to neither: a detected drift is reported, but ``standard`` is kept because no
  alternative has demonstrated a benefit (stated in the rationale).

A significant *drop* in recommendation acceptance adds e(p_adj of acceptance) to ``explore``.
The answer is the arg-max; because c is the significance threshold, a non-standard answer requires a
significant (Holm-adjusted) aspect. Option scores are normalised to sum 1; confidence = (best -
second) / (best + second), kind ``margin`` (not a probability). More drift evidence can never lower
an alternative's score (monotone). The decision abstains (answer null, fallback ``standard``) when
the drift report is ``insufficient_data``.

Strategy parameters (``StrategyConfig``): ``adapt_to_recent`` = the profile's recency decay with
half-life = ``adapt_half_life_factor`` x the recent window's span, clamped to
[``adapt_half_life_min_days``, ``adapt_half_life_max_days``], and weight floor
``adapt_weight_floor`` (weight x (floor + (1 - floor) x 0.5^(age / half-life)), the same form as
``UserProfile.from_events``); ``explore`` = MMR diversity λ lowered to ``explore_lambda``.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import time
import weakref
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from jev_ml.core.common import evidence, fnum, iso_from_epoch, short_hash, stable_id
from jev_ml.core.drift import (
    DAY,
    INSUFFICIENT,
    OK,
    AspectResult,
    DriftConfig,
    WindowSplit,
    categorical_shift,
    centroid_shift,
    level_shift,
    median_shift,
    proportion_shift,
    rate_shift,
    split_windows,
    summarize,
)
from jev_ml.engine import Interaction, RecommendationEngine
from jev_ml.signals import UserProfile

DOMAIN = "movie"
SPEC_ID = "recommendation_strategy"
STRATEGIES = ("standard", "adapt_to_recent", "explore")
ASPECTS = (
    "genre_distribution",
    "rating_level",
    "activity_rate",
    "release_year",
    "content_similarity",
    "acceptance",
)
QUESTION = "How should this user's recommendations be produced?"
ACCEPTED = {"accepted", "like", "clicked", "accept"}
REJECTED = {"rejected", "dislike", "not_interested", "reject"}
DECADE_PREFIX = "y:"
DISPERSION_BLOCK_DAYS = 28
YEAR_SECONDS = 365.2425 * DAY

# Measured by scripts/evaluate_drift.py; see docs/platform.md "Implementation notes (preference
# drift & user intelligence)". Paired bootstrap 95 % CI of mean per-user NDCG@10 (strategy -
# standard) on users whose train history shows drift, models refitted on train+validation only.
# Only 7 users qualified (most MovieLens histories are one or two sessions), so neither strategy
# meets ``evidence_min_users`` and drifting users keep ``standard`` until more evidence exists.
_EVAL_PROTOCOL = (
    "components refitted on train+validation; per-user temporal split; relevance = test rating >= 4; "
    "users with preference drift in their train+validation history"
)
_EVAL_SOURCE = "experiments/drift-eval-20260924T045528Z/report.json"
EVALUATED_EFFECTS: dict[str, dict[str, Any]] = {
    "adapt_to_recent": {
        "delta_ndcg10": 0.00633,
        "ci95": [0.00138, 0.01343],
        "n_users": 7,
        "n_improved": 3,
        "n_worse": 0,
        "protocol": _EVAL_PROTOCOL,
        "source": _EVAL_SOURCE,
    },
    "explore": {
        "delta_ndcg10": 0.00277,
        "ci95": [-0.00144, 0.00771],
        "n_users": 7,
        "n_improved": 2,
        "n_worse": 1,
        "protocol": _EVAL_PROTOCOL,
        "source": _EVAL_SOURCE,
    },
}


@dataclass(frozen=True)
class StrategyConfig:
    drift: DriftConfig = field(default_factory=DriftConfig)
    policy_version: str = "strategy-1.0.0"
    session_clusters: bool = True  # permutation tests permute whole active days (see drift_aspects)
    preference_aspects: tuple[str, ...] = (
        "genre_distribution",
        "content_similarity",
        "release_year",
        "rating_level",
    )
    evidence_cap: float = 4.0  # bans; permutation p-values bottom out at 1/(B+1) anyway
    adapt_half_life_factor: float = 1.0
    adapt_half_life_min_days: float = 14.0
    adapt_half_life_max_days: float = 365.0
    adapt_weight_floor: float = 0.5
    explore_lambda: float = 0.6
    adapt_min_gain: float = 0.0  # adapt needs CI lower end of ΔNDCG@10 > this
    explore_max_ndcg_cost: float = 0.01  # explore needs CI lower end of ΔNDCG@10 >= -this
    evidence_min_users: int = 30  # ... measured on at least this many drifting users
    evaluated_effects: Mapping[str, Mapping[str, Any]] = field(default_factory=lambda: EVALUATED_EFFECTS)
    n_time_buckets: int = 4
    history_top_categories: int = 8


# --- per-engine item features (cached) --------------------------------------------------------------
@dataclass
class _ItemFeatures:
    genres: list[str]
    genre_matrix: np.ndarray  # items x genres, float32 0/1
    years: np.ndarray  # float64, NaN unknown
    vectors: sp.csr_matrix  # content item vectors


_FEATURES: weakref.WeakKeyDictionary[RecommendationEngine, _ItemFeatures] = weakref.WeakKeyDictionary()


def item_features(engine: RecommendationEngine) -> _ItemFeatures:
    f = _FEATURES.get(engine)
    if f is None:
        gm = engine.ranker.genre_matrix
        dense = gm.toarray() if sp.issparse(gm) else np.asarray(gm)
        years = pd.to_numeric(engine.movies["year"], errors="coerce").to_numpy(dtype=np.float64)
        vectors = sp.csr_matrix(engine.content.item_vectors, dtype=np.float32)
        names = list(getattr(engine.content, "_feature_names", None) or [])
        if len(names) == vectors.shape[1]:
            # drop the release-decade tokens: centroid drift should not re-measure the calendar
            keep = np.asarray([not n.startswith(DECADE_PREFIX) for n in names], dtype=np.float32)
            vectors = sp.csr_matrix(vectors @ sp.diags(keep))
            vectors.eliminate_zeros()
        f = _ItemFeatures(list(engine.ranker.genres), (dense > 0).astype(np.float32), years, vectors)
        _FEATURES[engine] = f
    return f


# --- inputs ------------------------------------------------------------------------------------------
def _epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return float(ts.timestamp())


@dataclass
class UserEvents:
    items: np.ndarray  # item indices (one per movie)
    ts: np.ndarray
    rating: np.ndarray  # NaN when the movie was never rated
    as_of: float


def user_events(
    engine: RecommendationEngine, interactions: Iterable[Interaction], as_of: Any = None
) -> UserEvents:
    cutoff = _epoch(as_of)
    per: dict[int, list[float]] = {}  # idx -> [first ts, rating, rating ts]
    for it in interactions:
        if cutoff is not None and it.timestamp > cutoff:
            continue
        idx = engine.item_index.index_of(it.movie_id)
        if idx is None:
            continue
        cur = per.setdefault(idx, [it.timestamp, math.nan, -math.inf])
        cur[0] = min(cur[0], it.timestamp)
        if it.kind == "rating" and it.value is not None and it.timestamp >= cur[2]:
            cur[1], cur[2] = float(it.value), it.timestamp
    keys = sorted(per, key=lambda i: (per[i][0], i))
    ts = np.asarray([per[i][0] for i in keys], dtype=np.float64)
    ref = cutoff if cutoff is not None else (float(ts.max()) if len(ts) else time.time())
    return UserEvents(
        np.asarray(keys, dtype=np.int64),
        ts,
        np.asarray([per[i][1] for i in keys], dtype=np.float64),
        float(ref),
    )


def _as_of_key(as_of: float, n_events: int) -> str:
    """Id key for decisions and signals. With no events the reference time is the wall clock, so it is left
    out: otherwise the same empty profile would get a new decision id every second."""
    return f"{iso_from_epoch(as_of)}|{n_events}" if n_events else "no-events|0"


def _feedback_rows(feedback: Iterable[Any] | None, as_of: float) -> list[tuple[float, bool]]:
    out: list[tuple[float, bool]] = []
    for fb in feedback or ():
        get = fb.get if isinstance(fb, Mapping) else lambda k, fb=fb: getattr(fb, k, None)
        ts = _epoch(get("timestamp") or get("created_at"))
        if ts is None or ts > as_of:
            continue
        acc = get("accepted")
        if acc is None:
            label = str(get("verdict") or get("feedback") or "").lower()
            acc = True if label in ACCEPTED else False if label in REJECTED else None
        if acc is not None:
            out.append((ts, bool(acc)))
    return out


# --- drift report ------------------------------------------------------------------------------------
def _window(ts: np.ndarray) -> dict[str, Any]:
    if len(ts) == 0:
        return {"start": None, "end": None, "n": 0}
    return {"start": iso_from_epoch(ts.min()), "end": iso_from_epoch(ts.max()), "n": len(ts)}


def _activity(ev: UserEvents, split: WindowSplit, cfg: DriftConfig) -> AspectResult:
    """Rate of active days. Both windows *start* on an active day by construction (the first
    event; the first recent event), and the recent window also *ends* on one when ``as_of`` is the
    last event. Counting those boundary days would inflate short windows (a recent window of 3
    days always holds >= 2 active days), so boundary days are excluded from count and exposure."""
    days = np.floor(ev.ts / DAY)
    assert split.t_split is not None
    split_day = math.floor(split.t_split / DAY)
    first_day, as_of_day = float(days.min()), math.floor(ev.as_of / DAY)
    end_is_event = ev.as_of <= float(ev.ts.max())
    active = np.unique(days)
    hist_active = int(((active > first_day) & (active < split_day)).sum())
    upper = (active < as_of_day) if end_is_event else (active <= as_of_day)
    recent_active = int(((active > split_day) & upper).sum())
    res = rate_shift(
        hist_active,
        max(0.0, split_day - first_day - 1),
        recent_active,
        max(0.0, as_of_day - split_day - (1 if end_is_event else 0)),
        cfg,
        aspect="activity_rate",
        unit="active days per day",
        dispersion=_dispersion(active[(active > first_day) & (active < split_day)], first_day + 1, split_day),
    )
    res.effect["boundary_days_excluded"] = True
    res.effect["dispersion_block_days"] = DISPERSION_BLOCK_DAYS
    return res


def _dispersion(active_days: np.ndarray, start: float, end: float) -> float:
    """Quasi-Poisson overdispersion of active days: Pearson chi2 / df of the counts per
    DISPERSION_BLOCK_DAYS-day block of the historical window (>= 1; 1 with fewer than 3 blocks)."""
    n_blocks = int((end - start) // DISPERSION_BLOCK_DAYS)
    if n_blocks < 3:
        return 1.0
    edges = start + DISPERSION_BLOCK_DAYS * np.arange(n_blocks + 1)
    counts = np.histogram(active_days, bins=edges)[0].astype(np.float64)
    mean = counts.mean()
    if mean <= 0:
        return 1.0
    return float(max(1.0, ((counts - mean) ** 2).sum() / mean / (n_blocks - 1)))


def _film_age(
    years: np.ndarray,
    ev: UserEvents,
    h: np.ndarray,
    r: np.ndarray,
    cfg: DriftConfig,
    clusters: dict[str, np.ndarray],
) -> AspectResult:
    """release_year aspect, measured as the film's age when the user engaged with it.

    Raw release years drift for every long-lived user simply because the calendar moves (a viewer
    in 2018 watches 2017 releases). The age at engagement (event year - release year) keeps the
    preference (new releases vs catalogue titles) and removes that confound."""
    event_year = 1970.0 + ev.ts / YEAR_SECONDS
    age = event_year - years[ev.items]
    res = median_shift(age[h], age[r], cfg, "release_year", **clusters)
    yh, yr = years[ev.items[h]], years[ev.items[r]]
    yh, yr = yh[np.isfinite(yh)], yr[np.isfinite(yr)]
    res.effect["measure"] = "film age at engagement in years (event year - release year)"
    res.effect["historical_median_release_year"] = float(np.median(yh)) if len(yh) else None
    res.effect["recent_median_release_year"] = float(np.median(yr)) if len(yr) else None
    if res.status == OK:
        res.detail = f"film age when watched: {res.detail}"
    return res


def drift_aspects(
    engine: RecommendationEngine,
    ev: UserEvents,
    feedback: Iterable[Any] | None,
    cfg: DriftConfig,
    sessions: bool = True,
) -> tuple[list[AspectResult], WindowSplit]:
    """The six aspects. ``sessions``: permute whole active days (UTC) instead of single events."""
    days = np.floor(ev.ts / DAY).astype(np.int64)
    split = split_windows(ev.ts, cfg, ev.as_of, clusters=days if sessions else None)
    if split.status != OK:
        return [AspectResult(a, INSUFFICIENT, "not run", detail=split.detail) for a in ASPECTS], split
    f = item_features(engine)
    h, r = split.historical, split.recent
    ih, ir = ev.items[h], ev.items[r]
    cl = {"hist_clusters": days[h], "recent_clusters": days[r]} if sessions else {}
    out = [
        categorical_shift(f.genre_matrix[ih], f.genre_matrix[ir], f.genres, cfg, "genre_distribution", **cl),
        level_shift(ev.rating[h], ev.rating[r], cfg, "rating_level", **cl),
        _activity(ev, split, cfg),
        _film_age(f.years, ev, h, r, cfg, cl),
        centroid_shift(f.vectors[ih], f.vectors[ir], cfg, "content_similarity", **cl),
    ]
    rows = _feedback_rows(feedback, ev.as_of)
    if not rows:
        out.append(
            AspectResult(
                "acceptance",
                INSUFFICIENT,
                "two-proportion z test (pooled)",
                detail="no recommendation feedback provided",
            )
        )
    else:
        assert split.t_split is not None
        hs = [a for t, a in rows if t < split.t_split]
        rs = [a for t, a in rows if t >= split.t_split]
        out.append(proportion_shift(sum(hs), len(hs), sum(rs), len(rs), cfg, "acceptance"))
    return out, split


_ASPECT_LABEL = {
    "genre_distribution": "genre mix",
    "rating_level": "rating level",
    "activity_rate": "activity",
    "release_year": "release years",
    "content_similarity": "content profile",
    "acceptance": "recommendation acceptance",
}


def insufficient_reason(aspects: list[AspectResult], split: WindowSplit) -> str:
    if split.status != OK:
        return split.detail
    reasons = dict.fromkeys(a.detail for a in aspects if a.status != OK and a.aspect != "acceptance")
    return "; ".join(reasons) or split.detail


def _summary(aspects: list[AspectResult], head: dict[str, Any], split: WindowSplit) -> str:
    if head["status"] != OK:
        return f"Not enough history to test for drift: {insufficient_reason(aspects, split)}."
    sig = [a for a in aspects if a.significant]
    ran = [a for a in aspects if a.status == OK]
    if not sig:
        best = min(a.p_adjusted for a in ran if a.p_adjusted is not None)
        return f"No drift detected across {len(ran)} tested aspects (smallest Holm-adjusted p = {best:.3f})."
    parts = [f"{_ASPECT_LABEL[a.aspect]} ({a.detail}; adjusted p = {a.p_adjusted:.3g})" for a in sig]
    return f"Recent behaviour differs from history in {len(sig)} of {len(ran)} aspects: " + "; ".join(parts)


def drift_report(
    engine: RecommendationEngine,
    ev: UserEvents,
    feedback: Iterable[Any] | None,
    cfg: DriftConfig,
    sessions: bool = True,
) -> tuple[dict[str, Any], list[AspectResult], WindowSplit]:
    aspects, split = drift_aspects(engine, ev, feedback, cfg, sessions)
    head = summarize(aspects, cfg)
    ev_list = [evidence("record", "window rule", split.detail, f"alpha {cfg.alpha}, Holm correction")]
    for a in aspects:
        if a.significant:
            ev_list.append(evidence("test", f"{a.aspect}: {a.test}", fnum(a.p_adjusted, 6), a.detail))
    ran = [a for a in aspects if a.status == OK and a.p_adjusted is not None]
    if ran and not any(a.significant for a in ran):
        best = min(ran, key=lambda a: a.p_adjusted or 1.0)
        ev_list.append(
            evidence("test", f"strongest aspect: {best.aspect}", fnum(best.p_adjusted, 6), best.detail)
        )
    report = {
        **head,
        "historical_window": _window(ev.ts[split.historical]),
        "recent_window": _window(ev.ts[split.recent]),
        "aspects": [a.to_dict() for a in aspects],
        "summary": _summary(aspects, head, split),
        "evidence": ev_list,
    }
    return report, aspects, split


# --- strategy decision --------------------------------------------------------------------------------
def _bans(p: float | None, cap: float) -> float:
    if p is None:
        return 0.0
    return float(min(cap, -math.log10(max(p, 10.0**-cap))))


def adapt_half_life_days(split: WindowSplit, cfg: StrategyConfig) -> float:
    span = 0.0 if split.t_split is None or split.last is None else (split.last - split.t_split) / DAY
    return float(
        np.clip(cfg.adapt_half_life_factor * span, cfg.adapt_half_life_min_days, cfg.adapt_half_life_max_days)
    )


def _eligibility(cfg: StrategyConfig) -> tuple[bool, bool, list[str]]:
    notes = []
    ad = cfg.evaluated_effects.get("adapt_to_recent")
    ex = cfg.evaluated_effects.get("explore")
    adapt_ok = bool(ad) and ad is not None and ad["n_users"] >= cfg.evidence_min_users
    adapt_ok = adapt_ok and ad is not None and ad["ci95"][0] > cfg.adapt_min_gain
    explore_ok = bool(ex) and ex is not None and ex["n_users"] >= cfg.evidence_min_users
    explore_ok = explore_ok and ex is not None and ex["ci95"][0] >= -cfg.explore_max_ndcg_cost
    for name, e, ok in (("adapt_to_recent", ad, adapt_ok), ("explore", ex, explore_ok)):
        if not e:
            notes.append(f"{name}: no offline evaluation recorded, so it is not eligible for drift")
        else:
            notes.append(
                f"{name}: evaluated ΔNDCG@10 vs standard {e['delta_ndcg10']:+.4f} "
                f"(95% CI {e['ci95'][0]:+.4f}..{e['ci95'][1]:+.4f}, {e['n_users']} drifting users) "
                f"-> {'eligible' if ok else 'not eligible'}"
                + (
                    ""
                    if e["n_users"] >= cfg.evidence_min_users
                    else f" (needs >= {cfg.evidence_min_users} users)"
                )
            )
    return adapt_ok, explore_ok, notes


def _confidence_stats(engine: RecommendationEngine, profile: UserProfile, k: int) -> dict[str, Any]:
    cal = getattr(engine, "_calibrator", None)
    if cal is None or profile.n_interactions == 0:
        return {"mean": None, "min": None, "n": 0, "kind": None, "detail": "model has no calibration"}
    ranked = engine.ranker.rank(profile, k=k, explain=False)
    vals = [
        v
        for i, r in enumerate(ranked)
        if (v := cal.predict(r.score, i + 1, profile.n_interactions)) is not None
    ]
    if not vals:
        return {"mean": None, "min": None, "n": 0, "kind": None, "detail": "outside the calibrated range"}
    return {
        "mean": fnum(np.mean(vals), 4),
        "min": fnum(np.min(vals), 4),
        "n": len(vals),
        "kind": "probability",
        "detail": f"calibrated P(rating >= 4) of the top-{k} standard recommendations",
    }


def _decision(
    entity: str,
    as_of_key: str,
    answer: str | None,
    scores: dict[str, float],
    confidence: float | None,
    state: dict[str, Any],
    rationale: list[str],
    ev: list[dict[str, Any]],
    policy: str,
    fallback: str | None,
) -> dict[str, Any]:
    """Same wire shape as jev_ml.core.decisions.decision (+ ``domain``)."""
    return {
        "id": stable_id("dec", SPEC_ID, entity, as_of_key),
        "key": SPEC_ID,
        "spec_id": SPEC_ID,
        "policy_version": policy,
        "question": QUESTION,
        "kind": "choice",
        "options": list(STRATEGIES),
        "answer": answer,
        "option_scores": {k: fnum(v, 4) for k, v in scores.items()},
        "confidence": fnum(confidence, 4),
        "confidence_kind": "margin",
        "state": state,
        "rationale": rationale,
        "evidence": ev,
        "abstained": fallback is not None,
        "fallback_reason": fallback,
        "entity_type": "user",
        "entity": entity,
        "scale": None,
        "answer_interval": None,
        "batch_id": None,
        "domain": DOMAIN,
    }


def strategy_decision(
    report: dict[str, Any],
    aspects: list[AspectResult],
    split: WindowSplit,
    n_events: int,
    conf_stats: dict[str, Any],
    cfg: StrategyConfig,
    entity: str,
    as_of: float,
    standard_lambda: float,
) -> dict[str, Any]:
    by = {a.aspect: a for a in aspects}
    alpha = cfg.drift.alpha
    c = -math.log10(alpha)
    half_life = adapt_half_life_days(split, cfg)
    adapt_ok, explore_ok, elig_notes = _eligibility(cfg)
    params = {
        "adapt_half_life_days": fnum(half_life, 2),
        "adapt_weight_floor": cfg.adapt_weight_floor,
        "explore_lambda": cfg.explore_lambda,
        "standard_lambda": standard_lambda,
    }
    base_state: dict[str, Any] = {
        "drift_status": report["status"],
        "drift_detected": report["drift_detected"],
        "drift_confidence": report["confidence"],
        "profile_events": n_events,
        "historical_events": split.n_historical,
        "recent_events": split.n_recent,
        "recommendation_confidence": conf_stats,
        "parameters": params,
        "evaluated_effects": {k: dict(v) for k, v in cfg.evaluated_effects.items()},
    }
    as_of_key = _as_of_key(as_of, n_events)
    if report["status"] != OK:
        base_state["served_strategy"] = "standard"
        base_state["state_hash"] = short_hash(base_state)
        return _decision(
            entity,
            as_of_key,
            None,
            {},
            None,
            base_state,
            [f"drift analysis needs more history: {insufficient_reason(aspects, split)}"],
            report["evidence"],
            cfg.policy_version,
            f"insufficient history for drift analysis ({insufficient_reason(aspects, split)}); "
            "serving standard",
        )
    pref = [by[a] for a in cfg.preference_aspects if a in by and by[a].status == OK]
    p_pref = min((a.p_adjusted for a in pref if a.p_adjusted is not None), default=None)
    e_pref = _bans(p_pref, cfg.evidence_cap)
    acc = by.get("acceptance")
    e_acc = 0.0
    if acc is not None and acc.status == OK and (acc.statistic or 0.0) < 0:
        e_acc = _bans(acc.p_adjusted, cfg.evidence_cap)
    route = "adapt_to_recent" if adapt_ok else "explore" if explore_ok else None
    raw = {
        "standard": c,
        "adapt_to_recent": e_pref if route == "adapt_to_recent" else 0.0,
        "explore": max(e_pref if route == "explore" else 0.0, e_acc),
    }
    order = sorted(STRATEGIES, key=lambda s: (-raw[s], STRATEGIES.index(s)))
    best, second = order[0], order[1]
    total = sum(raw.values())
    scores = {s: raw[s] / total for s in STRATEGIES}
    conf = (raw[best] - raw[second]) / (raw[best] + raw[second]) if raw[best] + raw[second] > 0 else 0.0
    sig_pref = [a.aspect for a in pref if a.significant]
    rationale = [
        f"preference-drift evidence {e_pref:.2f} bans (smallest adjusted p over "
        f"{', '.join(cfg.preference_aspects)} = {p_pref if p_pref is not None else 'n/a'}) vs the "
        f"standard threshold {c:.2f} bans (alpha {alpha})",
    ]
    if sig_pref:
        rationale.append(f"significant preference aspects: {', '.join(sig_pref)}")
    if e_acc > 0:
        rationale.append(f"recommendation acceptance fell (evidence {e_acc:.2f} bans) -> supports explore")
    rationale.extend(elig_notes)
    if sig_pref and route is None:
        rationale.append(
            "drift detected, but no alternative strategy has shown a benefit for drifting users in the "
            "offline evaluation, so standard is kept"
        )
    if best == "adapt_to_recent":
        rationale.append(f"adapting: profile recency half-life {half_life:.0f} days")
    elif best == "explore":
        rationale.append(f"exploring: MMR λ {standard_lambda} -> {cfg.explore_lambda}")
    ev = list(report["evidence"])
    for name, e in cfg.evaluated_effects.items():
        ev.append(
            evidence(
                "metric",
                f"offline ΔNDCG@10 {name} vs standard",
                e.get("delta_ndcg10"),
                f"95% CI {e['ci95'][0]:+.4f}..{e['ci95'][1]:+.4f}; {e['n_users']} users; "
                f"{e.get('protocol', '')}",
                e.get("source"),
            )
        )
    state = {
        **base_state,
        "preference_evidence_bans": fnum(e_pref, 4),
        "acceptance_drop_evidence_bans": fnum(e_acc, 4),
        "significant_aspects": [a.aspect for a in aspects if a.significant],
        "drift_route": route,
        "served_strategy": best,
    }
    state["state_hash"] = short_hash(state)
    return _decision(entity, as_of_key, best, scores, conf, state, rationale, ev, cfg.policy_version, None)


# --- serving under a strategy ------------------------------------------------------------------------
def build_strategy_profile(
    engine: RecommendationEngine,
    interactions: Iterable[Interaction],
    genre_prefs: Iterable[str] = (),
    excluded_movie_ids: Iterable[int] = (),
    recency_half_life_days: float | None = None,
    weight_floor: float = 0.5,
) -> UserProfile:
    """``engine.build_profile`` plus the recency decay of ``adapt_to_recent`` (no engine change)."""
    profile = engine.build_profile(interactions, genre_prefs, excluded_movie_ids)
    return decay_profile(profile, recency_half_life_days, weight_floor)


def decay_profile(profile: UserProfile, half_life_days: float | None, floor: float = 0.5) -> UserProfile:
    if not half_life_days or profile.n_interactions == 0 or profile.timestamps.max() <= 0:
        return profile
    age = (profile.timestamps.max() - profile.timestamps) / DAY
    m = floor + (1.0 - floor) * np.power(0.5, age / half_life_days)
    return dataclasses.replace(profile, weights=profile.weights * m, pref_weights=profile.pref_weights * m)


def engine_with_overrides(engine: RecommendationEngine, overrides: Mapping[str, Any]) -> RecommendationEngine:
    """Shallow engine copy whose ranker uses a modified HybridConfig (the shared engine is untouched)."""
    if not overrides:
        return engine
    e = copy.copy(engine)
    e.ranker = copy.copy(engine.ranker)
    e.ranker.config = dataclasses.replace(engine.ranker.config, **dict(overrides))
    return e


def strategy_parameters(
    strategy: str, split: WindowSplit, cfg: StrategyConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(profile_kwargs for build_strategy_profile, HybridConfig overrides) of a strategy."""
    if strategy == "adapt_to_recent":
        return {
            "recency_half_life_days": adapt_half_life_days(split, cfg),
            "weight_floor": cfg.adapt_weight_floor,
        }, {}
    if strategy == "explore":
        return {}, {"diversity_lambda": cfg.explore_lambda}
    return {}, {}


def _entity(user_id: Any) -> str:
    return "anonymous" if user_id is None else str(user_id)


def strategy_for(
    engine: RecommendationEngine,
    interactions: Iterable[Interaction],
    feedback: Iterable[Any] | None = None,
    *,
    user_id: Any = None,
    as_of: Any = None,
    genre_prefs: Iterable[str] = (),
    excluded_movie_ids: Iterable[int] = (),
    config: StrategyConfig | None = None,
    k: int = 10,
    profile: UserProfile | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Light per-request helper: (decision, profile_kwargs, config_overrides).

    Serve with ``build_strategy_profile(engine, interactions, genre_prefs, excluded, **profile_kwargs)``
    and ``engine_with_overrides(engine, config_overrides).recommend(...)``. An abstained decision
    returns empty kwargs/overrides (= standard). Pass the request's already built standard
    ``profile`` (``engine.build_profile(...)``) to avoid building it twice."""
    cfg = config or StrategyConfig()
    its = list(interactions)
    ev = user_events(engine, its, as_of)
    report, aspects, split = drift_report(engine, ev, feedback, cfg.drift, cfg.session_clusters)
    if profile is None:
        profile = engine.build_profile(its, genre_prefs, excluded_movie_ids)
    stats = _confidence_stats(engine, profile, k)
    dec = strategy_decision(
        report,
        aspects,
        split,
        len(ev.ts),
        stats,
        cfg,
        _entity(user_id),
        ev.as_of,
        engine.config.diversity_lambda,
    )
    pk, co = strategy_parameters(dec["state"]["served_strategy"], split, cfg)
    return dec, pk, co


# --- preference history + signals ---------------------------------------------------------------------
def _share_dict(weights: np.ndarray, genres: list[str], keep: list[str]) -> dict[str, float]:
    w = weights[weights.sum(axis=1) > 0]
    if len(w) == 0:
        return {}
    s = (w / w.sum(axis=1, keepdims=True)).sum(axis=0) / len(w)
    out: dict[str, float] = {}
    for g in keep:
        if g == "Other":
            continue
        out[g] = round(float(s[genres.index(g)]), 4)
    if "Other" in keep:
        out["Other"] = round(max(0.0, 1.0 - sum(out.values())), 4)
    return out


def preference_history(
    engine: RecommendationEngine, ev: UserEvents, split: WindowSplit, cfg: StrategyConfig
) -> dict[str, Any]:
    f = item_features(engine)
    gm = f.genre_matrix[ev.items] if len(ev.items) else np.zeros((0, len(f.genres)), dtype=np.float32)
    total = gm[gm.sum(axis=1) > 0]
    overall = (
        (total / total.sum(axis=1, keepdims=True)).sum(axis=0) if len(total) else np.zeros(len(f.genres))
    )
    order = [int(i) for i in np.lexsort((np.arange(len(overall)), -overall)) if overall[i] > 0]
    cats = [f.genres[i] for i in order[: cfg.history_top_categories]]
    if len(order) > cfg.history_top_categories:
        cats.append("Other")
    windows: list[dict[str, Any]] = []

    def add(label: str, mask: np.ndarray) -> None:
        t = ev.ts[mask]
        windows.append(
            {
                "label": label,
                "start": iso_from_epoch(t.min()) if len(t) else None,
                "end": iso_from_epoch(t.max()) if len(t) else None,
                "n": int(mask.sum()),
                "shares": _share_dict(gm[mask], f.genres, cats),
            }
        )

    if len(ev.ts):
        if split.n_recent and split.n_historical:
            add("historical", split.historical)
            add("recent", split.recent)
        lo, hi = float(ev.ts.min()), float(ev.ts.max())
        nb = cfg.n_time_buckets if hi > lo else 1
        edges = np.linspace(lo, hi, nb + 1)
        for b in range(nb):
            mask = (ev.ts >= edges[b]) & ((ev.ts < edges[b + 1]) if b < nb - 1 else (ev.ts <= edges[b + 1]))
            windows.append(
                {
                    "label": f"{datetime.fromtimestamp(edges[b], UTC):%Y-%m-%d}.."
                    f"{datetime.fromtimestamp(edges[b + 1], UTC):%Y-%m-%d}",
                    "start": iso_from_epoch(edges[b]),
                    "end": iso_from_epoch(edges[b + 1]),
                    "n": int(mask.sum()),
                    "shares": _share_dict(gm[mask], f.genres, cats),
                }
            )
    return {"categories": cats, "windows": windows}


def _signal(
    aspect: AspectResult,
    entity: str,
    as_of: float,
    as_of_key: str,
    window: str,
) -> dict[str, Any]:
    e = aspect.effect
    title, value, baseline, unit, direction = (
        _ASPECT_LABEL[aspect.aspect] + " changed",
        None,
        None,
        None,
        None,
    )
    if aspect.aspect == "genre_distribution":
        mv = (e.get("largest_changes") or [{}])[0]
        value, baseline, unit = mv.get("recent"), mv.get("historical"), "share"
        direction = "up" if (mv.get("change") or 0) > 0 else "down"
        title = f"Genre mix shifted: {mv.get('category')} {direction} ({baseline} -> {value})"
    elif aspect.aspect == "rating_level":
        value, baseline, unit = e.get("recent_mean"), e.get("historical_mean"), "stars"
        direction = "up" if (aspect.statistic or 0) > 0 else "down"
        title = f"Ratings {'higher' if direction == 'up' else 'lower'} recently ({baseline} -> {value})"
    elif aspect.aspect == "activity_rate":
        value, baseline, unit = e.get("recent_rate"), e.get("historical_rate"), e.get("unit")
        direction = "up" if (aspect.statistic or 1) > 1 else "down"
        title = f"Activity {'up' if direction == 'up' else 'down'} (rate ratio {e.get('rate_ratio')})"
    elif aspect.aspect == "release_year":
        value, baseline, unit = e.get("recent_median"), e.get("historical_median"), "year"
        direction = "up" if (aspect.statistic or 0) > 0 else "down"
        title = f"Watching {'newer' if direction == 'up' else 'older'} films (median {baseline} -> {value})"
    elif aspect.aspect == "content_similarity":
        value, baseline, unit = e.get("cosine"), e.get("null_mean_cosine"), "cosine"
        direction = "down"
        title = f"Recent films differ in content from history (cosine {value} vs typical {baseline})"
    elif aspect.aspect == "acceptance":
        value, baseline, unit = e.get("recent_rate"), e.get("historical_rate"), "acceptance rate"
        direction = "up" if (aspect.statistic or 0) > 0 else "down"
        title = f"Recommendation acceptance {direction} ({baseline} -> {value})"
    change = None if value is None or baseline is None else fnum(float(value) - float(baseline), 6)
    strength = 1.0 - (aspect.p_adjusted if aspect.p_adjusted is not None else 1.0)
    return {
        "id": stable_id("sig", "drift", aspect.aspect, entity, as_of_key),
        "dedup_key": f"drift:{aspect.aspect}:user:{entity}",
        "kind": "drift",
        "entity_type": "user",
        "entity": entity,
        "title": title,
        "value": value,
        "unit": unit,
        "strength": fnum(strength, 4),
        "direction": direction,
        "source": "app" if entity != "anonymous" else "events",
        "observed_at": iso_from_epoch(as_of),
        "window": window,
        "freshness_days": 0.0,
        "evidence": [evidence("test", aspect.test, fnum(aspect.p_adjusted, 6), aspect.detail)],
        "entity_id": f"user:{entity}",
        "baseline": baseline,
        "change": change,
        "confidence": fnum(strength, 4),
        "confidence_kind": "evidence",
        "domain": DOMAIN,
    }


# --- public entry point ------------------------------------------------------------------------------
def _rec_dict(
    rec: Any, decision_id: str, strategy: str, params: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    ev = [evidence("model", "hybrid score", rec.score, f"reason code {rec.reason_code}")]
    if rec.anchor_movie_ids:
        ev.append(evidence("record", "anchor movies", ", ".join(map(str, rec.anchor_movie_ids[:3]))))
    if strategy == "adapt_to_recent":
        ev.append(
            evidence(
                "metric",
                "profile recency half-life (days)",
                fnum(params.get("recency_half_life_days"), 2),
                "adapt_to_recent: older events down-weighted",
                decision_id,
            )
        )
    elif strategy == "explore":
        ev.append(
            evidence("metric", "MMR diversity λ", overrides.get("diversity_lambda"), "explore", decision_id)
        )
    else:
        ev.append(evidence("record", "strategy", strategy, "served under the strategy decision", decision_id))
    return {
        "item_id": rec.movie_id,
        "title": rec.title,
        "rank": rec.rank,
        "score": rec.score,
        "reason": rec.reason,
        "confidence": rec.confidence,
        "confidence_kind": rec.confidence_kind,
        "decision_id": decision_id,
        "strategy": strategy,
        "evidence": ev,
    }


def user_intelligence(
    engine: RecommendationEngine,
    interactions: Iterable[Interaction],
    *,
    as_of: Any = None,
    feedback: Iterable[Any] | None = None,
    k: int = 10,
    config: StrategyConfig | None = None,
    user_id: Any = None,
    genre_prefs: Iterable[str] = (),
    excluded_movie_ids: Iterable[int] = (),
) -> dict[str, Any]:
    """The ``GET /me/intelligence`` payload (docs/platform.md §10); JSON-safe (no NaN)."""
    if not 1 <= int(k) <= 50:
        raise ValueError("k must be between 1 and 50")
    cfg = config or StrategyConfig()
    cutoff = _epoch(as_of)
    its = [it for it in interactions if cutoff is None or it.timestamp <= cutoff]
    ev = user_events(engine, its, cutoff)
    report, aspects, split = drift_report(engine, ev, feedback, cfg.drift, cfg.session_clusters)
    entity = _entity(user_id)
    base_profile = engine.build_profile(its, genre_prefs, excluded_movie_ids)
    stats = _confidence_stats(engine, base_profile, k)
    dec = strategy_decision(
        report, aspects, split, len(ev.ts), stats, cfg, entity, ev.as_of, engine.config.diversity_lambda
    )
    served = dec["state"]["served_strategy"]
    pk, co = strategy_parameters(served, split, cfg)
    profile = decay_profile(base_profile, pk.get("recency_half_life_days"), pk.get("weight_floor", 0.5))
    recs = engine_with_overrides(engine, co).recommend(profile, k=int(k))
    as_of_key = _as_of_key(ev.as_of, len(ev.ts))
    window = None
    if split.t_split is not None:
        window = f"{iso_from_epoch(split.first)}..{iso_from_epoch(split.last)}"
    signals = [_signal(a, entity, ev.as_of, as_of_key, window or "") for a in aspects if a.significant]
    return {
        "user_id": user_id,
        "as_of": iso_from_epoch(ev.as_of),
        "profile": {
            "n_events": len(ev.ts),
            "first_event": iso_from_epoch(ev.ts.min()) if len(ev.ts) else None,
            "last_event": iso_from_epoch(ev.ts.max()) if len(ev.ts) else None,
        },
        "preference_history": preference_history(engine, ev, split, cfg),
        "drift": report,
        "strategy": dec,
        "signals": signals,
        "recommendations": [_rec_dict(r, dec["id"], served, pk, co) for r in recs],
    }
