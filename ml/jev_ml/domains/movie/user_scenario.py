"""Preference what-if scenarios for one user (``POST /me/intelligence/scenarios``, docs/platform.md §10).

Trajectory
----------
The user's genre shares in the historical window (s_h) and the recent window (s_r) are the same
windows as the drift report (``user_intel``). One *window step* is d = s_r - s_h. Scenarios:

* ``continue``  : s_r + f * d, f = ``factor`` in [0, 1] (default 1: one more step of the same size)
* ``accelerate``: s_r + f * d, f = ``factor`` > 1 (default 2)
* ``reverse``   : s_r + f * (s_h - s_r), f = ``factor`` in (0, 1] (default 1: back to s_h)

Negative shares are clipped to 0 and the vector renormalised to sum 1.

Uncertainty
-----------
80 % bands from a seeded bootstrap: the user's events are resampled with replacement within each
window, s_h / s_r and the projection are recomputed, and the 10th / 90th percentiles taken per
category. ``mean`` is the projection of the observed shares; when the bootstrap distribution is so
degenerate that its 10–90 % range misses that point (rare categories), the band is widened to
include it. The bands cover sampling noise in the observed shares only, not whether the trend
continues.

Ranking under a projected preference
-------------------------------------
The real hybrid model ranks the catalogue for a *reweighted* profile: every profile event is
multiplied by m_i = sum_g a_ig * p_g / b_g (a_ig = the film's genre membership normalised to sum 1,
p = projected share, b = baseline (recent-window) share; for a genre absent from the baseline the
profile's own weight-share of it), clipped to [``MIN_MULT``, ``MAX_MULT``]. A scenario that
projects no change therefore reproduces the baseline ranking exactly. Collaborative (item-kNN),
latent (ALS fold-in), content and genre-preference scores all follow from the reweighted events;
explicit onboarding genres are left as they are.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from jev_ml.core.drift import DAY, OK, split_windows
from jev_ml.domains.movie.user_intel import (
    StrategyConfig,
    item_features,
    preference_history,
    user_events,
)
from jev_ml.engine import Interaction, RecommendationEngine
from jev_ml.intel.common import evidence, fnum, iso_from_epoch
from jev_ml.signals import UserProfile

KINDS = ("continue", "accelerate", "reverse")
DEFAULT_FACTOR = {"continue": 1.0, "accelerate": 2.0, "reverse": 1.0}
DEFAULT_NAMES = {
    "continue": "Trend continues",
    "accelerate": "Trend accelerates",
    "reverse": "Trend reverses",
}
MAX_SCENARIOS = 4
N_BOOTSTRAP = 200
BAND = (10.0, 90.0)
MIN_MULT, MAX_MULT = 0.1, 10.0
SEED = 20260924


def validate_spec(spec: Mapping[str, Any] | None) -> tuple[int, list[dict[str, Any]]]:
    """(k, scenarios) from the request body; ValueError on anything invalid."""
    spec = dict(spec or {})
    k = spec.get("k", 10)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 50:
        raise ValueError("k must be an integer between 1 and 50")
    raw = spec.get("scenarios")
    if raw is None:
        raw = [{"kind": kd} for kd in KINDS]
    if not isinstance(raw, list) or not raw:
        raise ValueError("scenarios must be a non-empty list")
    if len(raw) > MAX_SCENARIOS:
        raise ValueError(f"at most {MAX_SCENARIOS} scenarios")
    out = []
    for i, sc in enumerate(raw):
        if not isinstance(sc, Mapping):
            raise ValueError(f"scenario {i} must be an object")
        kind = sc.get("kind")
        if kind not in KINDS:
            raise ValueError(f"scenario {i}: unknown kind {kind!r} (expected one of {', '.join(KINDS)})")
        f = sc.get("factor")
        f = DEFAULT_FACTOR[kind] if f is None else f
        if isinstance(f, bool) or not isinstance(f, int | float) or not math.isfinite(f):
            raise ValueError(f"scenario {i}: factor must be a finite number")
        f = float(f)
        if kind == "continue" and not 0.0 <= f <= 1.0:
            raise ValueError(f"scenario {i}: continue needs 0 <= factor <= 1 (fraction of one window step)")
        if kind == "accelerate" and not 1.0 < f <= 5.0:
            raise ValueError(f"scenario {i}: accelerate needs 1 < factor <= 5")
        if kind == "reverse" and not 0.0 < f <= 1.0:
            raise ValueError(f"scenario {i}: reverse needs 0 < factor <= 1 (fraction of the way back)")
        name = sc.get("name") or DEFAULT_NAMES[kind]
        if not isinstance(name, str) or len(name) > 80:
            raise ValueError(f"scenario {i}: name must be a string of at most 80 characters")
        out.append({"name": name, "kind": kind, "factor": f})
    return k, out


def project(s_h: np.ndarray, s_r: np.ndarray, kind: str, factor: float) -> np.ndarray:
    """Projected share vector(s) (last axis = categories), clipped at 0 and renormalised."""
    step = (s_h - s_r) if kind == "reverse" else (s_r - s_h)
    p = np.clip(s_r + factor * step, 0.0, None)
    tot = p.sum(axis=-1, keepdims=True)
    return np.asarray(np.divide(p, tot, out=np.zeros_like(p), where=tot > 0))


def _window_shares(a: np.ndarray) -> np.ndarray:
    """Mean per-event genre shares (rows = events with >= 1 genre)."""
    if len(a) == 0:
        return np.zeros(a.shape[1])
    return np.asarray((a / a.sum(axis=1, keepdims=True)).mean(axis=0))


def _boot_shares(a: np.ndarray, rng: np.random.Generator, b: int) -> np.ndarray:
    if len(a) == 0:
        return np.zeros((b, a.shape[1]))
    rows = a / a.sum(axis=1, keepdims=True)
    idx = rng.integers(0, len(rows), (b, len(rows)))
    return np.asarray(rows[idx].mean(axis=1))


def _collapse(v: np.ndarray, genres: list[str], cats: list[str]) -> dict[str, float]:
    out = {g: float(v[..., genres.index(g)]) for g in cats if g != "Other"}
    if "Other" in cats:
        out["Other"] = float(max(0.0, 1.0 - sum(out.values())))
    return out


def reweight_profile(
    profile: UserProfile, genre_matrix: np.ndarray, target: np.ndarray, reference: np.ndarray
) -> UserProfile:
    """Multiply every profile event by m_i = sum_g a_ig * r_g, r_g = target_g / reference_g (the
    projected vs the baseline share of genre g; for a genre absent from the baseline the profile's
    own weight-share is the reference), clipped to [MIN_MULT, MAX_MULT]. target == reference leaves
    the profile unchanged."""
    if profile.n_interactions == 0:
        return profile
    a = genre_matrix[profile.items].astype(np.float64)
    rs = a.sum(axis=1, keepdims=True)
    a = np.divide(a, rs, out=np.zeros_like(a), where=rs > 0)
    cur = (a * profile.weights[:, None]).sum(axis=0)
    cur = cur / cur.sum() if cur.sum() > 0 else cur
    ref = np.where(reference > 0, reference, cur)
    ratio = np.divide(target, ref, out=np.ones_like(target), where=ref > 0)
    mult = np.clip(np.where(rs.ravel() > 0, a @ ratio, 1.0), MIN_MULT, MAX_MULT)
    return dataclasses.replace(
        profile, weights=profile.weights * mult, pref_weights=profile.pref_weights * mult
    )


def _rec(rec: Any, detail: str) -> dict[str, Any]:
    return {
        "item_id": rec.movie_id,
        "title": rec.title,
        "rank": rec.rank,
        "score": rec.score,
        "reason": rec.reason,
        "confidence": rec.confidence,
        "confidence_kind": rec.confidence_kind,
        "decision_id": None,
        "evidence": [evidence("model", "hybrid score", rec.score, detail)],
    }


def user_preference_scenarios(
    engine: RecommendationEngine,
    interactions: Iterable[Interaction],
    spec: Mapping[str, Any] | None,
    *,
    as_of: Any = None,
    genre_prefs: Iterable[str] = (),
    excluded_movie_ids: Iterable[int] = (),
    config: StrategyConfig | None = None,
) -> dict[str, Any]:
    """The ``POST /me/intelligence/scenarios`` payload (docs/platform.md §10); JSON-safe."""
    k, scenarios = validate_spec(spec)
    cfg = config or StrategyConfig()
    its = list(interactions)
    ev = user_events(engine, its, as_of)
    its = [it for it in its if it.timestamp <= ev.as_of]
    f = item_features(engine)
    days = np.floor(ev.ts / DAY).astype(np.int64)
    split = split_windows(ev.ts, cfg.drift, ev.as_of, clusters=days if cfg.session_clusters else None)
    hist_cats = preference_history(engine, ev, split, cfg)["categories"]
    gm = f.genre_matrix[ev.items].astype(np.float64) if len(ev.items) else np.zeros((0, len(f.genres)))
    has = gm.sum(axis=1) > 0
    gh, gr = gm[split.historical & has], gm[split.recent & has]
    s_h, s_r = _window_shares(gh), _window_shares(gr)
    usable = len(gh) > 0 and len(gr) > 0
    rng = np.random.default_rng(SEED)
    bh, br = _boot_shares(gh, rng, N_BOOTSTRAP), _boot_shares(gr, rng, N_BOOTSTRAP)

    base_profile = engine.build_profile(its, genre_prefs, excluded_movie_ids)
    baseline_recs = engine.recommend(base_profile, k=k)
    base_ids = [r.movie_id for r in baseline_recs]
    out_sc = []
    for sc in scenarios:
        point = project(s_h, s_r, sc["kind"], sc["factor"]) if usable else s_r
        boots = project(bh, br, sc["kind"], sc["factor"]) if usable else np.tile(s_r, (N_BOOTSTRAP, 1))
        lo, hi = np.percentile(boots, BAND, axis=0)
        mean_c = _collapse(point, f.genres, hist_cats)
        lo_c, hi_c = _collapse(lo, f.genres, hist_cats), _collapse(hi, f.genres, hist_cats)
        if "Other" in hist_cats:  # the "Other" band comes from the bootstrap of the residual share
            resid = 1.0 - boots[:, [f.genres.index(g) for g in hist_cats if g != "Other"]].sum(axis=1)
            lo_c["Other"], hi_c["Other"] = (float(x) for x in np.percentile(resid, BAND))
        shares = {
            g: {
                "mean": fnum(mean_c[g], 4),
                "lo80": fnum(min(lo_c[g], mean_c[g]), 4),
                "hi80": fnum(max(hi_c[g], mean_c[g]), 4),
            }
            for g in hist_cats
        }
        prof = reweight_profile(base_profile, f.genre_matrix, point, s_r)
        recs = engine.recommend(prof, k=k)
        ids = [r.movie_id for r in recs]
        movers = sorted(
            (
                (g, float(point[f.genres.index(g)] - s_r[f.genres.index(g)]))
                for g in hist_cats
                if g != "Other"
            ),
            key=lambda t: -abs(t[1]),
        )[:3]
        verb = "one more window step" if sc["kind"] != "reverse" else "back toward the historical mix"
        out_sc.append(
            {
                "name": sc["name"],
                "kind": sc["kind"],
                "factor": sc["factor"],
                "assumptions": [
                    f"{sc['kind']}: shares move {verb} with factor {sc['factor']:g} "
                    f"({'s_r + f·(s_r - s_h)' if sc['kind'] != 'reverse' else 's_r + f·(s_h - s_r)'})",
                    "largest projected changes vs the recent window: "
                    + ", ".join(f"{g} {d:+.3f}" for g, d in movers),
                ],
                "projected_shares": shares,
                "recommendations": [
                    _rec(r, f"ranked for the '{sc['name']}' projected preference") for r in recs
                ],
                "overlap_with_baseline": fnum(len(set(ids) & set(base_ids)) / k, 4),
            }
        )
    n_sessions = len(np.unique(days)) if len(days) else 0
    notes = []
    if not usable:
        notes.append(
            "the history does not have both a historical and a recent window, so every scenario "
            "equals the current mix"
        )
    elif split.status != OK:
        notes.append(
            f"the drift guards are not met ({split.detail}); treat the trajectory with extra caution"
        )
    span = 0.0 if not len(ev.ts) else (float(ev.ts.max()) - float(ev.ts.min())) / DAY
    uncertainty = (
        f"80% bands come from {N_BOOTSTRAP} bootstrap resamples of this user's events within each window "
        f"({len(gh)} historical / {len(gr)} recent events with genres, {n_sessions} active days over "
        f"{span:.0f} days). They reflect sampling noise in the observed shares, not whether the trend "
        "will continue: a straight-line step from two windows is a scenario, not a forecast."
        + (" " + " ".join(notes) if notes else "")
    )
    return {
        "as_of": iso_from_epoch(ev.as_of),
        "assumptions": [
            "historical and recent windows are those of the drift report "
            f"(recent = last {cfg.drift.recent_n} events, extended to whole days)",
            "genre shares: each film counts once, split equally over its genres; categories beyond the "
            f"top {cfg.history_top_categories} are pooled as 'Other'",
            "recommendations under a scenario: each profile event is reweighted by the projected / baseline "
            f"share of its genres (multiplier clipped to [{MIN_MULT:g}, {MAX_MULT:g}]) and the profile is "
            "ranked by the served hybrid model; no projected change = the baseline ranking",
            "baseline = the unmodified profile and its recent-window genre shares",
        ],
        "uncertainty_note": uncertainty,
        "baseline": {
            "shares": {g: fnum(v, 4) for g, v in _collapse(s_r, f.genres, hist_cats).items()},
            "recommendations": [_rec(r, "standard profile") for r in baseline_recs],
        },
        "scenarios": out_sc,
        "evidence": [
            evidence("record", "historical window events", len(gh), iso_from_epoch(split.first)),
            evidence("record", "recent window events", len(gr), iso_from_epoch(split.t_split)),
            evidence(
                "metric",
                "largest genre-share step (recent - historical)",
                fnum(float(np.max(np.abs(s_r - s_h))) if usable else 0.0, 4),
                None,
            ),
            evidence("model", "ranking model", engine.version, "served hybrid model"),
        ],
    }
