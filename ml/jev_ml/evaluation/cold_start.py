"""Cold-start protocols and profile-size conditioned hybrid weights (``HybridConfig.cold_stages``).

Protocols (the same targets for every profile size N, so the curves over N are comparable)
-----------------------------------------------------------------------------------------
* ``user_temporal``: every evaluable user's profile is truncated to their first N *training*
  interactions and folded in as an unseen user; targets are the user's held-out rows. This is the
  historical cold-start protocol (N = 3), extended to N in {0, 1, 3, 5, 10}.
* ``global_temporal``: *new users*, i.e. users with no interaction before the time cut who are
  active in the held-out window. Their first ``HISTORY_PREFIX`` window ratings are the history
  they build in the app. The profile is the first N of those, and the targets are their window
  ratings after the prefix. The models never see any row after the cut, which is exactly how the
  service folds in a new app user.

Simulated onboarding
--------------------
The app asks new users for genres. MovieLens has no such answers, so the ``onboarding`` variant
sets the 3 MovieLens genres most frequent among the films the user liked (>= 4) within their first
``HISTORY_PREFIX`` interactions, which always precede the targets. This is an optimistic stand-in
for a self-declared taste, and both variants (with and without) are reported.

Stage tuning (validation split only)
------------------------------------
Buckets by profile size: 0, 1-3 and 4-10 interactions (representative N: 0 / 1, 3 / 5, 10). Larger
profiles keep the adaptive weights. Candidate strategies per bucket:

* ``incumbent``        no stage (the adaptive ramp/boost weights)
* ``popularity``       popularity only (the fallback)
* ``pop_blend``        (1 - a) * adaptive weights + a * popularity, a in POP_BLEND
* ``stage_hybrid``     free weights over the six signals (the incumbent weights without the ramp,
                       then seeded Dirichlet samples)
* ``profile_content``  content/profile-based: weights over content, genre preference,
                       popularity and recency only (no collaborative signal)

The candidate with the best mean validation NDCG@10 over the bucket's representative sizes and both
onboarding variants is kept. Test data is never used for selection.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.data.dataset import ItemIndex, profiles_from_interactions, split_list
from jev_ml.evaluation.evaluator import Evaluator
from jev_ml.models.hybrid import SIGNALS, HybridConfig, HybridRanker
from jev_ml.signals import LIKE_THRESHOLD

log = logging.getLogger(__name__)

HISTORY_PREFIX = 10
ONBOARDING_GENRES = 3
COLD_SIZES = (0, 1, 3, 5, 10)
BUCKETS: tuple[dict[str, Any], ...] = (
    {"name": "cold_0", "min_profile": 0, "max_profile": 0, "reps": (0,)},
    {"name": "cold_1_3", "min_profile": 1, "max_profile": 3, "reps": (1, 3)},
    {"name": "cold_4_10", "min_profile": 4, "max_profile": 10, "reps": (5, 10)},
)
POP_BLEND = (0.25, 0.5, 0.75)
PROFILE_SIGNALS = ("content", "preference", "popularity", "recency")
ALL_PROFILES = 10**9  # a stage bound that covers every profile of a cold evaluator


def bucket_of(n: int) -> str:
    for b in BUCKETS:
        if n <= int(b["max_profile"]):
            return str(b["name"])
    return "warm"


# --------------------------------------------------------------------------------------------------
# protocol builders


def onboarding_genres(history: pd.DataFrame, movies: pd.DataFrame) -> dict[int, dict[str, float]]:
    """Simulated onboarding answers: the top genres among the liked films of each user's
    ``history`` rows (already restricted to the prefix that precedes the targets)."""
    genres_of = dict(zip(movies["movie_id"].tolist(), movies["genres"].tolist(), strict=True))
    out: dict[int, dict[str, float]] = {}
    liked = history[history["rating"] >= LIKE_THRESHOLD]
    for uid, g in liked.groupby("user_id"):
        c: Counter[str] = Counter()
        for mid in g["movie_id"].tolist():
            c.update(split_list(genres_of.get(int(mid))))
        top = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:ONBOARDING_GENRES]
        if top:
            out[int(uid)] = {name: 1.0 for name, _ in top}
    return out


def _first_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort").groupby("user_id").head(n)


def new_user_cases(
    fit_rows: pd.DataFrame, window: pd.DataFrame, item_index: ItemIndex, n: int, threshold: float
) -> tuple[dict[int, Any], dict[int, set[int]], pd.DataFrame]:
    """Global-split new users: (profiles of their first n window ratings, relevant targets after the
    ``HISTORY_PREFIX`` prefix, the prefix rows used for onboarding genres)."""
    known = set(fit_rows["user_id"].unique().tolist())
    w = window[~window["user_id"].isin(known)]
    w = w.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort")
    pos = w.groupby("user_id").cumcount()
    prefix = w[pos < HISTORY_PREFIX]
    after = w[pos >= HISTORY_PREFIX]
    rel_rows = after[after["rating"] >= threshold]
    relevant = {
        int(u): set(item_index.indices_of(g["movie_id"].to_numpy()).tolist())
        for u, g in rel_rows.groupby("user_id")
    }
    users = np.asarray(sorted(relevant), dtype=np.int64)
    prof_rows = prefix[prefix.groupby("user_id").cumcount() < n] if n > 0 else prefix.iloc[0:0]
    profiles = profiles_from_interactions(prof_rows, item_index, users, {})
    return profiles, relevant, prefix[prefix["user_id"].isin(set(users.tolist()))]


def cold_evaluator(
    strategy: str,
    fit_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    movies: pd.DataFrame,
    item_index: ItemIndex,
    train_user_ids: np.ndarray,
    pairwise_sim: Any,
    pop_fraction: np.ndarray,
    n: int,
    onboarding: bool,
    ecfg: dict[str, Any],
    exclude_items: np.ndarray | None = None,
) -> Evaluator:
    """The cold-start evaluator of one protocol at profile size ``n``."""
    threshold = float(ecfg["relevance_threshold"])
    ks = tuple(ecfg["ks"])
    if strategy == "global_temporal":
        profiles, relevant, prefix = new_user_cases(fit_rows, target_rows, item_index, n, threshold)
        if onboarding:
            for u, g in onboarding_genres(prefix, movies).items():
                if u in profiles:
                    profiles[u].genre_prefs = g
        return Evaluator.from_profiles(
            profiles,
            relevant,
            item_index,
            len(train_user_ids),
            pairwise_sim,
            pop_fraction,
            ks=ks,
            exclude_items=exclude_items,
        )
    prefs = onboarding_genres(_first_rows(fit_rows, HISTORY_PREFIX), movies) if onboarding else None
    return Evaluator(
        fit_rows,
        target_rows,
        item_index,
        train_user_ids,
        pairwise_sim,
        pop_fraction,
        ks=ks,
        relevance_threshold=threshold,
        truncate_profile_to=n,
        genre_prefs=prefs,
        exclude_items=exclude_items,
    )


# --------------------------------------------------------------------------------------------------
# stage candidates and tuning


def stage_candidates(
    base: HybridConfig, seed: int, n_dirichlet: int = 20, n_profile: int = 10
) -> list[tuple[str, dict[str, Any] | None]]:
    """(family, stage-without-bounds) candidates; None = incumbent adaptive weights."""
    rng = np.random.default_rng(seed)
    cands: list[tuple[str, dict[str, Any] | None]] = [
        ("incumbent", None),
        ("popularity", {"weights": {"popularity": 1.0}}),
    ]
    cands += [("pop_blend", {"popularity_blend": a}) for a in POP_BLEND]
    cands.append(("stage_hybrid", {"weights": dict(base.weights)}))
    for _ in range(n_dirichlet):
        w = rng.dirichlet(np.ones(len(SIGNALS)))
        cands.append(("stage_hybrid", {"weights": dict(zip(SIGNALS, w.round(4).tolist(), strict=True))}))
    for _ in range(n_profile):
        w = rng.dirichlet(np.ones(len(PROFILE_SIGNALS)))
        cands.append(
            ("profile_content", {"weights": dict(zip(PROFILE_SIGNALS, w.round(4).tolist(), strict=True))})
        )
    return cands


def with_stage(base: HybridConfig, stage: dict[str, Any] | None) -> HybridConfig:
    """``base`` with one stage covering every profile (for evaluating a candidate on a cold
    evaluator whose profiles all fall in the bucket), or unchanged for the incumbent."""
    if stage is None:
        return HybridConfig.from_dict({**asdict(base), "cold_stages": None})
    return HybridConfig.from_dict({**asdict(base), "cold_stages": [{**stage, "max_profile": ALL_PROFILES}]})


def tune_cold_stages(
    ranker_factory: Any,
    evaluators: dict[tuple[int, bool], Evaluator],
    base: HybridConfig,
    seed: int,
    n_dirichlet: int = 20,
    n_profile: int = 10,
    metric: str = "ndcg@10",
) -> dict[str, Any]:
    """Pick one strategy per bucket on validation evaluators keyed by (profile size, onboarding).

    ``ranker_factory()`` returns a fresh signal-caching HybridRanker (a new cache per evaluator keeps
    memory at one evaluator's signals)."""
    from jev_ml.training import hybrid_fn

    cands = stage_candidates(base, seed, n_dirichlet, n_profile)
    scores: dict[tuple[int, bool], list[float]] = {}
    for key, ev in evaluators.items():
        ranker: HybridRanker = ranker_factory()
        vals = []
        for _, stage in cands:
            ranker.config = with_stage(base, stage)
            vals.append(ev.evaluate(f"cold-trial[n={key[0]},onb={key[1]}]", hybrid_fn(ranker)).metrics[metric])
        scores[key] = vals
    chosen: list[dict[str, Any]] = []
    report: dict[str, Any] = {"buckets": {}, "n_candidates": len(cands), "metric": metric}
    for b in BUCKETS:
        keys = [k for k in evaluators if k[0] in b["reps"]]
        if not keys:
            continue
        obj = np.mean([scores[k] for k in keys], axis=0)
        best = int(np.argmax(obj))  # first max: the incumbent wins ties
        fam = {f: float(max(o for (ff, _), o in zip(cands, obj, strict=True) if ff == f)) for f, _ in cands}
        family, stage = cands[best]
        report["buckets"][b["name"]] = {
            "reps": list(b["reps"]),
            "evaluators": [f"n={k[0]},onboarding={k[1]}" for k in keys],
            "chosen_family": family,
            "chosen_stage": stage,
            "validation_objective": float(obj[best]),
            "incumbent_objective": float(obj[0]),
            "best_per_family": fam,
            "trials": [
                {"family": f, "stage": s, "objective": float(o)} for (f, s), o in zip(cands, obj, strict=True)
            ],
        }
        if stage is not None:
            chosen.append(
                {
                    "name": b["name"],
                    "min_profile": int(b["min_profile"]),
                    "max_profile": int(b["max_profile"]),
                    **stage,
                }
            )
        log.info("cold stage %s: %s (val %.4f vs incumbent %.4f)", b["name"], family, obj[best], obj[0])
    report["cold_stages"] = chosen or None
    return report
