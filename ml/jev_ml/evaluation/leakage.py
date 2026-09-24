"""Leakage controls for offline evaluation.

What can leak into a temporal evaluation, and what this module does about it:

* **split order**: ``check_split`` asserts that, per protocol, no validation/test interaction
  precedes the training period (globally for ``global_temporal``, per user for ``user_temporal``)
  and that no (user, item) pair is in two parts.
* **user tags** (content features): MovieLens tags carry a timestamp and are written by the same
  users whose ratings are held out. ``data/processed/movies.csv`` aggregates *all* tags, so a
  content model fitted on it has seen tags written during the validation/test period.
  ``movies_with_tags_before`` rebuilds the ``tags`` column from ``tags.csv`` keeping only tags
  written before the evaluation cut (the same cut as the ratings: global, or per user).
* **popularity / IDF**: popularity statistics come from ``TrainContext.interactions`` only (the
  training rows). The TF-IDF vocabulary and IDF are fitted on item *metadata*; with the tag fix
  above, no field of that metadata is written after the cut except the static catalogue fields
  (title, genres, year, Wikidata people/keywords/description).
* **catalogue**: under a global time cut, films released after the cut cannot be recommended in
  reality. ``future_items`` lists them, and the evaluator excludes them from every candidate set.
* **catalogue statistics** (``movies.n_ratings`` / ``mean_rating`` are over all ratings) are used
  only by request filters, never by the unfiltered ranking that is evaluated; a unit test pins that.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.data.preprocess import LIST_SEP, normalize_tag
from jev_ml.evaluation.split import Split


class LeakageError(AssertionError):
    """A split or a fit input violates the temporal protocol."""


def check_split(split: Split) -> dict[str, Any]:
    """Raise LeakageError when the split breaks its own temporal contract; return the evidence."""
    keys = [set(zip(d["user_id"], d["movie_id"], strict=True)) for d in (split.train, split.val, split.test)]
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise LeakageError("a (user, item) pair appears in two parts of the split")
    out: dict[str, Any] = {"strategy": split.strategy, "overlapping_pairs": 0}
    if split.strategy == "global_temporal":
        tr_max = float(split.train["timestamp"].max()) if len(split.train) else -np.inf
        va_min = float(split.val["timestamp"].min()) if len(split.val) else np.inf
        va_max = float(split.val["timestamp"].max()) if len(split.val) else -np.inf
        te_min = float(split.test["timestamp"].min()) if len(split.test) else np.inf
        if not tr_max < va_min or not max(tr_max, va_max) < te_min:
            raise LeakageError(
                f"global split out of order: train max {tr_max}, val [{va_min}, {va_max}], test min {te_min}"
            )
        out.update(train_max_ts=tr_max, val_min_ts=va_min, val_max_ts=va_max, test_min_ts=te_min)
        return out
    tr = split.train.groupby("user_id")["timestamp"].max()
    va_lo = split.val.groupby("user_id")["timestamp"].min()
    va_hi = split.val.groupby("user_id")["timestamp"].max()
    te = split.test.groupby("user_id")["timestamp"].min()
    bad_val = (tr.reindex(va_lo.index) > va_lo).fillna(False)
    last_before_test = pd.concat([tr, va_hi], axis=1).max(axis=1)
    bad_test = (last_before_test.reindex(te.index) > te).fillna(False)
    if bad_val.any() or bad_test.any():
        raise LeakageError(
            f"per-user order violated for {int(bad_val.sum())} validation and "
            f"{int(bad_test.sum())} test users"
        )
    out.update(users_checked=len(te))
    return out


def tag_cutoffs(split: Split, stage: str) -> float | dict[int, float]:
    """The time before which a tag may be used when fitting for ``stage``.

    ``fit``  (models fitted on train, scored on validation): the start of the validation period.
    ``test`` (models fitted on train+validation, scored on test): the start of the test period.
    Global split: one timestamp. User-temporal split: one timestamp per user; users without held-out
    rows have no cut (all their rows are training rows)."""
    if stage not in ("fit", "test"):
        raise ValueError("stage must be 'fit' or 'test'")
    held = split.val if stage == "fit" else split.test
    if split.strategy == "global_temporal":
        return float(held["timestamp"].min()) if len(held) else float("inf")
    if stage == "fit":
        starts = pd.concat([split.val, split.test]).groupby("user_id")["timestamp"].min()
    else:
        starts = split.test.groupby("user_id")["timestamp"].min()
    return {int(u): float(t) for u, t in zip(starts.index.to_numpy(), starts.to_numpy(), strict=True)}


def movies_with_tags_before(
    movies: pd.DataFrame, tags: pd.DataFrame | None, cutoff: float | dict[int, float] | None
) -> pd.DataFrame:
    """Copy of ``movies`` whose ``tags`` column only aggregates tags written before ``cutoff``.

    ``tags`` is the raw MovieLens tags.csv (userId, movieId, tag, timestamp). ``cutoff`` None keeps
    every tag (reproduces preprocessing). Without the raw tags file the tags field is emptied, which
    is leak-free by removal. The aggregation (normalisation, order by frequency then text) is the
    same as ``jev_ml.data.preprocess``."""
    out = movies.copy()
    if tags is None:
        out["tags"] = ""
        return out
    t = tags.copy()
    if cutoff is not None:
        if isinstance(cutoff, dict):
            cut = t["userId"].map(cutoff).astype(np.float64).fillna(np.inf)
        else:
            cut = pd.Series(float(cutoff), index=t.index)
        t = t[t["timestamp"].astype(np.float64) < cut]
    t["tag_norm"] = t["tag"].map(normalize_tag)
    t = t[t["tag_norm"].str.len() > 1]
    counts = (
        t.groupby(["movieId", "tag_norm"])
        .size()
        .reset_index(name="n")
        .sort_values(["movieId", "n", "tag_norm"], ascending=[True, False, True])
    )
    agg = counts.groupby("movieId")["tag_norm"].agg(lambda s: LIST_SEP.join(s))
    out["tags"] = out["movie_id"].map(agg).fillna("").astype(str)
    return out


def future_items(movies: pd.DataFrame, seen_before: pd.DataFrame, cutoff_ts: float) -> np.ndarray:
    """Item indices (aligned with ``movies`` sorted by id) of films released after the year of
    ``cutoff_ts`` that nobody interacted with before it: they did not exist at the cut."""
    cut_year = pd.Timestamp(cutoff_ts, unit="s").year
    years = pd.to_numeric(movies["year"], errors="coerce").to_numpy(dtype=np.float64)
    seen = movies["movie_id"].isin(set(seen_before["movie_id"].unique().tolist())).to_numpy()
    return np.flatnonzero((years > cut_year) & ~seen).astype(np.int64)
