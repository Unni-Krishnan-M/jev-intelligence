"""Versioned training snapshots: MovieLens processed data + app feedback, content-hashed.

A snapshot is a ``data/processed``-shaped directory, so the training public API
(``run_pipeline(processed_dir=...)``) and calibration read it unchanged:

    data/snapshots/<snapshot_id>/
        interactions.csv      MovieLens rows + app rows (user_id, movie_id, rating, timestamp)
        movies.csv            the base catalogue, byte for byte
        dataset_meta.json     base meta with dataset_version = snapshot_id
        app_interactions.csv  provenance of every app row (which signal produced it)
        exclusions.csv        (user, movie) pairs a member marked not_interested
        manifest.json         sources, row counts, cut-off, watermark, semantics, file hashes

How app feedback becomes training rows (``SEMANTICS``). Training consumes one row per (user, movie)
whose ``rating`` sets both the implicit confidence (``signals.rating_weight``) and the signed taste
(``signals.preference_weight``). App signals are mapped to the pseudo-rating whose training weight
matches the serving weight in ``signals.py`` (engine.build_profile):

* an explicit app rating is kept as is and wins over every other signal on that film;
* a latest recommendation verdict of ``dislike`` becomes rating 0.5 (weight 0.1, taste -1): the same
  treatment MovieLens gives a very low rating. Implicit CF still counts it as weak consumption: the
  training API has no true negatives;
* a latest verdict of ``not_interested`` is an exclusion: no row is written for that film (the
  member's favourite/watch/like signals on it are dropped) and the pair is listed in
  exclusions.csv. Serving already excludes these films; training cannot express an exclusion;
* otherwise the strongest positive signal: favourite -> 5.0 (weight 2.0 = FAVORITE_WEIGHT),
  onboarding pick -> 4.5 (1.5 = ONBOARDING_PICK_WEIGHT), like -> 4.0 (1.0, the relevance
  threshold), watch -> 3.5 (0.5; WATCH_WEIGHT is 0.6, which no rating maps to exactly);
* ``clicked`` is an interaction, not a judgement, and is ignored.

App users are kept distinguishable: their ids are ``app_user_id + user_id_offset`` (default 10^7,
far above MovieLens ids). Rows after the cut-off are dropped, as are films outside the base
catalogue (counted in the manifest).

The snapshot id is ``snap-`` + the first 16 hex digits of a SHA-256 over the semantics version and
the bytes of interactions.csv, movies.csv and exclusions.csv. The same inputs (same DB, same base
data, same cut-off) always give the same id; ``created_at`` is recorded but not hashed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.data.dataset import load_dataset_meta, load_interactions
from jev_ml.signals import (
    DISLIKE_THRESHOLD,
    FAVORITE_WEIGHT,
    LIKE_THRESHOLD,
    ONBOARDING_PICK_WEIGHT,
    WATCH_WEIGHT,
    rating_weight,
)

SEMANTICS_VERSION = "app-feedback-1.0"
DEFAULT_USER_OFFSET = 10_000_000
APP_EVENT_COLUMNS = ("app_user_id", "movie_id", "kind", "value", "ts")
APP_KINDS = ("rating", "favorite", "onboarding", "watch", "like", "dislike", "not_interested", "clicked")
VERDICTS = ("like", "dislike", "not_interested")
POSITIVE_ORDER = ("favorite", "onboarding", "like", "watch")  # strongest first


def pseudo_rating(weight: float) -> float:
    """The MovieLens-scale rating whose training weight is closest to a serving weight.

    rating_weight is 1 + (r - 4) for r >= 4, 0.5 on (2.5, 4) and 0.1 at or below 2.5."""
    if weight >= 1.0:
        return float(min(5.0, LIKE_THRESHOLD + (weight - 1.0)))
    if weight > 0.1:
        return 3.5  # weight 0.5, the only value on (2.5, 4); taste +0.25
    return 0.5


PSEUDO_RATINGS = {
    "favorite": pseudo_rating(FAVORITE_WEIGHT),  # 5.0
    "onboarding": pseudo_rating(ONBOARDING_PICK_WEIGHT),  # 4.5
    "like": LIKE_THRESHOLD,  # 4.0: a like is relevant, like a rating >= 4
    "watch": pseudo_rating(WATCH_WEIGHT),  # 3.5
    "dislike": 0.5,  # <= DISLIKE_THRESHOLD
}
assert PSEUDO_RATINGS["dislike"] <= DISLIKE_THRESHOLD

SEMANTICS: dict[str, Any] = {
    "version": SEMANTICS_VERSION,
    "pseudo_ratings": PSEUDO_RATINGS,
    "training_weights": {k: rating_weight(v) for k, v in PSEUDO_RATINGS.items()},
    "serving_weights": {
        "favorite": FAVORITE_WEIGHT,
        "onboarding": ONBOARDING_PICK_WEIGHT,
        "watch": WATCH_WEIGHT,
    },
    "precedence": "app rating > latest verdict dislike/not_interested > favorite > onboarding > like > watch",
    "not_interested": "exclusion: no training row for that (user, film); listed in exclusions.csv",
    "clicked": "ignored (an interaction, not a judgement)",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_ts(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return int(value.timestamp())


def collapse_app_events(
    events: pd.DataFrame, movie_ids: np.ndarray, cutoff_ts: int | None, user_offset: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """App events -> (training rows with provenance, exclusions, counts). Deterministic."""
    ev = events.loc[:, list(APP_EVENT_COLUMNS)].copy()
    counts: dict[str, int] = {f"events_{k}": int((ev["kind"] == k).sum()) for k in APP_KINDS}
    unknown_kind = ~ev["kind"].isin(APP_KINDS)
    if unknown_kind.any():
        raise ValueError(f"unknown app event kinds: {sorted(set(ev.loc[unknown_kind, 'kind']))}")
    ev["ts"] = ev["ts"].astype(np.int64)
    ev["movie_id"] = ev["movie_id"].astype(np.int64)
    ev["app_user_id"] = ev["app_user_id"].astype(np.int64)
    after = ev["ts"] > cutoff_ts if cutoff_ts is not None else pd.Series(False, index=ev.index)
    counts["dropped_after_cutoff"] = int(after.sum())
    ev = ev[~after]
    unknown = ~ev["movie_id"].isin(movie_ids)
    counts["dropped_unknown_movie"] = int(unknown.sum())
    ev = ev[~unknown & (ev["kind"] != "clicked")]
    # a stable order so "latest" is well defined even for equal timestamps
    ev = ev.sort_values(["app_user_id", "movie_id", "ts", "kind"], kind="mergesort")

    rows: list[tuple[int, int, int, float, int, str]] = []
    excl: list[tuple[int, int, int]] = []
    for (uid, mid), g in ev.groupby(["app_user_id", "movie_id"], sort=True):
        u, m = int(uid), int(mid)  # type: ignore[call-overload]  # pandas types group keys as Hashable
        ratings = g[g["kind"] == "rating"]
        if len(ratings):
            last = ratings.iloc[-1]
            rows.append((u + user_offset, u, m, float(last["value"]), int(last["ts"]), "rating"))
            continue
        verdicts = g[g["kind"].isin(VERDICTS)]
        verdict = verdicts.iloc[-1] if len(verdicts) else None
        if verdict is not None and verdict["kind"] == "not_interested":
            excl.append((u + user_offset, m, int(verdict["ts"])))
            continue
        if verdict is not None and verdict["kind"] == "dislike":
            rows.append((u + user_offset, u, m, PSEUDO_RATINGS["dislike"], int(verdict["ts"]), "dislike"))
            continue
        for kind in POSITIVE_ORDER:
            hit = g[g["kind"] == kind]
            if kind == "like" and (verdict is None or verdict["kind"] != "like"):
                continue
            if len(hit):
                ts = int(hit["ts"].iloc[0] if kind == "watch" else hit["ts"].iloc[-1])  # first watch
                rows.append((u + user_offset, u, m, PSEUDO_RATINGS[kind], ts, kind))
                break
    app = pd.DataFrame(rows, columns=["user_id", "app_user_id", "movie_id", "rating", "timestamp", "signal"])
    exclusions = pd.DataFrame(excl, columns=["user_id", "movie_id", "timestamp"])
    counts["app_rows"] = len(app)
    counts["app_users"] = int(app["user_id"].nunique()) if len(app) else 0
    counts["exclusions"] = len(exclusions)
    for kind in ("rating", *POSITIVE_ORDER, "dislike"):
        counts[f"rows_from_{kind}"] = int((app["signal"] == kind).sum()) if len(app) else 0
    return app, exclusions, counts


def build_snapshot(
    base_dir: Path,
    app_events: pd.DataFrame,
    out_root: Path,
    cutoff: datetime | None = None,
    user_offset: int = DEFAULT_USER_OFFSET,
    watermark: dict[str, Any] | None = None,
    sources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write (or reuse) the snapshot of ``base_dir`` + ``app_events`` and return its manifest.

    ``app_events`` has the columns APP_EVENT_COLUMNS (``ts`` in epoch seconds, ``value`` the rating
    for kind == "rating"). ``cutoff`` defaults to the newest app event, so the same database always
    gives the same snapshot. The returned manifest has ``reused: True`` when an identical snapshot
    already existed."""
    base = load_interactions(base_dir / "interactions.csv")
    meta = load_dataset_meta(base_dir / "dataset_meta.json")
    movie_ids = pd.read_csv(base_dir / "movies.csv", usecols=["movie_id"])["movie_id"].to_numpy()
    if len(base) and int(base["user_id"].max()) >= user_offset:
        raise ValueError(f"base user ids reach {base['user_id'].max()}: raise the app user offset")
    events = app_events if len(app_events) else pd.DataFrame(columns=list(APP_EVENT_COLUMNS))
    cutoff_ts = _to_ts(cutoff)
    if cutoff_ts is None and len(events):
        cutoff_ts = int(pd.to_numeric(events["ts"]).max())
    app, exclusions, counts = collapse_app_events(events, movie_ids, cutoff_ts, user_offset)

    inter = pd.concat([base, app[["user_id", "movie_id", "rating", "timestamp"]]], ignore_index=True)
    inter = inter.astype(
        {"user_id": np.int64, "movie_id": np.int64, "rating": np.float64, "timestamp": np.int64}
    )
    inter = inter.sort_values(["user_id", "timestamp", "movie_id"], kind="mergesort").reset_index(drop=True)

    out_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out_root, prefix=".tmp-snap-"))
    try:
        inter.to_csv(tmp / "interactions.csv", index=False)
        app.to_csv(tmp / "app_interactions.csv", index=False)
        exclusions.to_csv(tmp / "exclusions.csv", index=False)
        shutil.copyfile(base_dir / "movies.csv", tmp / "movies.csv")
        h = hashlib.sha256(SEMANTICS_VERSION.encode())
        for name in ("interactions.csv", "movies.csv", "exclusions.csv"):
            h.update(name.encode())
            h.update(_sha256(tmp / name).encode())
        content_hash = h.hexdigest()
        snapshot_id = f"snap-{content_hash[:16]}"
        final = out_root / snapshot_id
        if (final / "manifest.json").exists():
            existing = load_snapshot_manifest(final)
            if existing.get("content_hash") == content_hash:
                return {**existing, "reused": True}
        manifest = {
            "snapshot_id": snapshot_id,
            "content_hash": content_hash,
            "created_at": datetime.now(UTC).isoformat(),
            "cutoff": None if cutoff_ts is None else datetime.fromtimestamp(cutoff_ts, UTC).isoformat(),
            "cutoff_ts": cutoff_ts,
            "base_dataset_version": meta.get("dataset_version"),
            "user_id_offset": user_offset,
            "semantics": SEMANTICS,
            "watermark": watermark or {},
            "sources": {
                "movielens": {
                    "dataset_version": meta.get("dataset_version"),
                    "rows": len(base),
                    "users": int(base["user_id"].nunique()),
                },
                "app": counts,
                **(sources or {}),
            },
            "row_counts": {
                "base": len(base),
                "app": len(app),
                "total": len(inter),
                "exclusions": len(exclusions),
                "users": int(inter["user_id"].nunique()),
                "movies": len(movie_ids),
            },
            "files": {p.name: _sha256(p) for p in sorted(tmp.iterdir()) if p.is_file()},
        }
        snap_meta = {
            **meta,
            "dataset_version": snapshot_id,
            "base_dataset_version": meta.get("dataset_version"),
            "snapshot": {k: manifest[k] for k in ("snapshot_id", "content_hash", "cutoff", "row_counts")},
        }
        (tmp / "dataset_meta.json").write_text(json.dumps(snap_meta, indent=2, sort_keys=True))
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))
        for p in tmp.iterdir():
            os.chmod(p, 0o644)
        os.chmod(tmp, 0o755)  # noqa: S103 - API and trainer may run as different users (like registry.json)
        if final.exists():  # a half-written or foreign directory with the same name: replace it
            shutil.rmtree(final)
        os.replace(tmp, final)
        return {**manifest, "reused": False}
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def load_snapshot_manifest(snapshot_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((snapshot_dir / "manifest.json").read_text())
    return data
