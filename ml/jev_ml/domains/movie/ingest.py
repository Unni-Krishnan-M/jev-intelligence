"""INGEST + VALIDATE: pipeline inputs, source provenance/freshness and data-quality checks.

``PipelineInputs`` is the only thing the pipeline reads. ``load_default_inputs`` fills it from the
real files (processed MovieLens data, the active model manifest and its experiment run); the API
fills the ``app_*`` frames from its database.

Leakage guard: every MovieLens row with ``timestamp > as_of`` is dropped here, before any other
stage sees the data, and the number dropped is reported. App (live) events are filtered by the wall
clock ``now`` in a live run (they are monitored in real time) and by ``as_of`` in a replay (an
explicit as_of, Phase 2 P2.4): a replay never mixes later app events into an earlier analysis, and
its live-window stages (live feedback, app freshness) run on the replay clock.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import iso, iso_from_epoch, to_utc
from jev_ml.core.quality import Checks, quality_evidence, source_row, to_epoch
from jev_ml.data.dataset import load_dataset_meta, load_interactions, load_movies, split_list
from jev_ml.domains.movie.config import IntelConfig

INTERACTION_COLUMNS = ("user_id", "movie_id", "rating", "timestamp")
APP_RATING_COLUMNS = ("user_id", "movie_id", "rating", "timestamp")
APP_FEEDBACK_COLUMNS = ("feedback", "timestamp")
APP_SERVED_COLUMNS = ("movie_id", "timestamp")
FEEDBACK_VALUES = ("like", "dislike", "not_interested", "clicked")
NEGATIVE_FEEDBACK = ("dislike", "not_interested")
NO_GENRE = "(no genres listed)"
MOVIELENS_URL = "https://grouplens.org/datasets/movielens/"
MOVIELENS_LICENSE = (
    "GroupLens research licence (ml-latest-small README): research use with attribution, "
    "no commercial use without permission; redistribution under the same terms"
)


@dataclass
class PipelineInputs:
    """Everything a run needs. Timestamps in the frames are UNIX seconds (datetimes are accepted
    for the app frames and converted).

    suppressed_keys: warning/anomaly dedup keys an operator dismissed. The value is either the
    severity the key had when it was dismissed ("low".."critical") -- then a later occurrence with a
    *higher* severity is treated as an escalation and is not suppressed -- or any other string,
    which is kept as the suppression reason and suppresses the key at every severity.
    """

    interactions: pd.DataFrame
    movies: pd.DataFrame
    app_ratings: pd.DataFrame | None = None
    app_feedback: pd.DataFrame | None = None
    app_served: pd.DataFrame | None = None
    model_manifest: dict[str, Any] | None = None
    experiment: dict[str, Any] | None = None
    per_user_ndcg: dict[str, list[float]] | None = None
    dataset_meta: dict[str, Any] | None = None
    as_of: datetime | None = None
    now: datetime | None = None
    suppressed_keys: dict[str, str] = field(default_factory=dict)
    data_version: str | None = None


@dataclass
class Prepared:
    """Validated, as_of-filtered data shared by all later stages."""

    ratings: pd.DataFrame  # MovieLens rows <= as_of, cleaned
    movies: pd.DataFrame
    genre_of: dict[int, list[str]]
    genres: list[str]
    app_ratings: pd.DataFrame
    app_feedback: pd.DataFrame
    app_served: pd.DataFrame
    as_of: datetime
    as_of_ts: float
    now: datetime
    now_ts: float
    data_version: str
    model_version: str | None
    model_cutoff_ts: float | None  # last event the active model was trained on (None = unknown)
    model_replayable: bool  # False when the model saw data after as_of (replay leakage guard)
    sources: list[dict[str, Any]]
    quality: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    excluded_after_as_of: int
    inputs: PipelineInputs | None = None  # the raw inputs (model manifest, per-user NDCG, ...)
    # P2.4: True for an explicit as_of; app (live) events are then cut at, and windowed on, as_of
    replay: bool = False
    event_clock_ts: float | None = None  # the clock of the app-event stages (None: now_ts)


# --------------------------------------------------------------------------------------------------
# loading real inputs


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def load_default_inputs(
    as_of: datetime | str | None = None,
    now: datetime | None = None,
    processed_dir: Path | None = None,
    models_dir: Path | None = None,
    experiments_dir: Path | None = None,
    model_version: str | None = None,
) -> PipelineInputs:
    """Read the real processed data, the active model manifest and its experiment run.

    Paths default to the project layout in ``jev_ml.paths`` (overridable with JEV_* env vars).
    Missing model/experiment files give ``None`` fields (the pipeline then skips those stages).

    Replay determinism: without ``model_version`` the manifest is the one *active at call time*
    (registry.json), so a promotion or rollback between two replays of the same ``as_of`` changes
    the model-based stages. The version used is recorded in ``run.model_version`` of every result
    (and ``intel_runs.model_version``); pass ``model_version`` to pin a replay to that manifest.
    """
    from jev_ml import paths

    processed_dir = processed_dir or paths.PROCESSED_DIR
    models_dir = models_dir or paths.MODELS_DIR
    experiments_dir = experiments_dir or paths.EXPERIMENTS_DIR
    interactions = load_interactions(processed_dir / "interactions.csv")
    movies = load_movies(processed_dir / "movies.csv")
    meta_path = processed_dir / "dataset_meta.json"
    meta = load_dataset_meta(meta_path) if meta_path.exists() else None

    manifest = experiment = per_user = None
    if model_version is not None and Path(model_version).name != model_version:
        raise ValueError(f"model_version must be a plain version name, got {model_version!r}")
    reg = _read_json(models_dir / "registry.json") or {}
    active = model_version or reg.get("active")
    if active:
        manifest = _read_json(models_dir / active / "manifest.json")
    if manifest and manifest.get("experiment_run"):
        run_dir = experiments_dir / str(manifest["experiment_run"])
        experiment = _read_json(run_dir / "metrics.json")
        per_user = _read_json(run_dir / "per_user_ndcg10.json")
    return PipelineInputs(
        interactions=interactions,
        movies=movies,
        model_manifest=manifest,
        experiment=experiment,
        per_user_ndcg=per_user,
        dataset_meta=meta,
        as_of=to_utc(as_of) if as_of is not None else None,
        now=now,
    )


# --------------------------------------------------------------------------------------------------
# helpers


_to_epoch = to_epoch


def _empty(cols: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=np.float64) for c in cols})


_Checks = Checks


def _validate_ratings(
    df: pd.DataFrame,
    source: str,
    movie_ids: set[int],
    cfg: IntelConfig,
    checks: _Checks,
    upper_ts: float,
    upper_label: str,
    now_ts: float,
) -> tuple[pd.DataFrame, int]:
    """Common rating checks. Returns (clean rows <= upper_ts, rows dropped for being > upper_ts)."""
    missing = [c for c in INTERACTION_COLUMNS if c not in df.columns]
    checks.add(
        f"{source}_schema",
        not missing,
        len(missing),
        0,
        "error",
        f"missing columns: {missing}" if missing else "all required columns present",
        source,
    )
    if missing:
        return _empty(INTERACTION_COLUMNS), 0
    d = df[list(INTERACTION_COLUMNS)].copy()
    d["timestamp"] = _to_epoch(d["timestamp"])
    # leakage guard first: rows after the upper bound never influence any check or stage
    after = d["timestamp"] > upper_ts
    n_after = int(after.sum())
    d = d[~after]
    n0 = len(d)
    nulls = int(d.isna().any(axis=1).sum())
    null_rate = nulls / n0 if n0 else 0.0
    checks.add(
        f"{source}_null_rate",
        nulls == 0,
        null_rate,
        0,
        "error",
        f"{nulls} of {n0} rows have a null in a required column",
        source,
    )
    d = d.dropna()
    d["rating"] = d["rating"].astype(np.float64)
    bad_range = ~d["rating"].between(cfg.rating_min, cfg.rating_max)
    checks.add(
        f"{source}_rating_range",
        not bad_range.any(),
        int(bad_range.sum()),
        0,
        "error",
        f"{int(bad_range.sum())} ratings outside [{cfg.rating_min}, {cfg.rating_max}]",
        source,
    )
    off_grid = ~np.isclose((d["rating"] * 2) % 1, 0) & ~bad_range
    checks.add(
        f"{source}_rating_grid",
        not off_grid.any(),
        int(off_grid.sum()),
        0,
        "warning",
        f"{int(off_grid.sum())} ratings not on the 0.5-star grid",
        source,
    )
    d = d[~bad_range]
    d["user_id"] = d["user_id"].astype(np.int64)
    d["movie_id"] = d["movie_id"].astype(np.int64)
    unknown = ~d["movie_id"].isin(movie_ids)
    checks.add(
        f"{source}_unknown_movies",
        not unknown.any(),
        int(unknown.sum()),
        0,
        "error",
        f"{int(unknown.sum())} ratings reference movie ids missing from the catalogue",
        source,
    )
    d = d[~unknown]
    ts = d["timestamp"]
    bad_ts = (ts < cfg.earliest_valid_ts) | (ts > now_ts)
    checks.add(
        f"{source}_timestamp_range",
        not bad_ts.any(),
        int(bad_ts.sum()),
        0,
        "error",
        f"{int(bad_ts.sum())} timestamps before 1995-01-01 or after the wall clock",
        source,
    )
    d = d[~bad_ts]
    dup = d.duplicated(["user_id", "movie_id", "timestamp"])
    checks.add(
        f"{source}_duplicates",
        not dup.any(),
        int(dup.sum()),
        0,
        "warning",
        f"{int(dup.sum())} duplicate (user, movie, timestamp) rows (kept once)",
        source,
    )
    d = d[~dup]
    # a user re-rating the same film is legitimate, but more than one rating per (user, movie)
    # in the *same second* is not; beyond that, check that per-user event order is sane:
    # the share of users whose ratings all carry one identical timestamp (bulk imports)
    if len(d):
        g = d.groupby("user_id")["timestamp"]
        span = g.max() - g.min()
        multi = g.size() > 1
        frozen = int(((span == 0) & multi).sum())
        n_users = int(multi.sum())
        share = frozen / n_users if n_users else 0.0
    else:
        frozen, share = 0, 0.0
    checks.add(
        f"{source}_timestamp_sanity",
        share <= 0.05,
        share,
        0.05,
        "warning",
        f"{frozen} users with >1 rating have all ratings at one identical second",
        source,
    )
    if upper_label:
        checks.add(
            f"{source}_after_{upper_label}",
            True,
            n_after,
            0,
            "info",
            f"{n_after} rows after {upper_label} excluded from this run (leakage guard)",
            source,
        )
    d = d.sort_values(["timestamp", "user_id", "movie_id"], kind="mergesort").reset_index(drop=True)
    return d, n_after


def _validate_events(
    df: pd.DataFrame | None, name: str, cols: tuple[str, ...], checks: _Checks, now_ts: float
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return _empty(cols)
    missing = [c for c in cols if c not in df.columns]
    checks.add(
        f"{name}_schema",
        not missing,
        len(missing),
        0,
        "error",
        f"missing columns: {missing}" if missing else "all required columns present",
        "app",
    )
    if missing:
        return _empty(cols)
    d = df.copy()
    d["timestamp"] = _to_epoch(d["timestamp"])
    bad = d["timestamp"].isna() | (d["timestamp"] > now_ts)
    d = d[~bad]
    if "feedback" in cols:
        bad_fb = ~d["feedback"].isin(FEEDBACK_VALUES)
        checks.add(
            f"{name}_values",
            not bad_fb.any(),
            int(bad_fb.sum()),
            0,
            "warning",
            f"{int(bad_fb.sum())} feedback rows with a value outside {list(FEEDBACK_VALUES)}",
            "app",
        )
        d = d[~bad_fb]
    return d.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


_source_row = source_row


def model_cutoff(manifest: dict[str, Any] | None, meta: dict[str, Any] | None) -> float | None:
    """Last event timestamp the model was trained on.

    Uses an explicit manifest field when present; otherwise, when the model was trained on this
    dataset version (``dataset_version`` matches) the snapshot's last timestamp is the cutoff.
    """
    if not manifest:
        return None
    for k in ("data_cutoff_ts", "trained_until_ts"):
        if manifest.get(k) is not None:
            return float(manifest[k])
    if meta and manifest.get("dataset_version") == meta.get("dataset_version"):
        last = (meta.get("validation") or {}).get("last_ts")
        return float(last) if last is not None else None
    return None


def prepare(inputs: PipelineInputs, cfg: IntelConfig) -> Prepared:
    now = to_utc(inputs.now) or datetime.now(UTC)
    now_ts = now.timestamp()
    checks = _Checks(cfg.check_weights)
    diagnostics: list[dict[str, Any]] = []

    movies = inputs.movies.copy()
    movie_ids = {int(m) for m in movies["movie_id"]}
    genre_of: dict[int, list[str]] = {}
    for mid, g in zip(movies["movie_id"], movies.get("genres", pd.Series([""] * len(movies))), strict=True):
        genre_of[int(mid)] = [x for x in split_list(g) if x != NO_GENRE]
    no_genre = sum(1 for v in genre_of.values() if not v)
    checks.add(
        "catalogue_genres",
        no_genre / max(len(genre_of), 1) <= 0.05,
        no_genre / max(len(genre_of), 1),
        0.05,
        "info",
        f"{no_genre} of {len(genre_of)} films have no genre",
        "movielens",
    )

    raw = inputs.interactions
    ts_raw = _to_epoch(raw["timestamp"]) if "timestamp" in raw.columns else pd.Series(dtype=float)
    if inputs.as_of is not None:
        as_of = to_utc(inputs.as_of)
        assert as_of is not None
    else:
        valid = ts_raw[(ts_raw <= now_ts) & (ts_raw >= cfg.earliest_valid_ts)]
        if not len(valid):
            raise ValueError("no valid MovieLens timestamps to derive as_of from")
        as_of = datetime.fromtimestamp(float(valid.max()), tz=UTC)
    as_of_ts = as_of.timestamp()

    ratings, n_after = _validate_ratings(raw, "movielens", movie_ids, cfg, checks, as_of_ts, "as_of", now_ts)
    if not len(ratings):
        raise ValueError(f"no MovieLens ratings at or before as_of={iso(as_of)}")

    # Phase 2 (P2.4): a replay (explicit as_of) sees app events only up to as_of, and every live-window
    # stage (live feedback, app freshness) runs on that clock; a live run keeps the wall clock
    replay = inputs.as_of is not None
    clock_ts = min(as_of_ts, now_ts) if replay else now_ts
    app_r_raw = inputs.app_ratings
    if app_r_raw is not None and len(app_r_raw):
        app_r, _ = _validate_ratings(
            app_r_raw, "app_ratings", movie_ids, cfg, checks, clock_ts, "as_of" if replay else "", now_ts
        )
    else:
        app_r = _empty(APP_RATING_COLUMNS)
    app_f = _validate_events(inputs.app_feedback, "app_feedback", APP_FEEDBACK_COLUMNS, checks, clock_ts)
    app_s = _validate_events(inputs.app_served, "app_served", APP_SERVED_COLUMNS, checks, clock_ts)

    meta = inputs.dataset_meta or {}
    data_version = inputs.data_version or meta.get("dataset_version") or "unknown"
    manifest = inputs.model_manifest
    model_version = manifest.get("version") if manifest else None
    cutoff = model_cutoff(manifest, meta)
    replayable = bool(manifest) and (cutoff is None or cutoff <= as_of_ts + 1e-6)
    if manifest and cutoff is not None and cutoff > as_of_ts + 1e-6:
        diagnostics.append(
            {
                "stage": "model",
                "status": "skipped",
                "reason": f"active model was trained on data up to {iso_from_epoch(cutoff)}, after "
                f"as_of {iso(as_of)}; model-based risks and decisions are not replayed (leakage guard)",
            }
        )

    sources = [
        _source_row(
            "movielens",
            "static_snapshot",
            len(ratings),
            float(ratings["timestamp"].min()),
            float(ratings["timestamp"].max()),
            now_ts,
            as_of_ts,
            None,
            None,
            "archival snapshot: its age is reported, not alarmed",
        )
    ]
    sources[0].update(
        {
            "license": MOVIELENS_LICENSE,
            "url": MOVIELENS_URL,
            "checksum": f"md5:{meta['source_md5']}" if meta.get("source_md5") else None,
        }
    )
    app_ts = pd.concat([app_r["timestamp"], app_f["timestamp"], app_s["timestamp"]], ignore_index=True)
    if len(app_ts):
        last = float(app_ts.max())
        fresh = (clock_ts - last) / 86400.0 <= cfg.live_fresh_days
        sources.append(
            _source_row(
                "app",
                "live",
                len(app_ts),
                float(app_ts.min()),
                last,
                clock_ts,
                None,
                "daily",
                bool(fresh),
                f"{len(app_r)} ratings, {len(app_f)} feedback, {len(app_s)} served recommendations",
            )
        )
        sources[-1].update(
            {
                "sla_days": cfg.live_fresh_days,
                "staleness_days": sources[-1]["age_days"],
                "stale_response": "Check the app's event ingestion (API health, background jobs).",
            }
        )
    else:
        sources.append(
            _source_row(
                "app",
                "live",
                0,
                None,
                None,
                now_ts,
                None,
                "daily",
                None,
                "no app events recorded yet: freshness cannot be assessed",
            )
        )
    if manifest:
        created = pd.Timestamp(str(manifest["created_at"])) if manifest.get("created_at") else None
        created_ts = created.timestamp() if created is not None else None
        sources.append(
            {
                "source": "model",
                "kind": "artefact",
                "rows": int(manifest.get("trained_on_rows") or 0),
                "first_event": None,
                "last_event": iso_from_epoch(cutoff),
                "age_days": round((now_ts - created_ts) / 86400.0, 2) if created_ts else None,
                "lag_days": round((as_of_ts - cutoff) / 86400.0, 2) if cutoff is not None else None,
                "expected_update": None,
                "fresh": None,
                "detail": f"model {model_version} trained {iso(created) if created is not None else '?'}",
            }
        )

    quality = {"score": round(checks.score(), 4), "checks": checks.items}
    return Prepared(
        ratings=ratings,
        movies=movies,
        genre_of=genre_of,
        genres=sorted({g for v in genre_of.values() for g in v}),
        app_ratings=app_r,
        app_feedback=app_f,
        app_served=app_s,
        as_of=as_of,
        as_of_ts=as_of_ts,
        now=now,
        now_ts=now_ts,
        data_version=str(data_version),
        model_version=model_version,
        model_cutoff_ts=cutoff,
        model_replayable=replayable,
        sources=sources,
        quality=quality,
        diagnostics=diagnostics,
        excluded_after_as_of=n_after,
        inputs=inputs,
        replay=replay,
        event_clock_ts=clock_ts,
    )


__all__ = [
    "APP_FEEDBACK_COLUMNS",
    "APP_RATING_COLUMNS",
    "APP_SERVED_COLUMNS",
    "FEEDBACK_VALUES",
    "INTERACTION_COLUMNS",
    "NEGATIVE_FEEDBACK",
    "NO_GENRE",
    "PipelineInputs",
    "Prepared",
    "load_default_inputs",
    "model_cutoff",
    "prepare",
    "quality_evidence",
]
