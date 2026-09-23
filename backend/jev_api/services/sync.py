"""Idempotent sync of files → database: movie catalog, model versions, experiments + metrics."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from jev_api.config import get_settings
from jev_api.models import (
    EvaluationMetric,
    Experiment,
    Genre,
    ModelVersion,
    Movie,
    MovieGenre,
    User,
)
from jev_api.security import hash_password
from jev_ml.data.dataset import load_movies, split_list
from jev_ml.paths import PROCESSED_DIR
from jev_ml.registry import active_version, list_manifests

log = logging.getLogger(__name__)
_METRIC_KEY = re.compile(r"^(?P<metric>[a-z_]+)@(?P<k>\d+)$")


def _int_or_none(v: Any) -> int | None:
    return None if v is None or pd.isna(v) else int(v)


def seed_catalog(db: Session, movies_csv: Path | None = None) -> int:
    """Load movies + genres if the catalog is empty. Returns number of movies inserted."""
    if db.scalar(select(Movie.id).limit(1)) is not None:
        return 0
    settings = get_settings()
    if movies_csv is None:
        version = active_version(settings.models_dir)
        candidate = settings.models_dir / version / "movies.csv" if version else None
        movies_csv = candidate if candidate and candidate.exists() else PROCESSED_DIR / "movies.csv"
    if not movies_csv.exists():
        log.warning("no movie catalog found at %s; run the data pipeline", movies_csv)
        return 0
    df = load_movies(movies_csv)
    genre_names = sorted({g for gs in df["genres"] for g in split_list(gs)})
    genres = {name: Genre(name=name) for name in genre_names}
    db.add_all(genres.values())
    db.flush()
    rows, links = [], []
    for r in df.to_dict("records"):
        directors, cast = split_list(r["directors"]), split_list(r["cast"])
        rows.append(
            {
                "id": int(r["movie_id"]),
                "title": r["title"],
                "year": _int_or_none(r["year"]),
                "description": r["description"] or None,
                "runtime_min": _int_or_none(r["runtime_min"]),
                "directors": directors,
                "cast": cast,
                "keywords": split_list(r["keywords"]),
                "tags": split_list(r["tags"])[:25],
                "countries": split_list(r["countries"]),
                "imdb_id": r["imdb_id"] or None,
                "tmdb_id": _int_or_none(r["tmdb_id"]),
                "n_ratings": int(r["n_ratings"]),
                "mean_rating": None if pd.isna(r["mean_rating"]) else float(r["mean_rating"]),
                "search_text": " ".join([r["title"], *directors, *cast[:10]]).lower(),
            }
        )
        links.extend(
            {"movie_id": int(r["movie_id"]), "genre_id": genres[g].id} for g in split_list(r["genres"])
        )
    db.execute(Movie.__table__.insert(), rows)
    db.execute(MovieGenre.__table__.insert(), links)
    db.commit()
    log.info("seeded catalog: %d movies, %d genres", len(rows), len(genres))
    return len(rows)


def sync_model_versions(db: Session) -> int:
    settings = get_settings()
    active = active_version(settings.models_dir)
    n = 0
    for m in list_manifests(settings.models_dir):
        mv = db.scalar(select(ModelVersion).where(ModelVersion.version == m["version"]))
        if mv is None:
            mv = ModelVersion(
                version=m["version"],
                model_type=m.get("model_type", "hybrid"),
                dataset_version=m["dataset_version"],
                training_config=m.get("training_config", {}),
                training_seed=int(m.get("training_seed", 0)),
                metrics=m.get("metrics", {}),
                artifact_path=str(Path(m["artifact_path"]).name),
                artifact_bytes=int(m.get("artifact_bytes", 0)),
                trained_at=datetime.fromisoformat(m["created_at"]),
            )
            db.add(mv)
            n += 1
        mv.is_active = m["version"] == active
    db.commit()
    return n


def sync_experiments(db: Session) -> int:
    settings = get_settings()
    n = 0
    for path in sorted(settings.experiments_dir.glob("*/metrics.json")):
        data = json.loads(path.read_text())
        run_id = data["run_id"]
        if db.scalar(select(Experiment.id).where(Experiment.run_id == run_id)) is not None:
            continue
        mv_id = None
        if data.get("model_version"):
            mv_id = db.scalar(select(ModelVersion.id).where(ModelVersion.version == data["model_version"]))
        tuning = data.get("tuning") or {}
        exp = Experiment(
            run_id=run_id,
            experiment_name=data["experiment_name"],
            dataset_version=data["dataset_version"],
            model_type="hybrid",
            parameters=data["config"].get("models_tuned", {}),
            training_seed=int(data["training_seed"]),
            split=data.get("split", {}),
            summary={
                "seconds_total": data.get("seconds_total"),
                "quick": data.get("quick"),
                "n_eval_users": data["metrics"].get("n_eval_users"),
                "selection_metric": tuning.get("selection_metric"),
                "tuning_trials": {k: len(v) for k, v in (tuning.get("trials") or {}).items()},
                "als_loss_history": data.get("als_loss_history", []),
                "evaluation": data["config"].get("evaluation", {}),
            },
            model_version_id=mv_id,
            created_at=datetime.fromisoformat(data["created_at"]),
        )
        for protocol in ("test", "cold_start"):
            for model_name, metrics in (data["metrics"].get(protocol) or {}).items():
                for key, value in metrics.items():
                    m = _METRIC_KEY.match(key)
                    metric, k = (m.group("metric"), int(m.group("k"))) if m else (key, 0)
                    exp.metrics.append(
                        EvaluationMetric(
                            model_name=model_name, protocol=protocol, metric=metric, k=k, value=float(value)
                        )
                    )
        db.add(exp)
        n += 1
    db.commit()
    return n


def ensure_admin(db: Session) -> None:
    s = get_settings()
    if not s.admin_email or not s.admin_password:
        return
    email = s.admin_email.lower()
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        db.add(
            User(
                email=email,
                password_hash=hash_password(s.admin_password.get_secret_value()),
                display_name="Admin",
                is_admin=True,
                onboarding_completed=True,
            )
        )
        db.commit()
        log.info("created admin account from JEV_ADMIN_EMAIL")
    elif not user.is_admin:
        user.is_admin = True
        db.commit()


def sync_all(db: Session) -> dict[str, int]:
    return {
        "movies": seed_catalog(db),
        "model_versions": sync_model_versions(db),
        "experiments": sync_experiments(db),
    }
