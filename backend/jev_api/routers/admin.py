"""Model registry, experiments and system stats (admin only)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import func, select

from jev_api.deps import DB, AdminUser
from jev_api.models import (
    Experiment,
    ModelVersion,
    Rating,
    Recommendation,
    RecommendationFeedback,
    User,
    WatchHistory,
)
from jev_api.schemas import ExperimentDetail, ExperimentOut, ModelVersionDetail, ModelVersionOut
from jev_api.services.sync import sync_experiments, sync_model_versions
from jev_ml.registry import load_manifest

router = APIRouter(tags=["models"])

HEADLINE = ("ndcg", "recall", "precision", "hit_rate")


def _headline(exp: Experiment) -> dict[str, float]:
    return {
        f"{m.metric}@{m.k}": m.value
        for m in exp.metrics
        if m.model_name == "hybrid" and m.protocol == "test" and m.k == 10 and m.metric in HEADLINE
    }


@router.get("/models", response_model=list[ModelVersionOut])
def list_models(_: AdminUser, db: DB) -> list[ModelVersion]:
    sync_model_versions(db)
    return list(db.scalars(select(ModelVersion).order_by(ModelVersion.trained_at.desc())).all())


@router.get("/models/{model_id}", response_model=ModelVersionDetail)
def get_model(model_id: int, _: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    out = ModelVersionDetail.model_validate(mv).model_dump()
    try:
        manifest = load_manifest(mv.version, request.app.state.settings.models_dir)
        out["manifest"] = {
            k: manifest[k]
            for k in (
                "components",
                "hybrid_config",
                "artifact_files",
                "als_loss_history",
                "split",
                "git_commit",
                "python",
                "trained_on_rows",
                "n_items",
            )
            if k in manifest
        }
    except FileNotFoundError:
        out["manifest"] = {"error": "artifact directory missing"}
    return out


@router.post("/models/{model_id}/activate", response_model=ModelVersionOut)
def activate_model(model_id: int, _: AdminUser, db: DB, request: Request) -> ModelVersion:
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    try:
        request.app.state.engines.activate(mv.version)
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "model artifacts failed to load") from exc
    for other in db.scalars(select(ModelVersion)).all():
        other.is_active = other.id == mv.id
    db.commit()
    request.app.state.cache.delete_prefix("rec:")
    return mv


@router.get("/experiments", response_model=list[ExperimentOut])
def list_experiments(_: AdminUser, db: DB) -> list[dict[str, Any]]:
    sync_model_versions(db)
    sync_experiments(db)
    exps = db.scalars(select(Experiment).order_by(Experiment.created_at.desc())).all()
    return [
        {
            **ExperimentOut.model_validate(e).model_dump(exclude={"model_version", "headline"}),
            "model_version": e.model_version.version if e.model_version else None,
            "headline": _headline(e),
        }
        for e in exps
    ]


@router.get("/experiments/{experiment_id}", response_model=ExperimentDetail)
def get_experiment(experiment_id: int, _: AdminUser, db: DB) -> dict[str, Any]:
    e = db.get(Experiment, experiment_id)
    if e is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "experiment not found")
    base = ExperimentDetail.model_validate(e).model_dump(exclude={"model_version", "headline"})
    return {
        **base,
        "model_version": e.model_version.version if e.model_version else None,
        "headline": _headline(e),
    }


@router.get("/admin/stats")
def stats(_: AdminUser, db: DB) -> dict[str, Any]:
    since = datetime.now(UTC) - timedelta(days=14)
    day = func.date(Recommendation.created_at)
    per_day = db.execute(
        select(day, func.count()).where(Recommendation.created_at >= since).group_by(day).order_by(day)
    ).all()
    feedback = dict(
        db.execute(
            select(RecommendationFeedback.feedback, func.count()).group_by(RecommendationFeedback.feedback)
        ).all()
    )
    reasons = db.execute(
        select(Recommendation.reason_code, func.count())
        .group_by(Recommendation.reason_code)
        .order_by(func.count().desc())
        .limit(10)
    ).all()
    return {
        "users": db.scalar(select(func.count(User.id))) or 0,
        "ratings": db.scalar(select(func.count(Rating.id))) or 0,
        "watches": db.scalar(select(func.count(WatchHistory.id))) or 0,
        "recommendations_served": db.scalar(select(func.count(Recommendation.id))) or 0,
        "feedback": feedback,
        "recommendations_per_day": [{"date": str(d), "count": c} for d, c in per_day],
        "reason_codes": [{"code": c, "count": n} for c, n in reasons],
    }
