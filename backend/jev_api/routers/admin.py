"""Model registry, experiments, system stats and the audit log (admin only)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import func, select

from jev_api.deps import DB, MAX_OFFSET, AdminUser, IdPath
from jev_api.metrics import metrics
from jev_api.models import (
    AUDIT_ACTIONS,
    AuditLog,
    Experiment,
    ModelVersion,
    Rating,
    Recommendation,
    User,
    WatchHistory,
)
from jev_api.schemas import (
    AuditPage,
    ExperimentDetail,
    ExperimentOut,
    ModelVersionDetail,
    ModelVersionOut,
)
from jev_api.services import audit
from jev_api.services.feedback import feedback_counts
from jev_api.services.ml import engine_calibration
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


@router.get("/models/active/summary")
def active_summary(db: DB, request: Request) -> dict[str, Any]:
    """Public, non-sensitive facts about the serving model (used by the landing page)."""
    engine = request.app.state.engines.engine
    if engine is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no model loaded")
    m = engine.manifest
    exp = db.scalar(
        select(Experiment)
        .join(ModelVersion, Experiment.model_version_id == ModelVersion.id)
        .where(ModelVersion.version == engine.version)
    )
    comparison: dict[str, dict[str, float]] = {}
    n_users = None
    if exp is not None:
        for mt in exp.metrics:
            if (
                mt.protocol == "test"
                and mt.k == 10
                and mt.metric in ("ndcg", "recall", "precision", "hit_rate")
            ):
                comparison.setdefault(mt.model_name, {})[f"{mt.metric}@10"] = mt.value
        n_users = (exp.summary.get("n_eval_users") or {}).get("test")
    return {
        "model_version": engine.version,
        "dataset_version": m.get("dataset_version"),
        "n_items": len(engine.movies),
        "trained_on_rows": m.get("trained_on_rows"),
        "trained_at": m.get("created_at"),
        "split": m.get("split", {}).get("strategy"),
        "eval_users": n_users,
        "comparison": comparison,
        "calibration": engine_calibration(engine),  # recommendation confidence metrics, or null
    }


@router.get("/models", response_model=list[ModelVersionOut])
def list_models(_: AdminUser, db: DB) -> list[ModelVersion]:
    sync_model_versions(db)
    return list(db.scalars(select(ModelVersion).order_by(ModelVersion.trained_at.desc())).all())


@router.get("/models/{model_id}", response_model=ModelVersionDetail)
def get_model(model_id: IdPath, _: AdminUser, db: DB, request: Request) -> dict[str, Any]:
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
def activate_model(model_id: IdPath, user: AdminUser, db: DB, request: Request) -> ModelVersion:
    mv = db.get(ModelVersion, model_id)
    if mv is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "model version not found")
    previous = request.app.state.engines.engine
    try:
        request.app.state.engines.activate(mv.version)
    except (OSError, ValueError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, "model artifacts failed to load") from exc
    for other in db.scalars(select(ModelVersion)).all():
        other.is_active = other.id == mv.id
    audit.record(
        db,
        "model.activate",
        user,
        "model_version",
        mv.id,
        {"version": mv.version, "previous": previous.version if previous is not None else None},
    )
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
def get_experiment(experiment_id: IdPath, _: AdminUser, db: DB) -> dict[str, Any]:
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
    feedback: dict[str, int] = {
        kind: n
        for kind, n in db.execute(feedback_counts())  # distinct per member per movie
    }
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


@router.get("/admin/metrics")
def admin_metrics(_: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    """In-process counters (since this process started) plus DB-derived intelligence gauges.

    Groups: http_* (requests by status class and route template, errors, latency, rate limiting),
    intel_pipeline / intel_pipeline_ms / intel_stage_ms (runs, failures, durations, stage timings
    incl. forecast and lapse-prediction latency), intel_warnings (lifecycle counters),
    intel_warnings_open, intel_decisions_latest_run (by spec and answer), data_freshness, model;
    v1.1: audit_entries_by_action, intel_rows_persisted (per table, summed over runs) and
    intel_evidence_rows_per_run.
    """
    snap = metrics.snapshot()
    snap.update(request.app.state.intel.metrics_snapshot(db))
    holder = request.app.state.engines
    snap["model"] = {
        **snap.get("model", {}),
        "loaded": holder.engine is not None,
        "version": holder.engine.version if holder.engine is not None else None,
        "last_error": holder.last_error,
    }
    return snap


@router.get("/admin/audit", response_model=AuditPage)
def audit_log(
    _: AdminUser,
    db: DB,
    action: str | None = Query(None, max_length=40),
    actor: str | None = Query(None, max_length=320),
    target_type: str | None = Query(None, max_length=32),
    target_id: str | None = Query(None, max_length=200),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0, le=MAX_OFFSET),
) -> dict[str, Any]:
    """Newest first. `action` is exact (422 for an unknown one); `actor` matches the email exactly
    (case-insensitive) or "system"."""
    if action is not None and action not in AUDIT_ACTIONS:
        raise HTTPException(422, f"unknown action {action!r}; use one of: {', '.join(AUDIT_ACTIONS)}")
    q = select(AuditLog)
    if action:
        q = q.where(AuditLog.action == action)
    if actor:
        q = q.where(func.lower(AuditLog.actor) == actor.strip().lower())
    if target_type:
        q = q.where(AuditLog.target_type == target_type)
    if target_id:
        q = q.where(AuditLog.target_id == target_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(AuditLog.at.desc(), AuditLog.id.desc()).limit(limit).offset(offset)).all()
    return {"items": [audit.entry_out(e) for e in rows], "total": total, "limit": limit, "offset": offset}
