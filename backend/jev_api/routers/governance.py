"""Retraining and model governance API (admin only; docs/RETRAINING_AND_MODEL_GOVERNANCE.md).

GET  /governance/snapshots                  versioned training snapshots
POST /governance/snapshots                  export one now (reused when the data is unchanged)
GET  /governance/jobs[/{job_id}]            retrain and evaluate jobs: status, timings, steps, error
GET  /governance/models                     every version with its state, gate results and blockers
GET  /governance/models/{version}/lineage   snapshot, config hash, git commit, seed, metrics, gate
POST /governance/retrain                    202: snapshot -> train -> candidate -> gate (one at a time)
POST /governance/models/{version}/evaluate  202: gate an existing version against the active one
POST /governance/models/{version}/promote   swap to it; the gate must have passed unless force + reason
POST /governance/models/rollback            restore the version that served before the active one
"""

from __future__ import annotations

from typing import Annotated, Any, NoReturn

from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from sqlalchemy import select

from jev_api.deps import DB, AdminUser
from jev_api.models import ModelVersion
from jev_api.models.enums import TRAINING_JOB_STATUSES
from jev_api.models.governance import DatasetSnapshot, ModelGovernance, TrainingJob
from jev_api.schemas.governance import (
    EvaluateRequest,
    GovernedModelList,
    PromoteRequest,
    PromotionResult,
    RetrainRequest,
    RollbackRequest,
    SnapshotOut,
    TrainingJobOut,
)
from jev_api.services import governance as gov
from jev_api.services.sync import sync_model_versions
from jev_ml.registry import load_manifest, read_registry

router = APIRouter(prefix="/governance", tags=["governance"])

# a model version names a directory under models/: never let a path through
VersionPath = Annotated[str, Path(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")]
JobPath = Annotated[str, Path(pattern=r"^[0-9a-f-]{36}$")]


def _raise(exc: gov.GovernanceError) -> NoReturn:
    raise gov.http_error(exc) from exc


def _known(request: Request, version: str) -> None:
    if version not in read_registry(request.app.state.settings.models_dir)["versions"]:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown model version")


@router.get("/snapshots", response_model=list[SnapshotOut])
def list_snapshots(_: AdminUser, db: DB, limit: int = Query(50, ge=1, le=200)) -> list[DatasetSnapshot]:
    q = select(DatasetSnapshot).order_by(DatasetSnapshot.created_at.desc(), DatasetSnapshot.id.desc())
    return list(db.scalars(q.limit(limit)).all())


@router.post("/snapshots", response_model=SnapshotOut)
def create_snapshot(user: AdminUser, db: DB, request: Request) -> DatasetSnapshot:
    """Export a snapshot now (idempotent: the same data gives the same snapshot id)."""
    svc = gov.get_service(request.app)
    row, _ = gov.export_snapshot(db, svc.settings, actor=user)
    return row


@router.get("/jobs", response_model=list[TrainingJobOut])
def list_jobs(
    _: AdminUser,
    db: DB,
    job_status: str | None = Query(None, alias="status", max_length=16),
    limit: int = Query(50, ge=1, le=200),
) -> list[TrainingJob]:
    if job_status is not None and job_status not in TRAINING_JOB_STATUSES:
        raise HTTPException(422, f"status must be one of {', '.join(TRAINING_JOB_STATUSES)}")
    q = select(TrainingJob)
    if job_status:
        q = q.where(TrainingJob.status == job_status)
    return list(
        db.scalars(q.order_by(TrainingJob.created_at.desc(), TrainingJob.id.desc()).limit(limit)).all()
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobOut)
def get_job(job_id: JobPath, _: AdminUser, db: DB) -> TrainingJob:
    job = db.scalar(select(TrainingJob).where(TrainingJob.job_id == job_id))
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job


@router.get("/models", response_model=GovernedModelList)
def list_governed_models(_: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    sync_model_versions(db)
    reg = read_registry(settings.models_dir)
    mvs = {m.version: m for m in db.scalars(select(ModelVersion)).all()}
    rows = db.scalars(select(ModelGovernance)).all()
    items = [gov.governed_model(db, r, reg["active"], mvs.get(r.version)) for r in rows]
    items.sort(key=lambda m: (m["trained_at"] is None, m["trained_at"]), reverse=True)
    return {"active": reg["active"], "previous": reg.get("previous"), "items": items}


@router.get("/models/{version}/lineage")
def lineage(version: VersionPath, _: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    """Everything needed to reproduce and audit a version: snapshot (with its manifest), config hash,
    git commit, seed, training metrics, the job and decision behind it, the gate and the registry
    history of the version."""
    settings = request.app.state.settings
    _known(request, version)
    try:
        row = gov.governance_row(db, version)
    except gov.GovernanceError as exc:
        _raise(exc)
    manifest = load_manifest(version, settings.models_dir)
    snap = None
    if row.snapshot_id:
        s = db.scalar(select(DatasetSnapshot).where(DatasetSnapshot.snapshot_id == row.snapshot_id))
        if s is not None:
            snap = {
                **SnapshotOut.model_validate(s).model_dump(mode="json"),
                "manifest": gov.snapshot_manifest(s),
            }
    job = db.scalar(select(TrainingJob).where(TrainingJob.job_id == row.job_id)) if row.job_id else None
    reg = read_registry(settings.models_dir)
    return {
        "version": version,
        "state": row.state,
        "snapshot": snap,
        "snapshot_id": row.snapshot_id,
        "config_hash": row.config_hash,
        "git_commit": row.git_commit or manifest.get("git_commit"),
        "seed": row.seed if row.seed is not None else manifest.get("training_seed"),
        "dataset_version": manifest.get("dataset_version"),
        "experiment_run": manifest.get("experiment_run"),
        "training_metrics": manifest.get("metrics", {}),
        "training_config": manifest.get("training_config", {}),
        "lineage": row.lineage,
        "job": TrainingJobOut.model_validate(job).model_dump(mode="json") if job else None,
        "decision_id": row.decision_id,
        "gate": row.gate,
        "promoted_at": gov.aware(row.promoted_at),
        "promoted_by": row.promoted_by,
        "forced": row.forced,
        "force_reason": row.force_reason,
        "history": [h for h in reg["history"] if h.get("version") == version or h.get("previous") == version],
    }


@router.post("/retrain", response_model=TrainingJobOut, status_code=status.HTTP_202_ACCEPTED)
def retrain(body: RetrainRequest, user: AdminUser, db: DB, request: Request) -> TrainingJob:
    """Start snapshot -> train -> candidate -> gate in a worker thread (409 while a job runs)."""
    svc = gov.get_service(request.app)
    try:
        return svc.submit(
            db, kind="retrain", trigger="manual", actor=user, quick=body.quick, config=body.config
        )
    except gov.GovernanceError as exc:
        _raise(exc)


@router.post(
    "/models/{version}/evaluate", response_model=TrainingJobOut, status_code=status.HTTP_202_ACCEPTED
)
def evaluate(
    version: VersionPath, body: EvaluateRequest, user: AdminUser, db: DB, request: Request
) -> TrainingJob:
    """Gate an existing version against the active model (on its snapshot) in a worker thread."""
    _known(request, version)
    svc = gov.get_service(request.app)
    try:
        return svc.submit(
            db, kind="evaluate", trigger="manual", actor=user, quick=body.quick, version=version
        )
    except gov.GovernanceError as exc:
        _raise(exc)


@router.post("/models/{version}/promote", response_model=PromotionResult)
def promote(
    version: VersionPath, body: PromoteRequest, user: AdminUser, db: DB, request: Request
) -> dict[str, Any]:
    """409 with the blockers unless the gate passed against the current active model; ``force``
    (with a reason) overrides the gate, never the load check. Audited as model.promote either way."""
    _known(request, version)
    svc = gov.get_service(request.app)
    try:
        return svc.promote(db, version, user, force=body.force, reason=body.reason)
    except gov.GovernanceError as exc:
        _raise(exc)


@router.post("/models/rollback", response_model=PromotionResult)
def rollback(body: RollbackRequest, user: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    svc = gov.get_service(request.app)
    try:
        return svc.rollback(db, user, reason=body.reason)
    except gov.GovernanceError as exc:
        _raise(exc)
