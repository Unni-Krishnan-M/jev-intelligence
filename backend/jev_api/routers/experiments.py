"""Online A/B experiments (admin only; docs/EXPERIMENTATION.md). The member side is transparent: members
are assigned and served inside GET /recommendations and never see which variant they are in.

Prefix ``/experiments/online``. The offline evaluation runs keep ``/experiments`` and
``/experiments/{id}`` (routers/admin.py); because that router is mounted first and ``GET
/experiments/{id}`` would capture ``GET /experiments/online``, the listing lives at
``GET /experiments/online/list`` ("list" is a reserved experiment key).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Path, Query, status

from jev_api.deps import DB, AdminUser
from jev_api.schemas.experiments import (
    KEY_PATTERN,
    ExperimentCreate,
    ExperimentList,
    ExperimentOut,
    ExperimentUpdate,
    RampIn,
)
from jev_api.services import experiments as svc
from jev_api.services.experiments import ExperimentConflictError, ExperimentError, ExperimentNotFoundError

router = APIRouter(prefix="/experiments/online", tags=["experiments"])

KeyPath = Path(pattern=KEY_PATTERN)


def _load(db: DB, key: str) -> Any:
    try:
        return svc.get_experiment(db, key)
    except ExperimentNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"online experiment {key!r} not found") from None


def _guard(fn: Any, *args: Any) -> Any:
    try:
        return fn(*args)
    except ExperimentConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from None
    except ExperimentError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None


@router.post("", response_model=ExperimentOut, status_code=status.HTTP_201_CREATED)
def create(body: ExperimentCreate, admin: AdminUser, db: DB) -> dict[str, Any]:
    """A draft experiment. Exactly one variant is the control; start it to begin serving."""
    if body.key == "list":
        raise HTTPException(422, "'list' is a reserved key")
    exp = _guard(svc.create_experiment, db, body.model_dump(mode="json"), admin)
    return svc.experiment_out(db, exp)


@router.get("/list", response_model=ExperimentList)
def list_online(
    _: AdminUser,
    db: DB,
    status_: str | None = Query(None, alias="status", pattern=r"^(draft|running|paused|stopped|concluded)$"),
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    from sqlalchemy import select

    from jev_api.models.experiments import AbExperiment

    q = select(AbExperiment).order_by(AbExperiment.created_at.desc(), AbExperiment.id.desc()).limit(limit)
    if status_:
        q = q.where(AbExperiment.status == status_)
    items = [svc.experiment_out(db, e) for e in db.scalars(q).all()]
    return {"items": items, "total": len(items)}


@router.get("/{key}", response_model=ExperimentOut)
def get_one(_: AdminUser, db: DB, key: str = KeyPath) -> dict[str, Any]:
    return svc.experiment_out(db, _load(db, key))


@router.patch("/{key}", response_model=ExperimentOut)
def update(body: ExperimentUpdate, _: AdminUser, db: DB, key: str = KeyPath) -> dict[str, Any]:
    """Draft only (409 afterwards): a started experiment's config is immutable."""
    exp = _load(db, key)
    exp = _guard(svc.update_experiment, db, exp, body.model_dump(mode="json", exclude_none=True), _)
    return svc.experiment_out(db, exp)


@router.delete("/{key}", status_code=status.HTTP_204_NO_CONTENT)
def delete(_: AdminUser, db: DB, key: str = KeyPath) -> None:
    _guard(svc.delete_experiment, db, _load(db, key))


def _transition(action: str, admin: Any, db: DB, key: str) -> dict[str, Any]:
    exp = _guard(svc.transition, db, _load(db, key), action, admin)
    return svc.experiment_out(db, exp)


@router.post("/{key}/start", response_model=ExperimentOut)
def start(admin: AdminUser, db: DB, key: str = KeyPath) -> dict[str, Any]:
    """draft | paused -> running (409 if another experiment is active on the surface)."""
    return _transition("start", admin, db, key)


@router.post("/{key}/pause", response_model=ExperimentOut)
def pause(admin: AdminUser, db: DB, key: str = KeyPath) -> dict[str, Any]:
    """running -> paused: everyone is served the default; assignments are kept."""
    return _transition("pause", admin, db, key)


@router.post("/{key}/stop", response_model=ExperimentOut)
def stop(admin: AdminUser, db: DB, key: str = KeyPath) -> dict[str, Any]:
    """running | paused -> stopped: serving returns to the default for everyone."""
    return _transition("stop", admin, db, key)


@router.post("/{key}/conclude", response_model=ExperimentOut)
def conclude(admin: AdminUser, db: DB, key: str = KeyPath) -> dict[str, Any]:
    """stopped -> concluded: the results are frozen with the decision (ship | keep_control |
    inconclusive)."""
    return _transition("conclude", admin, db, key)


@router.post("/{key}/ramp", response_model=ExperimentOut)
def ramp(body: RampIn, admin: AdminUser, db: DB, key: str = KeyPath) -> dict[str, Any]:
    """Change the enrolled traffic share. Sticky: enrolled members stay in their variant; a ramp down
    only stops new enrolment."""
    exp = _guard(svc.ramp, db, _load(db, key), body.traffic_percent, admin)
    return svc.experiment_out(db, exp)


@router.get("/{key}/results")
def results(_: AdminUser, db: DB, key: str = KeyPath) -> dict[str, Any]:
    """Per-variant metrics, statistics, SRM, guardrails and the conclusion. A concluded experiment
    returns its frozen result; otherwise outcomes are attributed and the result computed now."""
    exp = _load(db, key)
    if exp.status == "concluded" and exp.result:
        return dict(exp.result)
    return svc.compute_results(db, exp)
