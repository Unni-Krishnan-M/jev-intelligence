"""Intelligence layer endpoints (/intel/*, admin only). Contract: docs/intelligence.md, section 6.

Run-backed lists (signals, trends, anomalies, risks, actions, predictions, series) read the latest
successful run, or `?run_id=`. DB-backed resources (runs, warnings, decisions, feedback, scenarios)
come from the intel_* tables.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session, selectinload

from jev_api.deps import DB, AdminUser
from jev_api.models import (
    FEEDBACK_VERDICTS,
    IntelDecision,
    IntelFeedback,
    IntelRun,
    IntelScenario,
    IntelWarning,
)
from jev_api.schemas import (
    IntelDecisionList,
    IntelDecisionOut,
    IntelFeedbackIn,
    IntelFeedbackList,
    IntelFeedbackOut,
    IntelPage,
    IntelRunList,
    IntelRunOut,
    IntelWarningList,
    IntelWarningOut,
    RunTrigger,
    SavedScenarioList,
    ScenarioRequest,
    Severity,
    WarningStatus,
    WarningUpdate,
)
from jev_api.services.intel import (
    IntelBusyError,
    IntelInputError,
    IntelService,
    InvalidTransitionError,
    decision_out,
    iso,
    parse_as_of,
    run_out,
    warning_out,
)

router = APIRouter(prefix="/intel", tags=["intelligence"])

NO_RUN = "no intelligence run yet"
Limit = Query(50, ge=1, le=200)
Offset = Query(0, ge=0)
HISTOGRAM_BINS = 5


def _service(request: Request) -> IntelService:
    svc: IntelService = request.app.state.intel
    return svc


def _run_and_result(db: Session, svc: IntelService, run_id: str | None) -> tuple[IntelRun, dict[str, Any]]:
    if run_id:
        run = db.scalar(select(IntelRun).where(IntelRun.run_id == run_id))
        if run is None or run.status != "succeeded":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "intelligence run not found")
    else:
        latest = svc.latest_run(db)
        if latest is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, NO_RUN)
        run = latest
    return run, svc.result_for(db, run)


def _page(
    items: list[dict[str, Any]], limit: int, offset: int, run: IntelRun, filters: dict[str, Any]
) -> dict[str, Any]:
    for field, value in filters.items():
        if value is not None:
            items = [i for i in items if i.get(field) == value]
    return {
        "items": items[offset : offset + limit],
        "total": len(items),
        "limit": limit,
        "offset": offset,
        "run_id": run.run_id,
        "as_of": iso(run.as_of),
    }


# --- runs -----------------------------------------------------------------------------------------
@router.get("/runs", response_model=IntelRunList)
def list_runs(_: AdminUser, db: DB, limit: int = Limit, offset: int = Offset) -> dict[str, Any]:
    runs = db.scalars(
        select(IntelRun).order_by(IntelRun.started_at.desc(), IntelRun.id.desc()).limit(limit).offset(offset)
    ).all()
    return {"items": [run_out(r) for r in runs], "total": db.scalar(select(func.count(IntelRun.id))) or 0}


@router.post("/runs", response_model=IntelRunOut)
def trigger_run(body: RunTrigger, user: AdminUser, request: Request) -> dict[str, Any]:
    """Runs synchronously (a few seconds) and returns the finished run; failures come back as a
    run with status "failed" and its error text."""
    try:
        run = _service(request).run("manual", body.as_of, user)
    except IntelBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except IntelInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    return run_out(run)


@router.get("/runs/{run_ref}", response_model=IntelRunOut)
def get_run(run_ref: str, _: AdminUser, db: DB) -> dict[str, Any]:
    q = select(IntelRun).where(IntelRun.id == int(run_ref)) if run_ref.isdigit() else None
    run = db.scalar(q if q is not None else select(IntelRun).where(IntelRun.run_id == run_ref))
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "intelligence run not found")
    return run_out(run)


# --- run-backed lists -------------------------------------------------------------------------------
@router.get("/signals", response_model=IntelPage)
def signals(
    _: AdminUser,
    db: DB,
    request: Request,
    run_id: str | None = None,
    kind: str | None = Query(None, max_length=32),
    entity_type: str | None = Query(None, max_length=32),
    direction: Literal["up", "down", "flat"] | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id)
    filters = {"kind": kind, "entity_type": entity_type, "direction": direction}
    return _page(res.get("signals", []), limit, offset, run, filters)


@router.get("/trends", response_model=IntelPage)
def trends(
    _: AdminUser,
    db: DB,
    request: Request,
    run_id: str | None = None,
    direction: Literal["up", "down", "flat"] | None = None,
    metric: str | None = Query(None, max_length=32),
    entity: str | None = Query(None, max_length=200),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id)
    filters = {"direction": direction, "metric": metric, "entity": entity}
    return _page(res.get("trends", []), limit, offset, run, filters)


@router.get("/anomalies", response_model=IntelPage)
def anomalies(
    _: AdminUser,
    db: DB,
    request: Request,
    run_id: str | None = None,
    kind: str | None = Query(None, max_length=32),
    severity: Severity | None = None,
    entity_type: str | None = Query(None, max_length=32),
    suppressed: bool | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id)
    filters = {"kind": kind, "severity": severity, "entity_type": entity_type, "suppressed": suppressed}
    return _page(res.get("anomalies", []), limit, offset, run, filters)


@router.get("/risks", response_model=IntelPage)
def risks(
    _: AdminUser,
    db: DB,
    request: Request,
    run_id: str | None = None,
    kind: str | None = Query(None, max_length=40),
    level: Severity | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id)
    return _page(res.get("risks", []), limit, offset, run, {"kind": kind, "level": level})


@router.get("/actions", response_model=IntelPage)
def actions(
    _: AdminUser,
    db: DB,
    request: Request,
    run_id: str | None = None,
    priority: Literal["P1", "P2", "P3"] | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id)
    return _page(res.get("actions", []), limit, offset, run, {"priority": priority})


@router.get("/predictions")
def predictions(
    _: AdminUser, db: DB, request: Request, run_id: str | None = None, series_id: str | None = None
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id)
    pred = res.get("predictions") or {}
    forecasts = pred.get("forecasts", [])
    if series_id:
        forecasts = [f for f in forecasts if f.get("series_id") == series_id]
    return {"run_id": run.run_id, "as_of": iso(run.as_of), "forecasts": forecasts, "lapse": pred.get("lapse")}


@router.get("/series/{series_id:path}")
def series(
    series_id: str, _: AdminUser, db: DB, request: Request, run_id: str | None = None
) -> dict[str, Any]:
    """A series with its trend, anomalies and forecast (ids contain colons: volume:genre:Drama)."""
    run, res = _run_and_result(db, _service(request), run_id)
    s = next((x for x in res.get("series", []) if x.get("id") == series_id), None)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "series not found")
    forecasts = (res.get("predictions") or {}).get("forecasts", [])
    return {
        "run_id": run.run_id,
        "as_of": iso(run.as_of),
        "series": s,
        "trend": next((t for t in res.get("trends", []) if t.get("series_id") == series_id), None),
        "anomalies": [a for a in res.get("anomalies", []) if a.get("series_id") == series_id],
        "forecast": next((f for f in forecasts if f.get("series_id") == series_id), None),
    }


# --- status -----------------------------------------------------------------------------------------
def _histogram(values: list[float]) -> list[dict[str, Any]]:
    width = 1.0 / HISTOGRAM_BINS
    counts = [0] * HISTOGRAM_BINS
    for v in values:
        counts[min(int(max(v, 0.0) / width), HISTOGRAM_BINS - 1)] += 1
    return [{"bin": f"{i * width:.1f}-{(i + 1) * width:.1f}", "n": n} for i, n in enumerate(counts)]


@router.get("/status")
def intel_status(_: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    svc = _service(request)
    latest_any = svc.latest_run(db, status=None)
    latest = svc.latest_run(db)
    res = svc.result_for(db, latest) if latest else {}
    engine = request.app.state.engines.engine
    model = None
    if engine is not None:
        m = engine.manifest
        trained = parse_as_of(m.get("created_at")) if m.get("created_at") else None
        now = datetime.now(UTC)
        model = {
            "version": engine.version,
            "trained_at": iso(trained),
            "age_days": round((now - trained).total_seconds() / 86400, 2) if trained else None,
            "dataset_version": m.get("dataset_version"),
        }
    decisions: list[dict[str, Any]] = []
    confidences: list[float] = []
    if latest is not None:
        rows = db.scalars(
            select(IntelDecision).where(IntelDecision.run_id == latest.run_id).order_by(IntelDecision.id)
        ).all()
        fb = _decision_feedback(db, [d.decision_id for d in rows[:5]])
        decisions = [decision_out(d, fb.get(d.decision_id)) for d in rows[:5]]
        confidences = [d.confidence for d in rows if d.confidence is not None]
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        request.app.state.log.exception("database health check failed")
        db_ok = False
    pipeline = "never_run"
    finished = db.scalar(
        select(IntelRun.status)
        .where(IntelRun.status != "running")
        .order_by(IntelRun.started_at.desc(), IntelRun.id.desc())
        .limit(1)
    )
    if finished is not None:
        pipeline = "ok" if finished == "succeeded" else "failed"
    return {
        "latest_run": run_out(latest_any) if latest_any else None,
        "summary": res.get("summary"),
        "data": res.get("data"),
        "model": model,
        "warnings_open": svc.open_warning_counts(db),
        "recent_decisions": decisions,
        "top_signals": res.get("signals", [])[:5],
        "top_risks": res.get("risks", [])[:5],
        "confidence_histogram": _histogram(confidences),
        "health": {
            "database": "ok" if db_ok else "error",
            "cache": "ok" if request.app.state.cache.ping() else "error",
            "model": "ok" if engine is not None else "unavailable",
            "pipeline": pipeline,
        },
    }


# --- warnings ---------------------------------------------------------------------------------------
@router.get("/warnings", response_model=IntelWarningList)
def list_warnings(
    _: AdminUser,
    db: DB,
    status_: WarningStatus | None = Query(None, alias="status"),
    severity: Severity | None = None,
    key: str | None = Query(None, max_length=200),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    q = select(IntelWarning)
    if status_:
        q = q.where(IntelWarning.status == status_)
    if severity:
        q = q.where(IntelWarning.severity == severity)
    if key:
        q = q.where(IntelWarning.key == key)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(
        q.order_by(IntelWarning.last_seen_at.desc(), IntelWarning.id.desc()).limit(limit).offset(offset)
    )
    return {"items": [warning_out(w) for w in rows], "total": total, "limit": limit, "offset": offset}


def _warning(db: Session, warning_id: int) -> IntelWarning:
    w = db.scalar(
        select(IntelWarning).options(selectinload(IntelWarning.events)).where(IntelWarning.id == warning_id)
    )
    if w is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "warning not found")
    return w


@router.get("/warnings/{warning_id}", response_model=IntelWarningOut)
def get_warning(warning_id: int, _: AdminUser, db: DB) -> dict[str, Any]:
    return warning_out(_warning(db, warning_id), history=True)


@router.patch("/warnings/{warning_id}", response_model=IntelWarningOut)
def update_warning(
    warning_id: int, body: WarningUpdate, user: AdminUser, db: DB, request: Request
) -> dict[str, Any]:
    w = _warning(db, warning_id)
    try:
        w = _service(request).update_warning(db, w, body.status, body.note, user)
    except InvalidTransitionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return warning_out(_warning(db, w.id), history=True)


# --- decisions --------------------------------------------------------------------------------------
def _decision_feedback(db: Session, decision_ids: list[str]) -> dict[str, dict[str, int]]:
    if not decision_ids:
        return {}
    out: dict[str, dict[str, int]] = {}
    for target, verdict, n in db.execute(
        select(IntelFeedback.target_id, IntelFeedback.verdict, func.count())
        .where(IntelFeedback.target_type == "decision", IntelFeedback.target_id.in_(set(decision_ids)))
        .group_by(IntelFeedback.target_id, IntelFeedback.verdict)
    ):
        out.setdefault(target, {})[verdict] = int(n)
    return out


@router.get("/decisions", response_model=IntelDecisionList)
def list_decisions(
    _: AdminUser,
    db: DB,
    key: str | None = Query(None, max_length=64),
    run_id: str | None = Query(None, max_length=36),
    entity: str | None = Query(None, max_length=200),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    q = select(IntelDecision)
    if key:
        q = q.where(IntelDecision.key == key)
    if run_id:
        q = q.where(IntelDecision.run_id == run_id)
    if entity:
        q = q.where(IntelDecision.entity == entity)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(IntelDecision.id.desc()).limit(limit).offset(offset)).all()
    fb = _decision_feedback(db, [d.decision_id for d in rows])
    return {
        "items": [decision_out(d, fb.get(d.decision_id)) for d in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/decisions/{decision_ref}", response_model=IntelDecisionOut)
def get_decision(decision_ref: str, _: AdminUser, db: DB) -> dict[str, Any]:
    """By db_id, or by contract id ("dec-…": the most recent run's copy of that decision)."""
    if decision_ref.isdigit():
        d = db.get(IntelDecision, int(decision_ref))
    else:
        d = db.scalar(
            select(IntelDecision)
            .where(IntelDecision.decision_id == decision_ref)
            .order_by(IntelDecision.id.desc())
            .limit(1)
        )
    if d is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "decision not found")
    return decision_out(d, _decision_feedback(db, [d.decision_id]).get(d.decision_id))


# --- feedback ---------------------------------------------------------------------------------------
def _feedback_out(f: IntelFeedback) -> dict[str, Any]:
    return {
        "id": f.id,
        "target_type": f.target_type,
        "target_id": f.target_id,
        "verdict": f.verdict,
        "note": f.note,
        "outcome": f.outcome,
        "actor": f.actor,
        "created_at": iso(f.created_at),
    }


@router.post("/feedback", response_model=IntelFeedbackOut, status_code=status.HTTP_201_CREATED)
def record_feedback(body: IntelFeedbackIn, user: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    if body.verdict not in FEEDBACK_VERDICTS[body.target_type]:
        allowed = ", ".join(FEEDBACK_VERDICTS[body.target_type])
        raise HTTPException(
            422,
            f"verdict {body.verdict!r} is not valid for a {body.target_type}; use one of: {allowed}",
        )
    svc = _service(request)
    run_id: str | None = None
    tid = body.target_id.strip()
    if body.target_type == "decision":
        run_id = db.scalar(
            select(IntelDecision.run_id)
            .where(IntelDecision.decision_id == tid)
            .order_by(IntelDecision.id.desc())
            .limit(1)
        )
        found = run_id is not None
    elif body.target_type == "warning":
        found = tid.isdigit() and db.get(IntelWarning, int(tid)) is not None
    else:
        latest = svc.latest_run(db)
        res = svc.result_for(db, latest) if latest else {}
        if body.target_type == "action":
            ids = {a.get("id") for a in res.get("actions", [])}
        else:
            ids = {f.get("id") for f in (res.get("predictions") or {}).get("forecasts", [])}
        found = tid in ids
        run_id = latest.run_id if latest and found else None
    if not found:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{body.target_type} {tid!r} not found")
    f = IntelFeedback(
        target_type=body.target_type,
        target_id=tid,
        verdict=body.verdict,
        note=body.note,
        outcome=body.outcome,
        actor=user.email,
        user_id=user.id,
        run_id=run_id,
    )
    db.add(f)
    db.commit()
    return _feedback_out(f)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


@router.get("/feedback", response_model=IntelFeedbackList)
def list_feedback(
    _: AdminUser,
    db: DB,
    target_type: Literal["decision", "warning", "action", "prediction"] | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    q = select(IntelFeedback)
    if target_type:
        q = q.where(IntelFeedback.target_type == target_type)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(IntelFeedback.id.desc()).limit(limit).offset(offset)).all()
    counts = {t: dict.fromkeys(v, 0) for t, v in FEEDBACK_VERDICTS.items()}
    for t, verdict, n in db.execute(
        select(IntelFeedback.target_type, IntelFeedback.verdict, func.count()).group_by(
            IntelFeedback.target_type, IntelFeedback.verdict
        )
    ):
        counts[t][verdict] = int(n)
    dec, warn = counts["decision"], counts["warning"]
    summary = {
        "decision": {**dec, "accuracy": _ratio(dec["correct"], dec["correct"] + dec["incorrect"])},
        "warning": {**warn, "precision": _ratio(warn["useful"], sum(warn.values()))},
        "action": counts["action"],
        "prediction": counts["prediction"],
    }
    return {
        "items": [_feedback_out(f) for f in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "summary": summary,
    }


# --- scenarios --------------------------------------------------------------------------------------
@router.post("/scenarios")
def create_scenario(body: ScenarioRequest, user: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    spec = body.model_dump(exclude={"save", "title"}, exclude_none=True)
    spec["scenarios"] = [s.model_dump(exclude_none=True) for s in body.scenarios]
    if not spec["scenarios"]:
        spec.pop("scenarios")
    if body.as_of is None and _service(request).latest_run(db) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, NO_RUN)
    try:
        out = _service(request).scenario(db, dict(spec))
    except ValueError as exc:  # IntelInputError included
        raise HTTPException(422, str(exc)) from exc
    saved_id: int | None = None
    if body.save:
        row = IntelScenario(
            title=(body.title or f"{body.series_id} · {len(out.get('scenarios', []))} scenarios")[:200],
            series_id=body.series_id,
            as_of=parse_as_of(out.get("as_of")),
            input={**spec, "as_of": body.as_of},
            output=json.loads(json.dumps(out, allow_nan=False)),
            created_by_id=user.id,
        )
        db.add(row)
        db.commit()
        saved_id = row.id
    return {**out, "id": saved_id}


@router.get("/scenarios", response_model=SavedScenarioList)
def list_scenarios(
    _: AdminUser,
    db: DB,
    series_id: str | None = Query(None, max_length=200),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    q = select(IntelScenario)
    if series_id:
        q = q.where(IntelScenario.series_id == series_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(IntelScenario.id.desc()).limit(limit).offset(offset)).all()
    items = [
        {
            "id": s.id,
            "title": s.title,
            "series_id": s.series_id,
            "created_at": iso(s.created_at),
            "input": s.input,
            "output": s.output,
        }
        for s in rows
    ]
    return {"items": items, "total": total}


# --- evaluation -------------------------------------------------------------------------------------
@router.get("/evaluation")
def evaluation(_: AdminUser, request: Request) -> dict[str, Any]:
    """The newest experiments/intel-eval-*/report.json (written by scripts/evaluate_intelligence.py)."""
    base = request.app.state.settings.experiments_dir
    runs = sorted(
        (p for p in base.glob("intel-eval-*") if (p / "report.json").is_file()), key=lambda p: p.name
    )
    if not runs:
        return {"available": False, "run_dir": None, "report": None}
    latest = runs[-1]
    try:
        report = json.loads((latest / "report.json").read_text())
    except (OSError, ValueError):
        request.app.state.log.exception("unreadable intelligence evaluation report")
        return {"available": False, "run_dir": latest.name, "report": None}
    return {"available": True, "run_dir": latest.name, "report": report}
