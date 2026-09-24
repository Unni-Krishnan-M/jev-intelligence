"""Intelligence layer endpoints (/intel/*, admin only). Contract: docs/intelligence.md, section 6.

Run-backed lists (signals, trends, anomalies, risks, actions, predictions, series) read the latest
successful run, or `?run_id=`. DB-backed resources (runs, warnings, decisions, feedback, scenarios)
come from the intel_* tables. v1.1 (section 9.3): evidence and history read the normalised tables,
recommendation monitoring reads recommendations + recommendation_feedback, evaluation runs are
synced from experiments/intel-eval-*.

v1.2 (docs/platform.md, section 8): every endpoint takes `?domain=` (default "movie"; POST /intel/runs
also reads it from the body). An unknown key is a 404; a registered domain that cannot load on this
machine is a 409 with its reason (reads of a domain that already has stored runs keep working).
Objects addressed by id (runs, warnings, decisions) must belong to the requested domain, else 404.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status
from sqlalchemy import case, func, or_, select, text
from sqlalchemy.orm import Session, selectinload

from jev_api.deps import DB, MAX_OFFSET, AdminUser, IdPath, parse_db_id
from jev_api.models import (
    DEFAULT_DOMAIN,
    FEEDBACK_VERDICTS,
    IntelAnomalyRow,
    IntelDecision,
    IntelEvaluationRun,
    IntelEvidenceRow,
    IntelFeedback,
    IntelRiskRow,
    IntelRun,
    IntelScenario,
    IntelSignalRow,
    IntelTrendRow,
    IntelWarning,
    Recommendation,
)
from jev_api.schemas import (
    DecisionBatchList,
    EvaluationRunList,
    EvidenceOwner,
    HistoryEntity,
    IntelDecisionList,
    IntelDecisionOut,
    IntelEvidencePage,
    IntelFeedbackIn,
    IntelFeedbackList,
    IntelFeedbackOut,
    IntelHistory,
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
from jev_api.services import audit
from jev_api.services.feedback import feedback_counts, latest_feedback
from jev_api.services.intel import (
    DOMAIN_PATTERN,
    DomainUnavailableError,
    IntelBusyError,
    IntelInputError,
    IntelService,
    InvalidTransitionError,
    UnknownDomainError,
    decision_out,
    iso,
    parse_as_of,
    run_out,
    warning_out,
)
from jev_api.services.ml import engine_calibration
from jev_api.services.sync import sync_intel_evaluations

router = APIRouter(prefix="/intel", tags=["intelligence"])

NO_RUN = "no intelligence run yet"
Limit = Query(50, ge=1, le=200)
Offset = Query(0, ge=0, le=MAX_OFFSET)
RunId = Query(None, max_length=36)  # a run uuid
SeriesId = Query(None, max_length=200)
HISTOGRAM_BINS = 5


def _service(request: Request) -> IntelService:
    svc: IntelService = request.app.state.intel
    return svc


DomainQuery = Query(
    DEFAULT_DOMAIN,
    min_length=1,
    max_length=64,
    pattern=DOMAIN_PATTERN,
    description='domain adapter key: "movie" (default) or "generic:<name>" (GET /intel/domains)',
)


def unknown_domain(key: str) -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, f"unknown domain {key!r}; see GET /intel/domains")


def unavailable_domain(exc: DomainUnavailableError) -> HTTPException:
    return HTTPException(status.HTTP_409_CONFLICT, f"domain {exc.key!r} is unavailable: {exc.reason}")


def read_domain(request: Request, db: DB, domain: str = DomainQuery) -> str:
    """?domain= of a read: registered (else 404) and loadable here, or with stored runs (else 409)."""
    svc = _service(request)
    try:
        info = svc.domain(domain)
    except UnknownDomainError as exc:
        raise unknown_domain(domain) from exc
    if not info.get("available") and not svc.has_runs(db, domain):
        raise unavailable_domain(DomainUnavailableError(domain, info.get("reason")))
    return domain


Domain = Annotated[str, Depends(read_domain)]


def _resolve_run(db: Session, svc: IntelService, run_id: str | None, domain: str) -> IntelRun:
    if run_id:
        run = db.scalar(select(IntelRun).where(IntelRun.run_id == run_id, IntelRun.domain == domain))
        if run is None or run.status != "succeeded":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "intelligence run not found")
        return run
    latest = svc.latest_run(db, domain=domain)
    if latest is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, NO_RUN)
    return latest


def _run_and_result(
    db: Session, svc: IntelService, run_id: str | None, domain: str
) -> tuple[IntelRun, dict[str, Any]]:
    run = _resolve_run(db, svc, run_id, domain)
    return run, svc.result_for(db, run)


def _capability(request: Request, domain: str, capability: str, what: str) -> None:
    caps = _service(request).domain(domain).get("capabilities") or {}
    if not caps.get(capability):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"the {domain!r} domain has no {what}")


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
        "domain": run.domain,
    }


# --- domains (docs/platform.md, sections 8 and 10) ----------------------------------------------------
@router.get("/domains")
def list_domains(_: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    """Every registered domain adapter: info, capabilities (from the adapter's DomainInfo),
    availability on this machine (+ reason), its latest run (any status) and open-warning count."""
    svc = _service(request)
    open_by = svc.open_warnings_by_domain(db)
    items = []
    for d in svc.domains():
        key = d["key"]
        latest = svc.latest_run(db, status=None, domain=key)
        if key == DEFAULT_DOMAIN:
            d = svc.domain(key)  # live availability of the movie inputs
        items.append(
            {
                "key": key,
                "name": d.get("name"),
                "description": d.get("description") or "",
                "entity_types": d.get("entity_types") or [],
                "frequency": d.get("frequency"),
                "sources": d.get("sources") or [],
                "capabilities": d.get("capabilities") or {},
                "available": bool(d.get("available")),
                "reason": None if d.get("available") else d.get("reason"),
                "latest_run": run_out(latest) if latest else None,
                "warnings_open": open_by.get(key, 0),
            }
        )
    return {"items": items}


# --- runs -----------------------------------------------------------------------------------------
@router.get("/runs", response_model=IntelRunList)
def list_runs(
    _: AdminUser, db: DB, domain: Domain, limit: int = Limit, offset: int = Offset
) -> dict[str, Any]:
    runs = db.scalars(
        select(IntelRun)
        .where(IntelRun.domain == domain)
        .order_by(IntelRun.started_at.desc(), IntelRun.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    total = db.scalar(select(func.count(IntelRun.id)).where(IntelRun.domain == domain)) or 0
    return {"items": [run_out(r) for r in runs], "total": total}


@router.post("/runs", response_model=IntelRunOut)
def trigger_run(
    body: RunTrigger,
    user: AdminUser,
    request: Request,
    domain: str | None = Query(None, min_length=1, max_length=64, pattern=DOMAIN_PATTERN),
) -> dict[str, Any]:
    """Runs one domain synchronously (about a second) and returns the finished run; failures come back
    as a run with status "failed" and its error text. The domain comes from the body (or `?domain=`;
    default movie); as_of is validated against that domain's data range. Limited per admin (not per
    IP, so anonymous traffic behind the same proxy cannot use up an operator's budget) on top of the
    one-run lock."""
    if body.domain is not None and domain is not None and body.domain != domain:
        raise HTTPException(422, "the body and ?domain= name different domains")
    key = body.domain or domain or DEFAULT_DOMAIN
    window = int(time.time() // 60)
    count = request.app.state.cache.incr_window(f"rl:intel_run:{user.id}:{window}", 60)
    if count > request.app.state.settings.intel_run_rate_limit_per_minute:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many intelligence runs; try again in a minute",
            headers={"Retry-After": str(60 - int(time.time()) % 60)},
        )
    try:
        run = _service(request).run("manual", body.as_of, user, domain=key)
    except UnknownDomainError as exc:
        raise unknown_domain(key) from exc
    except DomainUnavailableError as exc:
        raise unavailable_domain(exc) from exc
    except IntelBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except IntelInputError as exc:
        raise HTTPException(422, str(exc)) from exc
    return run_out(run)


@router.get("/runs/{run_ref}", response_model=IntelRunOut)
def get_run(
    run_ref: Annotated[str, Path(max_length=64)], _: AdminUser, db: DB, domain: Domain
) -> dict[str, Any]:
    pk = parse_db_id(run_ref)
    run = db.scalar(
        select(IntelRun).where(IntelRun.id == pk if pk is not None else IntelRun.run_id == run_ref)
    )
    if run is None or run.domain != domain:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "intelligence run not found")
    return run_out(run)


# --- run-backed lists -------------------------------------------------------------------------------
@router.get("/signals", response_model=IntelPage)
def signals(
    _: AdminUser,
    db: DB,
    request: Request,
    domain: Domain,
    run_id: str | None = RunId,
    kind: str | None = Query(None, max_length=32),
    entity_type: str | None = Query(None, max_length=32),
    direction: Literal["up", "down", "flat"] | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id, domain)
    filters = {"kind": kind, "entity_type": entity_type, "direction": direction}
    return _page(res.get("signals", []), limit, offset, run, filters)


@router.get("/trends", response_model=IntelPage)
def trends(
    _: AdminUser,
    db: DB,
    request: Request,
    domain: Domain,
    run_id: str | None = RunId,
    direction: Literal["up", "down", "flat"] | None = None,
    metric: str | None = Query(None, max_length=32),
    entity: str | None = Query(None, max_length=200),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id, domain)
    filters = {"direction": direction, "metric": metric, "entity": entity}
    return _page(res.get("trends", []), limit, offset, run, filters)


@router.get("/anomalies", response_model=IntelPage)
def anomalies(
    _: AdminUser,
    db: DB,
    request: Request,
    domain: Domain,
    run_id: str | None = RunId,
    kind: str | None = Query(None, max_length=32),
    severity: Severity | None = None,
    entity_type: str | None = Query(None, max_length=32),
    suppressed: bool | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id, domain)
    filters = {"kind": kind, "severity": severity, "entity_type": entity_type, "suppressed": suppressed}
    return _page(res.get("anomalies", []), limit, offset, run, filters)


@router.get("/risks", response_model=IntelPage)
def risks(
    _: AdminUser,
    db: DB,
    request: Request,
    domain: Domain,
    run_id: str | None = RunId,
    kind: str | None = Query(None, max_length=40),
    level: Severity | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id, domain)
    return _page(res.get("risks", []), limit, offset, run, {"kind": kind, "level": level})


@router.get("/actions", response_model=IntelPage)
def actions(
    _: AdminUser,
    db: DB,
    request: Request,
    domain: Domain,
    run_id: str | None = RunId,
    priority: Literal["P1", "P2", "P3"] | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id, domain)
    return _page(res.get("actions", []), limit, offset, run, {"priority": priority})


@router.get("/predictions")
def predictions(
    _: AdminUser,
    db: DB,
    request: Request,
    domain: Domain,
    run_id: str | None = RunId,
    series_id: str | None = SeriesId,
) -> dict[str, Any]:
    run, res = _run_and_result(db, _service(request), run_id, domain)
    pred = res.get("predictions") or {}
    forecasts = pred.get("forecasts", [])
    if series_id:
        forecasts = [f for f in forecasts if f.get("series_id") == series_id]
    return {
        "run_id": run.run_id,
        "as_of": iso(run.as_of),
        "domain": run.domain,
        "forecasts": forecasts,
        "lapse": pred.get("lapse"),
    }


@router.get("/series/{series_id:path}")
def series(
    series_id: Annotated[str, Path(max_length=200)],
    _: AdminUser,
    db: DB,
    request: Request,
    domain: Domain,
    run_id: str | None = RunId,
) -> dict[str, Any]:
    """A series with its trend, anomalies and forecast (ids contain colons: volume:genre:Drama)."""
    run, res = _run_and_result(db, _service(request), run_id, domain)
    s = next((x for x in res.get("series", []) if x.get("id") == series_id), None)
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "series not found")
    forecasts = (res.get("predictions") or {}).get("forecasts", [])
    return {
        "run_id": run.run_id,
        "as_of": iso(run.as_of),
        "domain": run.domain,
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
def intel_status(_: AdminUser, db: DB, request: Request, domain: Domain) -> dict[str, Any]:
    svc = _service(request)
    info = svc.domain(domain)
    latest_any = svc.latest_run(db, status=None, domain=domain)
    latest = svc.latest_run(db, domain=domain)
    res = svc.result_for(db, latest) if latest else {}
    engine = request.app.state.engines.engine
    has_model = bool((info.get("capabilities") or {}).get("recommendation"))
    model = None
    if engine is not None and has_model:
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
        .where(IntelRun.status != "running", IntelRun.domain == domain)
        .order_by(IntelRun.started_at.desc(), IntelRun.id.desc())
        .limit(1)
    )
    if finished is not None:
        pipeline = "ok" if finished == "succeeded" else "failed"
    return {
        "domain": domain,
        "domain_info": {
            "name": info.get("name"),
            "available": bool(info.get("available")),
            "reason": None if info.get("available") else info.get("reason"),
            "capabilities": info.get("capabilities") or {},
        },
        "latest_run": run_out(latest_any) if latest_any else None,
        "summary": res.get("summary"),
        "data": res.get("data"),
        "model": model,
        "warnings_open": svc.open_warning_counts(db, domain),
        "recent_decisions": decisions,
        "top_signals": res.get("signals", [])[:5],
        "top_risks": res.get("risks", [])[:5],
        "confidence_histogram": _histogram(confidences),
        "health": {
            "database": "ok" if db_ok else "error",
            "cache": "ok" if request.app.state.cache.ping() else "error",
            # a domain without a recommender has no model to be healthy or not
            "model": ("ok" if engine is not None else "unavailable") if has_model else "not_applicable",
            "pipeline": pipeline,
        },
    }


# --- warnings ---------------------------------------------------------------------------------------
@router.get("/warnings", response_model=IntelWarningList)
def list_warnings(
    _: AdminUser,
    db: DB,
    domain: Domain,
    status_: WarningStatus | None = Query(None, alias="status"),
    severity: Severity | None = None,
    key: str | None = Query(None, max_length=200),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    q = select(IntelWarning).where(IntelWarning.domain == domain)
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


def _warning(db: Session, warning_id: int, domain: str) -> IntelWarning:
    w = db.scalar(
        select(IntelWarning).options(selectinload(IntelWarning.events)).where(IntelWarning.id == warning_id)
    )
    if w is None or w.domain != domain:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "warning not found")
    return w


@router.get("/warnings/{warning_id}", response_model=IntelWarningOut)
def get_warning(warning_id: IdPath, _: AdminUser, db: DB, domain: Domain) -> dict[str, Any]:
    return warning_out(_warning(db, warning_id, domain), history=True)


@router.patch("/warnings/{warning_id}", response_model=IntelWarningOut)
def update_warning(
    warning_id: IdPath, body: WarningUpdate, user: AdminUser, db: DB, request: Request, domain: Domain
) -> dict[str, Any]:
    w = _warning(db, warning_id, domain)
    try:
        w = _service(request).update_warning(db, w, body.status, body.note, user)
    except InvalidTransitionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return warning_out(_warning(db, w.id, domain), history=True)


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
    domain: Domain,
    key: str | None = Query(None, max_length=64),
    run_id: str | None = Query(None, max_length=36),
    entity: str | None = Query(None, max_length=200),
    batch_id: str | None = Query(None, max_length=64),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    q = select(IntelDecision).where(IntelDecision.domain == domain)
    if key:
        q = q.where(IntelDecision.key == key)
    if run_id:
        q = q.where(IntelDecision.run_id == run_id)
    if entity:
        q = q.where(IntelDecision.entity == entity)
    if batch_id:
        q = q.where(IntelDecision.batch_id == batch_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(IntelDecision.id.desc()).limit(limit).offset(offset)).all()
    fb = _decision_feedback(db, [d.decision_id for d in rows])
    return {
        "items": [decision_out(d, fb.get(d.decision_id)) for d in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/decisions/batches", response_model=DecisionBatchList)
def decision_batches(
    _: AdminUser, db: DB, request: Request, domain: Domain, run_id: str | None = Query(None, max_length=36)
) -> dict[str, Any]:
    """The run's multi-question decision calls (section 9.1). Declared before /decisions/{ref}."""
    run, res = _run_and_result(db, _service(request), run_id, domain)
    batches = res.get("decision_batches")
    if not isinstance(batches, list):
        # a result without decision_batches: rebuild the groups from the stored batch ids
        grouped: dict[str, dict[str, Any]] = {}
        for bid, did, key, pv in db.execute(
            select(
                IntelDecision.batch_id,
                IntelDecision.decision_id,
                IntelDecision.key,
                IntelDecision.policy_version,
            )
            .where(IntelDecision.run_id == run.run_id, IntelDecision.batch_id.is_not(None))
            .order_by(IntelDecision.id)
        ):
            b = grouped.setdefault(
                bid,
                {
                    "id": bid,
                    "name": None,
                    "question": None,
                    "keys": [],
                    "decision_ids": [],
                    "state_hash": None,
                    "policy_versions": {},
                },
            )
            b["decision_ids"].append(did)
            if key not in b["keys"]:
                b["keys"].append(key)
            b["policy_versions"][key] = pv
        batches = list(grouped.values())
    return {"items": batches, "run_id": run.run_id, "as_of": iso(run.as_of), "domain": run.domain}


@router.get("/decisions/{decision_ref}", response_model=IntelDecisionOut)
def get_decision(
    decision_ref: Annotated[str, Path(max_length=64)], _: AdminUser, db: DB, domain: Domain
) -> dict[str, Any]:
    """By db_id, or by contract id ("dec-…": the most recent run's copy of that decision)."""
    pk = parse_db_id(decision_ref)
    if pk is not None:
        d = db.get(IntelDecision, pk)
    else:
        d = db.scalar(
            select(IntelDecision)
            .where(IntelDecision.decision_id == decision_ref, IntelDecision.domain == domain)
            .order_by(IntelDecision.id.desc())
            .limit(1)
        )
    if d is None or d.domain != domain:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "decision not found")
    return decision_out(d, _decision_feedback(db, [d.decision_id]).get(d.decision_id))


# --- feedback ---------------------------------------------------------------------------------------
def _feedback_out(f: IntelFeedback) -> dict[str, Any]:
    return {
        "id": f.id,
        "domain": f.domain,
        "target_type": f.target_type,
        "target_id": f.target_id,
        "verdict": f.verdict,
        "note": f.note,
        "outcome": f.outcome,
        "actor": f.actor,
        "created_at": iso(f.created_at),
    }


@router.post("/feedback", response_model=IntelFeedbackOut, status_code=status.HTTP_201_CREATED)
def record_feedback(
    body: IntelFeedbackIn, user: AdminUser, db: DB, request: Request, domain: Domain
) -> dict[str, Any]:
    """The target must belong to `domain` (a decision/warning of that domain, or an action/forecast
    of its latest run); otherwise 404."""
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
            .where(IntelDecision.decision_id == tid, IntelDecision.domain == domain)
            .order_by(IntelDecision.id.desc())
            .limit(1)
        )
        found = run_id is not None
    elif body.target_type == "warning":
        pk = parse_db_id(tid)
        target = db.get(IntelWarning, pk) if pk is not None else None
        found = target is not None and target.domain == domain
    else:
        latest = svc.latest_run(db, domain=domain)
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
        domain=domain,
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
    db.flush()
    audit.record(
        db,
        "feedback.create",
        user,
        body.target_type,
        tid,
        {
            "feedback_id": f.id,
            "domain": domain,
            "verdict": body.verdict,
            "run_id": run_id,
            "has_note": bool(body.note),
        },
    )
    db.commit()
    return _feedback_out(f)


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


@router.get("/feedback", response_model=IntelFeedbackList)
def list_feedback(
    _: AdminUser,
    db: DB,
    domain: Domain,
    target_type: Literal["decision", "warning", "action", "prediction"] | None = None,
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    q = select(IntelFeedback).where(IntelFeedback.domain == domain)
    if target_type:
        q = q.where(IntelFeedback.target_type == target_type)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(IntelFeedback.id.desc()).limit(limit).offset(offset)).all()
    counts = {t: dict.fromkeys(v, 0) for t, v in FEEDBACK_VERDICTS.items()}
    for t, verdict, n in db.execute(
        select(IntelFeedback.target_type, IntelFeedback.verdict, func.count())
        .where(IntelFeedback.domain == domain)
        .group_by(IntelFeedback.target_type, IntelFeedback.verdict)
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
def create_scenario(
    body: ScenarioRequest, user: AdminUser, db: DB, request: Request, domain: Domain
) -> dict[str, Any]:
    spec = body.model_dump(exclude={"save", "title"}, exclude_none=True)
    spec["scenarios"] = [s.model_dump(exclude_none=True) for s in body.scenarios]
    if not spec["scenarios"]:
        spec.pop("scenarios")
    _capability(request, domain, "scenarios", "scenario engine")
    if body.as_of is None and _service(request).latest_run(db, domain=domain) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, NO_RUN)
    try:
        out = _service(request).scenario(db, dict(spec), domain)
    except DomainUnavailableError as exc:
        raise unavailable_domain(exc) from exc
    except ValueError as exc:  # IntelInputError included
        raise HTTPException(422, str(exc)) from exc
    saved_id: int | None = None
    if body.save:
        row = IntelScenario(
            domain=domain,
            title=(body.title or f"{body.series_id} · {len(out.get('scenarios', []))} scenarios")[:200],
            series_id=body.series_id,
            as_of=parse_as_of(out.get("as_of")),
            input={**spec, "as_of": body.as_of},
            output=json.loads(json.dumps(out, allow_nan=False)),
            created_by_id=user.id,
        )
        db.add(row)
        db.flush()
        audit.record(
            db,
            "scenario.save",
            user,
            "intel_scenario",
            row.id,
            {
                "domain": domain,
                "title": row.title,
                "series_id": row.series_id,
                "n_scenarios": len(out.get("scenarios", [])),
            },
        )
        db.commit()
        saved_id = row.id
    return {**out, "id": saved_id, "domain": domain}


@router.get("/scenarios", response_model=SavedScenarioList)
def list_scenarios(
    _: AdminUser,
    db: DB,
    domain: Domain,
    series_id: str | None = Query(None, max_length=200),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    q = select(IntelScenario).where(IntelScenario.domain == domain)
    if series_id:
        q = q.where(IntelScenario.series_id == series_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = db.scalars(q.order_by(IntelScenario.id.desc()).limit(limit).offset(offset)).all()
    items = [
        {
            "id": s.id,
            "domain": s.domain,
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
def _newest_report(request: Request, prefix: str) -> tuple[str | None, dict[str, Any] | None]:
    """(run_dir, report) of the newest experiments/<prefix>*/report.json; report None when absent or
    unreadable. NaN/Infinity become null (a JSON response cannot carry them)."""
    base = request.app.state.settings.experiments_dir
    runs = sorted((p for p in base.glob(f"{prefix}*") if (p / "report.json").is_file()), key=lambda p: p.name)
    if not runs:
        return None, None
    latest = runs[-1]
    try:
        report = json.loads((latest / "report.json").read_text(), parse_constant=lambda _: None)
    except (OSError, ValueError):
        request.app.state.log.exception(
            "unreadable evaluation report", extra={"extra_fields": {"dir": latest.name}}
        )
        return latest.name, None
    return latest.name, report if isinstance(report, dict) else None


def _platform_section(request: Request, domain: str) -> dict[str, Any] | None:
    """The domain's section of the newest experiments/platform-eval-*/report.json
    (scripts/evaluate_domains.py: forecast backtest, warning precision / FPR, decision consistency)."""
    run_dir, report = _newest_report(request, "platform-eval-")
    if report is None:
        return None
    section = next(
        (d for d in report.get("domains") or [] if isinstance(d, dict) and d.get("domain") == domain), None
    )
    if section is None:
        return None
    return {
        "run_dir": run_dir,
        "created_at": report.get("created_at"),
        "horizon": report.get("horizon"),
        "report": section,
    }


@router.get("/evaluation")
def evaluation(_: AdminUser, request: Request, domain: Domain) -> dict[str, Any]:
    """Movie: the newest experiments/intel-eval-*/report.json (scripts/evaluate_intelligence.py), as in
    v1.1. Every domain: `platform`, its section of the newest platform-eval report (null when none).
    The intel-eval report covers the movie domain only, so other domains have `available: false`."""
    platform = _platform_section(request, domain)
    if domain != DEFAULT_DOMAIN:
        return {
            "available": False,
            "run_dir": None,
            "report": None,
            "domain": domain,
            "reason": "the intelligence-layer evaluation (intel-eval) covers the movie domain; "
            "see `platform` for this domain's platform evaluation",
            "platform": platform,
        }
    run_dir, report = _newest_report(request, "intel-eval-")
    return {
        "available": report is not None,
        "run_dir": run_dir,
        "report": report,
        "domain": domain,
        "platform": platform,
    }


@router.get("/evaluation/drift")
def evaluation_drift(_: AdminUser, request: Request, domain: Domain) -> dict[str, Any]:
    """The newest experiments/drift-eval-*/report.json (scripts/evaluate_drift.py): preference-drift
    detector precision / recall and the adaptation effect behind the recommendation-strategy policy."""
    _capability(request, domain, "user_intelligence", "per-user drift evaluation")
    run_dir, report = _newest_report(request, "drift-eval-")
    return {"available": report is not None, "run_dir": run_dir, "report": report, "domain": domain}


@router.get("/evaluation/runs", response_model=EvaluationRunList)
def evaluation_runs(_: AdminUser, db: DB, request: Request, domain: Domain) -> dict[str, Any]:
    """Every offline evaluation run, newest first, with its headline numbers. Synced from
    experiments/intel-eval-*/report.json on each call (unchanged files are skipped). These evaluate
    the movie domain; other domains have none (their platform evaluation is in GET /intel/evaluation)."""
    if domain != DEFAULT_DOMAIN:
        return {"items": [], "total": 0}
    try:
        sync_intel_evaluations(db, request.app.state.settings.experiments_dir)
    except OSError:
        db.rollback()
        request.app.state.log.exception("intelligence evaluation sync failed")
    rows = db.scalars(select(IntelEvaluationRun).order_by(IntelEvaluationRun.run_dir.desc())).all()
    items = [
        {
            "id": r.id,
            "run_dir": r.run_dir,
            "created_at": iso(r.created_at),
            "pipeline_version": r.pipeline_version,
            "data_version": r.data_version,
            "headline": r.headline or {},
        }
        for r in rows
    ]
    return {"items": items, "total": len(items)}


# --- v1.1: evidence and history (normalised tables, section 9.3) -------------------------------------
def _like(term: str) -> str:
    # parameterized LIKE (SQLAlchemy binds the value); escape LIKE wildcards in user input
    return "%" + term.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _evidence_out(e: IntelEvidenceRow, run_uuid: str) -> dict[str, Any]:
    return {
        "id": e.id,
        "kind": e.kind,
        "label": e.label,
        "value": e.value_num if e.value_num is not None else e.value_text,
        "detail": e.detail,
        "ref": e.ref,
        "owner_type": e.owner_type,
        "owner_id": e.owner_id,
        "owner_title": e.owner_title,
        "position": e.position,
        "run_id": run_uuid,
    }


@router.get("/evidence", response_model=IntelEvidencePage)
def evidence(
    _: AdminUser,
    db: DB,
    request: Request,
    domain: Domain,
    run_id: str | None = RunId,
    owner_type: EvidenceOwner | None = None,
    owner_id: str | None = Query(None, max_length=200),
    kind: str | None = Query(None, max_length=16),
    q: str | None = Query(None, min_length=1, max_length=100),
    limit: int = Limit,
    offset: int = Offset,
) -> dict[str, Any]:
    """Every Evidence item of the latest (or `?run_id=`) run with its owner; `q` searches the label,
    detail and owner title (case-insensitive)."""
    run = _resolve_run(db, _service(request), run_id, domain)
    query = select(IntelEvidenceRow).where(IntelEvidenceRow.run_id == run.id)
    if owner_type:
        query = query.where(IntelEvidenceRow.owner_type == owner_type)
    if owner_id:
        query = query.where(IntelEvidenceRow.owner_id == owner_id)
    if kind:
        query = query.where(IntelEvidenceRow.kind == kind)
    if q and q.strip():
        term = _like(q.strip())
        query = query.where(
            or_(
                func.lower(IntelEvidenceRow.label).like(term, escape="\\"),
                func.lower(IntelEvidenceRow.detail).like(term, escape="\\"),
                func.lower(IntelEvidenceRow.owner_title).like(term, escape="\\"),
            )
        )
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.scalars(query.order_by(IntelEvidenceRow.id).limit(limit).offset(offset)).all()
    return {
        "items": [_evidence_out(e, run.run_id) for e in rows],
        "total": total,
        "limit": limit,
        "offset": offset,
        "run_id": run.run_id,
        "as_of": iso(run.as_of),
    }


_ANOMALY_DIRECTION = {"series_spike": "up", "series_drop": "down"}


def _history_point(entity: str, row: Any) -> dict[str, Any]:
    """value / score / level / direction per entity (docs/api.md lists the mapping)."""
    if entity == "signals":
        return {
            "id": row.signal_id,
            "observed_at": iso(row.observed_at),
            "value": row.value,
            "score": row.strength,
            "level": None,
            "direction": row.direction,
        }
    if entity == "risks":
        return {
            "id": row.risk_id,
            "observed_at": None,
            "value": row.exposure,
            "score": row.score,
            "level": row.level,
            "direction": None,
        }
    if entity == "trends":
        return {
            "id": row.trend_id,
            "observed_at": None,
            "value": row.slope,
            "score": row.evidence_strength,
            "level": None,
            "direction": row.direction,
        }
    return {
        "id": row.anomaly_id,
        "observed_at": iso(row.detected_at),
        "value": row.value,
        "score": row.score,
        "level": row.severity,
        "direction": _ANOMALY_DIRECTION.get(row.kind),
    }


# entity -> (table, key column, object id column)
HISTORY: dict[str, tuple[Any, Any, Any]] = {
    "signals": (IntelSignalRow, IntelSignalRow.dedup_key, IntelSignalRow.signal_id),
    "risks": (IntelRiskRow, IntelRiskRow.key, IntelRiskRow.risk_id),
    "trends": (IntelTrendRow, IntelTrendRow.series_id, IntelTrendRow.trend_id),
    "anomalies": (IntelAnomalyRow, IntelAnomalyRow.dedup_key, IntelAnomalyRow.anomaly_id),
}


@router.get("/history/{entity}", response_model=IntelHistory)
def history(
    entity: HistoryEntity,
    _: AdminUser,
    db: DB,
    domain: Domain,
    key: str = Query(..., min_length=1, max_length=200),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """One object across successful runs, oldest -> newest (the most recent `limit` points).
    `key` = signal/anomaly dedup_key, trend series_id, risk key (risk:<kind>:<entity>), or any
    object id of that entity, which resolves to its key."""
    model, key_col, id_col = HISTORY[entity]
    in_domain = (model.run_id == IntelRun.id, IntelRun.domain == domain)
    resolved = key
    if (
        db.scalar(
            select(func.count())
            .select_from(model)
            .join(IntelRun, in_domain[0])
            .where(key_col == key, in_domain[1])
        )
        == 0
    ):
        by_id = db.scalar(
            select(key_col).join(IntelRun, in_domain[0]).where(id_col == key, in_domain[1]).limit(1)
        )
        resolved = by_id if by_id is not None else key
    rows = db.execute(
        select(model, IntelRun)
        .join(IntelRun, model.run_id == IntelRun.id)
        .where(key_col == resolved, IntelRun.status == "succeeded", IntelRun.domain == domain)
        .order_by(IntelRun.started_at.desc(), IntelRun.id.desc(), model.id.desc())
        .limit(limit)
    ).all()
    items = [
        {
            "run_id": run.run_id,
            "as_of": iso(run.as_of),
            "created_at": iso(run.started_at),
            **_history_point(entity, row),
        }
        for row, run in reversed(rows)
    ]
    return {"entity": entity, "key": resolved, "items": items, "domain": domain}


# --- v1.1: recommender monitoring (section 9.3) ------------------------------------------------------
FEEDBACK_KINDS = ("like", "dislike", "not_interested", "clicked")
CONFIDENCE_BINS = 10
SERVED_DAYS = 14


def _positive_rate(c: dict[str, int]) -> float | None:
    # explicit verdicts only: clicks are not a judgement of the recommendation
    return _ratio(c["like"], c["like"] + c["dislike"] + c["not_interested"])


@router.get("/recommendations")
def recommender_monitoring(
    _: AdminUser, db: DB, request: Request, domain: Domain, recent: int = Query(20, ge=0, le=100)
) -> dict[str, Any]:
    """Serving volume, feedback by reason code, the confidence distribution of served items and the
    active model's calibration (null until models/<version>/calibration.json exists). v1.2: served
    recommendations per strategy (the recommendation_strategy decision). Recommender domains only."""
    _capability(request, domain, "recommendation", "recommender")
    engine = request.app.state.engines.engine
    since = datetime.now(UTC) - timedelta(days=SERVED_DAYS)
    day = func.date(Recommendation.created_at)
    per_day = db.execute(
        select(day, func.count()).where(Recommendation.created_at >= since).group_by(day).order_by(day)
    ).all()
    served_total = db.scalar(select(func.count(Recommendation.id))) or 0
    with_conf = db.scalar(select(func.count(Recommendation.id)).where(Recommendation.confidence.is_not(None)))
    totals = dict.fromkeys(FEEDBACK_KINDS, 0)
    # distinct verdicts: the newest verdict (and click) per member per movie, however often it was sent
    for kind, n in db.execute(feedback_counts()):
        totals[kind] = int(n)
    codes: dict[str, dict[str, Any]] = {}
    for code, n in db.execute(
        select(Recommendation.reason_code, func.count()).group_by(Recommendation.reason_code)
    ):
        codes[code] = {"code": code, "served": int(n), **dict.fromkeys(FEEDBACK_KINDS, 0)}
    # feedback attributed to a served recommendation (recommendation_id set)
    latest = latest_feedback()
    for code, kind, n in db.execute(
        select(Recommendation.reason_code, latest.c.feedback, func.count())
        .join(Recommendation, latest.c.recommendation_id == Recommendation.id)
        .group_by(Recommendation.reason_code, latest.c.feedback)
    ):
        codes.setdefault(code, {"code": code, "served": 0, **dict.fromkeys(FEEDBACK_KINDS, 0)})[kind] = int(n)
    reason_codes = sorted(
        ({**c, "positive_rate": _positive_rate(c)} for c in codes.values()),
        key=lambda c: (-c["served"], c["code"]),
    )
    # CASE buckets, portable across SQLite and PostgreSQL (CAST rounds on PostgreSQL)
    # edges as i / n, not i * (1 / n): 7 * 0.1 > 0.7 would put 0.7 in the 0.6-0.7 bin
    edges = [i / CONFIDENCE_BINS for i in range(CONFIDENCE_BINS + 1)]
    bucket = case(
        *[(Recommendation.confidence < edges[i + 1], i) for i in range(CONFIDENCE_BINS - 1)],
        else_=CONFIDENCE_BINS - 1,
    )
    counts = [0] * CONFIDENCE_BINS
    for b, n in db.execute(
        select(bucket, func.count()).where(Recommendation.confidence.is_not(None)).group_by(bucket)
    ):
        counts[int(b)] = int(n)
    rows = (
        db.scalars(
            select(Recommendation)
            .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
            .limit(recent)
        ).all()
        if recent
        else []
    )
    strategies = {
        str(k or "unrecorded"): int(n)
        for k, n in db.execute(
            select(Recommendation.strategy, func.count()).group_by(Recommendation.strategy)
        )
    }
    return {
        "domain": domain,
        "strategies": strategies,
        "model_version": engine.version if engine is not None else None,
        "calibration": engine_calibration(engine),
        "served": {
            "total": served_total,
            "with_confidence": with_conf or 0,
            "per_day": [{"date": str(d), "count": c} for d, c in per_day],
        },
        "feedback_totals": totals,
        "reason_codes": reason_codes,
        "confidence_histogram": [
            {"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": n} for i, n in enumerate(counts)
        ],
        "recent": [
            {
                "id": r.id,
                "user_id": r.user_id,
                "movie_id": r.movie_id,
                "title": r.movie.title,
                "rank": r.rank,
                "score": r.score,
                "confidence": r.confidence,
                "confidence_kind": r.confidence_kind,
                "reason": r.reason,
                "reason_code": r.reason_code,
                "model_version": r.model_version,
                "decision_id": r.decision_id,
                "strategy": r.strategy,
                "created_at": iso(r.created_at),
            }
            for r in rows
        ],
    }
