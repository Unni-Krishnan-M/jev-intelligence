"""Intelligence layer service: gather inputs, run the pipeline, persist runs/decisions/warnings.

The pipeline itself (ml/jev_ml/intel) is pure. This module owns everything stateful around it:
- inputs: processed MovieLens files and the model manifest (load_default_inputs) plus the app DB
  (ratings, recommendation feedback, served recommendations, dismissed warning keys),
- one run at a time (a process-wide lock; a second trigger gets IntelBusyError -> HTTP 409),
- persistence of the run (running -> succeeded | failed), its decisions and the warning lifecycle
  (docs/intelligence.md, section 5),
- a small cache of parsed run results, so list endpoints do not re-read ~0.5 MB of JSON per call.

Pipeline failures are recorded on the run row and never propagate to the API.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from jev_api.cache import Cache
from jev_api.config import Settings
from jev_api.db import SessionLocal
from jev_api.metrics import metrics
from jev_api.models import (
    WARNING_OPEN_STATUSES,
    IntelDecision,
    IntelRun,
    IntelWarning,
    IntelWarningEvent,
    Rating,
    Recommendation,
    RecommendationFeedback,
    User,
)
from jev_api.services.ml import EngineHolder
from jev_ml.intel import PIPELINE_VERSION, IntelConfig, PipelineInputs, load_default_inputs, run_pipeline
from jev_ml.intel.config import severity_rank
from jev_ml.intel.pipeline import PipelineResult
from jev_ml.intel.scenario import run_scenario

log = logging.getLogger(__name__)

# PATCH /intel/warnings/{id}: resolved and dismissed are terminal
TRANSITIONS: dict[str, tuple[str, ...]] = {
    "new": ("acknowledged", "investigating", "resolved", "dismissed"),
    "acknowledged": ("investigating", "resolved", "dismissed"),
    "investigating": ("resolved", "dismissed"),
    "resolved": (),
    "dismissed": (),
}
RESULT_CACHE_SIZE = 4
DAY_S = 86400.0


class IntelBusyError(Exception):
    """A pipeline run is already in progress."""


class IntelInputError(ValueError):
    """The caller asked for something the data cannot answer (bad date, as_of out of range)."""


class InvalidTransitionError(Exception):
    pass


def iso(dt: datetime | None) -> str | None:
    """ISO-8601 UTC with a Z suffix. SQLite returns naive datetimes; they are stored as UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def parse_as_of(value: str | datetime | None) -> datetime | None:
    """ "YYYY-MM-DD" (UTC midnight) or an ISO-8601 timestamp -> aware UTC datetime."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, datetime):
        return aware(value)
    try:
        dt = datetime.fromisoformat(value.strip())
    except ValueError as exc:
        raise IntelInputError("as_of must be YYYY-MM-DD or an ISO-8601 timestamp") from exc
    return aware(dt)


# --- serialisation ------------------------------------------------------------------------------
def run_out(run: IntelRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "run_id": run.run_id,
        "trigger": run.trigger,
        "status": run.status,
        "as_of": iso(run.as_of or run.requested_as_of),
        "started_at": iso(run.started_at),
        "finished_at": iso(run.finished_at),
        "duration_ms": run.duration_ms,
        "pipeline_version": run.pipeline_version,
        "data_version": run.data_version,
        "model_version": run.model_version,
        "summary": run.summary,
        "stage_ms": run.stage_ms or {},
        "error": run.error,
    }


def event_out(e: IntelWarningEvent) -> dict[str, Any]:
    return {
        "from_status": e.from_status,
        "to_status": e.to_status,
        "note": e.note,
        "actor": e.actor,
        "at": iso(e.at),
    }


def warning_out(w: IntelWarning, history: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": w.id,
        "key": w.key,
        "title": w.title,
        "description": w.description,
        "severity": w.severity,
        "confidence": w.confidence,
        "confidence_kind": w.confidence_kind,
        "status": w.status,
        "trigger": w.trigger or {},
        "evidence": w.evidence or [],
        "recommended_action": w.recommended_action,
        "source": w.source or {},
        "entity_type": w.entity_type,
        "entity": w.entity,
        "detected_at": iso(w.detected_at),
        "last_seen_at": iso(w.last_seen_at),
        "updated_at": iso(w.updated_at),
        "occurrences": w.occurrences,
        "first_seen_run_id": w.first_seen_run_id,
        "last_seen_run_id": w.last_seen_run_id,
        "reopened_from": w.reopened_from,
        "suppressed_until": iso(w.suppressed_until),
    }
    if history:
        out["history"] = [event_out(e) for e in w.events]
    return out


def decision_out(d: IntelDecision, feedback: dict[str, int] | None = None) -> dict[str, Any]:
    fb = feedback or {}
    return {
        "id": d.decision_id,
        "key": d.key,
        "spec_id": d.spec_id,
        "policy_version": d.policy_version,
        "question": d.question,
        "kind": d.kind,
        "options": d.options or [],
        "answer": d.answer,
        "option_scores": d.option_scores or {},
        "confidence": d.confidence,
        "confidence_kind": d.confidence_kind,
        "state": d.state or {},
        "rationale": d.rationale or [],
        "evidence": d.evidence or [],
        "abstained": d.abstained,
        "fallback_reason": d.fallback_reason,
        "entity_type": d.entity_type,
        "entity": d.entity,
        "db_id": d.id,
        "run_id": d.run_id,
        "as_of": iso(d.as_of),
        "created_at": iso(d.created_at),
        "feedback": {"correct": fb.get("correct", 0), "incorrect": fb.get("incorrect", 0)},
    }


def _cut(value: Any, n: int) -> str | None:
    return None if value is None else str(value)[:n]


# --- the service --------------------------------------------------------------------------------
class IntelService:
    def __init__(self, settings: Settings, engines: EngineHolder, cache: Cache) -> None:
        self.settings = settings
        self.engines = engines
        self.cache = cache
        self.config = IntelConfig()
        self._lock = threading.Lock()
        self._results_lock = threading.Lock()
        self._results: OrderedDict[str, dict[str, Any]] = OrderedDict()
        # the latest run's PipelineResult object: scenarios reuse its fitted series and models
        self._latest_obj: tuple[str, PipelineResult] | None = None
        # replaced in tests with a synthetic dataset; takes the requested as_of (or None)
        self.inputs_factory: Callable[[datetime | None], PipelineInputs] = self._file_inputs

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    # ---- inputs ---------------------------------------------------------------------------------
    def _file_inputs(self, as_of: datetime | None) -> PipelineInputs:
        return load_default_inputs(
            as_of=as_of,
            processed_dir=self.settings.processed_dir,
            models_dir=self.settings.models_dir,
            experiments_dir=self.settings.experiments_dir,
        )

    def suppressed_keys(self, db: Session, now: datetime) -> dict[str, str]:
        """Keys whose latest warning was dismissed and is still inside its suppression window,
        mapped to the severity at dismissal (a higher severity later counts as an escalation)."""
        latest: dict[str, IntelWarning] = {}
        for w in db.scalars(select(IntelWarning).order_by(IntelWarning.id)):
            latest[w.key] = w
        out: dict[str, str] = {}
        for key, w in latest.items():
            until = aware(w.suppressed_until)
            if w.status == "dismissed" and until is not None and until > now:
                out[key] = w.dismissed_severity or w.severity
        return out

    def gather_inputs(self, db: Session, as_of: datetime | None, now: datetime) -> PipelineInputs:
        inputs = self.inputs_factory(as_of)
        inputs.as_of = as_of
        inputs.now = now
        ratings = db.execute(select(Rating.user_id, Rating.movie_id, Rating.rating, Rating.updated_at)).all()
        inputs.app_ratings = pd.DataFrame(ratings, columns=["user_id", "movie_id", "rating", "timestamp"])
        feedback = db.execute(
            select(RecommendationFeedback.feedback, RecommendationFeedback.created_at)
        ).all()
        inputs.app_feedback = pd.DataFrame(feedback, columns=["feedback", "timestamp"])
        served = db.execute(select(Recommendation.movie_id, Recommendation.created_at)).all()
        inputs.app_served = pd.DataFrame(served, columns=["movie_id", "timestamp"])
        inputs.suppressed_keys = self.suppressed_keys(db, now)
        metrics.set("intel_warnings", "suppressed_keys", len(inputs.suppressed_keys))
        return inputs

    @staticmethod
    def check_as_of(inputs: PipelineInputs, as_of: datetime | None, now: datetime) -> None:
        if as_of is None:
            return
        if as_of > now:
            raise IntelInputError("as_of is in the future")
        ts = pd.to_numeric(inputs.interactions["timestamp"], errors="coerce").dropna()
        if not len(ts):
            raise IntelInputError("no MovieLens events to replay")
        first, last = float(ts.min()), float(ts.max())
        if not first <= as_of.timestamp() <= last + DAY_S:
            lo = datetime.fromtimestamp(first, UTC).date()
            hi = datetime.fromtimestamp(last, UTC).date()
            raise IntelInputError(f"as_of must be between {lo} and {hi} (the MovieLens data range)")

    # ---- runs -----------------------------------------------------------------------------------
    def run(self, trigger: str, as_of: str | datetime | None = None, user: User | None = None) -> IntelRun:
        """Run the pipeline synchronously and persist it. Raises IntelBusyError / IntelInputError only;
        every other failure is recorded on the returned (failed) run."""
        requested = parse_as_of(as_of)
        if not self._lock.acquire(blocking=False):
            raise IntelBusyError("an intelligence run is already in progress")
        metrics.set("intel_pipeline", "in_progress", True)
        try:
            with SessionLocal() as db:
                return self._run_locked(db, trigger, requested, user)
        finally:
            metrics.set("intel_pipeline", "in_progress", False)
            self._lock.release()

    def _run_locked(
        self, db: Session, trigger: str, requested: datetime | None, user: User | None
    ) -> IntelRun:
        now = datetime.now(UTC)
        t0 = time.perf_counter()
        run = IntelRun(
            run_id=str(uuid.uuid4()),
            trigger=trigger,
            status="running",
            requested_as_of=requested,
            started_at=now,
            pipeline_version=PIPELINE_VERSION,
            stage_ms={},
            created_by_id=user.id if user else None,
        )
        inputs: PipelineInputs | None = None
        error: str | None = None
        try:
            inputs = self.gather_inputs(db, requested, now)
            self.check_as_of(inputs, requested, now)
        except IntelInputError:
            raise
        except Exception as exc:
            log.exception("intelligence inputs failed to load")
            error = f"loading inputs failed: {type(exc).__name__}: {exc}"
        load_ms = round(1000 * (time.perf_counter() - t0), 2)
        metrics.inc("intel_pipeline", f"runs_{trigger}")
        db.add(run)
        db.commit()
        result: PipelineResult | None = None
        if inputs is not None and error is None:
            try:
                result = run_pipeline(inputs, self.config)
            except Exception as exc:
                log.exception("intelligence pipeline failed", extra={"extra_fields": {"run_id": run.run_id}})
                error = f"{type(exc).__name__}: {exc}"
        if result is not None:
            t1 = time.perf_counter()
            try:
                data = result.to_dict()
                self._persist_success(db, run, data, now)
                run.stage_ms = {"load_inputs": load_ms, **data["run"]["stage_ms"]}
                run.stage_ms["persist"] = round(1000 * (time.perf_counter() - t1), 2)
                run.duration_ms = round(1000 * (time.perf_counter() - t0), 2)
                run.finished_at = datetime.now(UTC)
                db.commit()
                self._remember(run.run_id, data, result)
            except Exception as exc:
                db.rollback()
                log.exception(
                    "persisting intelligence run failed", extra={"extra_fields": {"run_id": run.run_id}}
                )
                error = f"persisting the result failed: {type(exc).__name__}: {exc}"
        if error is not None:
            run = db.get(IntelRun, run.id) or run
            run.status = "failed"
            run.error = error[:4000]
            run.finished_at = datetime.now(UTC)
            run.duration_ms = round(1000 * (time.perf_counter() - t0), 2)
            run.stage_ms = {"load_inputs": load_ms}
            db.commit()
        self._record_metrics(run)
        log.info(
            "intelligence run %s",
            run.status,
            extra={
                "extra_fields": {
                    "run_id": run.run_id,
                    "trigger": trigger,
                    "status": run.status,
                    "as_of": iso(run.as_of or run.requested_as_of),
                    "duration_ms": run.duration_ms,
                    "counts": (run.summary or {}).get("counts"),
                    "error": run.error,
                }
            },
        )
        return run

    def _record_metrics(self, run: IntelRun) -> None:
        metrics.inc("intel_pipeline", run.status)
        metrics.set("intel_pipeline", "last_status", run.status)
        metrics.set("intel_pipeline", "last_run_id", run.run_id)
        metrics.set("intel_pipeline", "last_duration_ms", run.duration_ms)
        metrics.set("intel_pipeline", "last_finished_at", iso(run.finished_at))
        if run.duration_ms is not None:
            metrics.observe("intel_pipeline_ms", run.status, run.duration_ms)
        if run.status == "succeeded":
            for stage, ms in (run.stage_ms or {}).items():
                if isinstance(ms, int | float):
                    metrics.observe("intel_stage_ms", stage, float(ms))

    def _persist_success(self, db: Session, run: IntelRun, data: dict[str, Any], now: datetime) -> None:
        info = data["run"]
        run.status = "succeeded"
        run.as_of = parse_as_of(info["as_of"])
        run.data_version = _cut(info.get("data_version"), 120)
        run.model_version = _cut(info.get("model_version"), 80)
        run.summary = data.get("summary")
        run.result = data
        as_of = run.as_of or now
        for d in data.get("decisions", []):
            db.add(
                IntelDecision(
                    run_id=run.run_id,
                    decision_id=str(d["id"])[:40],
                    key=str(d["key"])[:64],
                    spec_id=str(d.get("spec_id") or d["key"])[:64],
                    policy_version=str(d.get("policy_version") or "")[:40],
                    question=str(d.get("question") or ""),
                    kind=d["kind"],
                    options=d.get("options") or [],
                    answer=_cut(d.get("answer"), 64),
                    option_scores=d.get("option_scores") or {},
                    confidence=d.get("confidence"),
                    confidence_kind=d["confidence_kind"],
                    state=d.get("state") or {},
                    rationale=d.get("rationale") or [],
                    evidence=d.get("evidence") or [],
                    abstained=bool(d.get("abstained")),
                    fallback_reason=d.get("fallback_reason"),
                    entity_type=_cut(d.get("entity_type"), 32),
                    entity=_cut(d.get("entity"), 200),
                    as_of=as_of,
                    created_at=now,
                )
            )
        self._upsert_warnings(db, run, data.get("warnings", []), now)

    # ---- warning lifecycle ------------------------------------------------------------------------
    @staticmethod
    def _apply_candidate(w: IntelWarning, c: dict[str, Any]) -> None:
        w.title = str(c.get("title") or c["key"])[:300]
        w.description = str(c.get("description") or "")
        w.severity = c["severity"]
        w.confidence = c.get("confidence")
        w.confidence_kind = _cut(c.get("confidence_kind"), 16)
        w.trigger = c.get("trigger") or {}
        w.evidence = c.get("evidence") or []
        w.recommended_action = str(c.get("recommended_action") or "")
        w.source = c.get("source") or {}
        w.entity_type = _cut(c.get("entity_type"), 32)
        w.entity = _cut(c.get("entity"), 200)

    def _upsert_warnings(
        self, db: Session, run: IntelRun, candidates: list[dict[str, Any]], now: datetime
    ) -> None:
        """Lifecycle rules (docs/intelligence.md, section 5):
        - an open warning with the same key is updated (last_seen, occurrences, severity, evidence),
        - a key dismissed within suppress_days stays quiet unless the severity escalated,
        - a resolved (or expired/escalated dismissed) key that fires again opens a new warning with
          reopened_from pointing at the previous one."""
        as_of = iso(run.as_of)
        for c in candidates:
            key = str(c["key"])[:200]
            open_w = db.scalar(
                select(IntelWarning).where(
                    IntelWarning.key == key, IntelWarning.status.in_(WARNING_OPEN_STATUSES)
                )
            )
            if open_w is not None:
                if open_w.severity != c["severity"]:
                    metrics.inc("intel_warnings", "severity_changed")
                self._apply_candidate(open_w, c)
                open_w.occurrences += 1
                open_w.last_seen_at = now
                open_w.last_seen_run_id = run.run_id
                metrics.inc("intel_warnings", "updated")
                continue
            prev = db.scalar(
                select(IntelWarning).where(IntelWarning.key == key).order_by(IntelWarning.id.desc()).limit(1)
            )
            note = f"detected by run {run.run_id[:8]} (as of {as_of})"
            if prev is not None and prev.status == "dismissed":
                at_dismissal = prev.dismissed_severity or prev.severity
                escalated = severity_rank(c["severity"]) > severity_rank(at_dismissal)
                until = aware(prev.suppressed_until)
                if until is not None and until > now and not escalated:
                    metrics.inc("intel_warnings", "suppressed")
                    continue
                note = (
                    f"severity escalated from {at_dismissal} to {c['severity']} after warning #{prev.id} "
                    "was dismissed"
                    if escalated
                    else f"fired again after the suppression of dismissed warning #{prev.id} expired"
                )
            elif prev is not None and prev.status == "resolved":
                note = f"reopened: warning #{prev.id} was resolved and fired again (run {run.run_id[:8]})"
            w = IntelWarning(
                key=key,
                status="new",
                detected_at=now,
                last_seen_at=now,
                occurrences=1,
                first_seen_run_id=run.run_id,
                last_seen_run_id=run.run_id,
                reopened_from=prev.id if prev is not None else None,
            )
            self._apply_candidate(w, c)
            db.add(w)
            db.flush()
            db.add(
                IntelWarningEvent(
                    warning_id=w.id,
                    from_status=None,
                    to_status="new",
                    note=note,
                    actor="system",
                    run_id=run.run_id,
                    at=now,
                )
            )
            metrics.inc("intel_warnings", "reopened" if prev is not None else "created")

    def update_warning(
        self, db: Session, w: IntelWarning, status: str, note: str | None, user: User
    ) -> IntelWarning:
        if status not in TRANSITIONS.get(w.status, ()):
            allowed = ", ".join(TRANSITIONS.get(w.status, ())) or "none (terminal)"
            raise InvalidTransitionError(f"cannot move a {w.status} warning to {status}; allowed: {allowed}")
        now = datetime.now(UTC)
        db.add(
            IntelWarningEvent(
                warning_id=w.id,
                from_status=w.status,
                to_status=status,
                note=note,
                actor=user.email,
                actor_id=user.id,
                at=now,
            )
        )
        w.status = status
        if status == "dismissed":
            w.dismissed_severity = w.severity
            w.suppressed_until = now + timedelta(days=self.settings.intel_suppress_days)
        if status in ("resolved", "dismissed"):
            w.closed_at = now
        db.commit()
        db.refresh(w)
        metrics.inc("intel_warnings", f"to_{status}")
        return w

    # ---- reading runs -------------------------------------------------------------------------------
    @staticmethod
    def latest_run(db: Session, status: str | None = "succeeded") -> IntelRun | None:
        q = select(IntelRun)
        if status:
            q = q.where(IntelRun.status == status)
        return db.scalar(q.order_by(IntelRun.started_at.desc(), IntelRun.id.desc()).limit(1))

    def result_for(self, db: Session, run: IntelRun) -> dict[str, Any]:
        with self._results_lock:
            hit = self._results.get(run.run_id)
            if hit is not None:
                self._results.move_to_end(run.run_id)
                return hit
        data = db.scalar(select(IntelRun.result).where(IntelRun.id == run.id)) or {}
        self._remember(run.run_id, data)
        return data

    def _remember(self, run_id: str, data: dict[str, Any], obj: PipelineResult | None = None) -> None:
        with self._results_lock:
            self._results[run_id] = data
            self._results.move_to_end(run_id)
            while len(self._results) > RESULT_CACHE_SIZE:
                self._results.popitem(last=False)
            if obj is not None:
                self._latest_obj = (run_id, obj)

    # ---- scenarios ----------------------------------------------------------------------------------
    def scenario(self, db: Session, spec: dict[str, Any]) -> dict[str, Any]:
        """run_scenario on the latest run (its fitted models when still in memory) or, with
        spec.as_of, on freshly prepared inputs at that date. ValueError -> 422 in the router."""
        now = datetime.now(UTC)
        as_of = parse_as_of(spec.pop("as_of", None))
        latest = self.latest_run(db)
        if as_of is None and latest is not None:
            obj = self._latest_obj
            if obj is not None and obj[0] == latest.run_id:
                return run_scenario(obj[1], spec, self.config)
            as_of = aware(latest.as_of)
        inputs = self.gather_inputs(db, as_of, now)
        self.check_as_of(inputs, as_of, now)
        return run_scenario(inputs, spec, self.config)

    # ---- startup refresh ------------------------------------------------------------------------------
    def start_background_refresh(self) -> threading.Thread | None:
        if not self.settings.intel_run_on_startup:
            return None
        t = threading.Thread(target=self._startup_refresh, name="intel-startup", daemon=True)
        t.start()
        return t

    def _startup_refresh(self) -> None:
        try:
            with SessionLocal() as db:
                latest = self.latest_run(db)
                started = aware(latest.started_at) if latest else None
            max_age = timedelta(hours=self.settings.intel_min_interval_hours)
            if started is not None and datetime.now(UTC) - started < max_age:
                log.info("intelligence run is fresh; startup refresh skipped")
                return
            self.run("startup")
        except IntelBusyError:
            log.info("intelligence run already in progress; startup refresh skipped")
        except Exception:
            # never let a background refresh take the API down
            log.exception("startup intelligence refresh failed")

    # ---- observability ----------------------------------------------------------------------------------
    def metrics_snapshot(self, db: Session) -> dict[str, Any]:
        """DB-derived gauges for GET /admin/metrics (the counters come from the registry)."""
        latest = self.latest_run(db)
        decisions: dict[str, dict[str, int]] = {}
        freshness: dict[str, Any] = {}
        if latest is not None:
            rows = db.execute(
                select(IntelDecision.key, IntelDecision.answer, IntelDecision.abstained, func.count())
                .where(IntelDecision.run_id == latest.run_id)
                .group_by(IntelDecision.key, IntelDecision.answer, IntelDecision.abstained)
            ).all()
            for key, answer, abstained, n in rows:
                label = "abstained" if abstained else str(answer)
                decisions.setdefault(key, {})[label] = decisions.get(key, {}).get(label, 0) + int(n)
            for s in (self.result_for(db, latest).get("data") or {}).get("sources", []):
                freshness[s["source"]] = {
                    "age_days": s.get("age_days"),
                    "lag_days": s.get("lag_days"),
                    "fresh": s.get("fresh"),
                    "rows": s.get("rows"),
                }
        open_by = self.open_warning_counts(db)
        return {
            "intel_latest_run": {
                "run_id": latest.run_id if latest else None,
                "as_of": iso(latest.as_of) if latest else None,
                "finished_at": iso(latest.finished_at) if latest else None,
                "duration_ms": latest.duration_ms if latest else None,
            },
            "intel_decisions_latest_run": decisions,
            "intel_warnings_open": open_by,
            "data_freshness": freshness,
        }

    @staticmethod
    def open_warning_counts(db: Session) -> dict[str, Any]:
        by = dict.fromkeys(("critical", "high", "medium", "low"), 0)
        for sev, n in db.execute(
            select(IntelWarning.severity, func.count())
            .where(IntelWarning.status.in_(WARNING_OPEN_STATUSES))
            .group_by(IntelWarning.severity)
        ):
            by[sev] = int(n)
        return {"total": sum(by.values()), "by_severity": by}
