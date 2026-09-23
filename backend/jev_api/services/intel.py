"""Intelligence layer service: gather inputs, run the pipeline, persist runs/decisions/warnings.

The pipeline itself (ml/jev_ml/intel) is pure. This module owns everything stateful around it:
- inputs: processed MovieLens files and the model manifest (load_default_inputs) plus the app DB
  (ratings, recommendation feedback, served recommendations, dismissed warning keys),
- one run at a time (a process-wide lock; a second trigger gets IntelBusyError -> HTTP 409),
- persistence of the run (running -> succeeded | failed), its decisions and the warning lifecycle
  (docs/intelligence.md, section 5), plus the normalised run objects (signals, trends, anomalies,
  forecasts, risks) and every Evidence item, in the same transaction (section 9.3),
- a small cache of parsed run results, so list endpoints do not re-read ~0.5 MB of JSON per call.

Pipeline failures are recorded on the run row and never propagate to the API.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from jev_api.cache import Cache
from jev_api.config import Settings
from jev_api.db import SessionLocal
from jev_api.metrics import metrics
from jev_api.models import (
    WARNING_OPEN_STATUSES,
    IntelAnomalyRow,
    IntelDecision,
    IntelEvidenceRow,
    IntelForecastRow,
    IntelRiskRow,
    IntelRun,
    IntelSignalRow,
    IntelTrendRow,
    IntelWarning,
    IntelWarningEvent,
    Rating,
    Recommendation,
    User,
)
from jev_api.services import audit
from jev_api.services.feedback import latest_feedback
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
        # OverflowError: a valid timestamp whose UTC conversion leaves year 1..9999
        return aware(datetime.fromisoformat(value.strip()))
    except (ValueError, OverflowError) as exc:
        raise IntelInputError("as_of must be YYYY-MM-DD or an ISO-8601 timestamp") from exc


_ABS_PATH = re.compile(r"(?<![\w.])(?:/[^\s'\"/:]+)+/([^\s'\"/:]+)")


def safe_error(exc: BaseException, context: str | None = None) -> str:
    """The run error shown to operators: exception type plus the first line of its message, with
    absolute paths cut to their file name and without SQLAlchemy's statement / parameter dump.
    The full traceback is logged server-side."""
    lines = str(exc).strip().splitlines()
    msg = _ABS_PATH.sub(r"\1", lines[0] if lines else "")[:500]
    text = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    return f"{context}: {text}" if context else text


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
    # score decisions answer with a number (section 9.1); `answer` is stored as its str()
    numeric = d.kind == "score" and d.answer_value is not None
    return {
        "id": d.decision_id,
        "key": d.key,
        "spec_id": d.spec_id,
        "policy_version": d.policy_version,
        "question": d.question,
        "kind": d.kind,
        "options": d.options or [],
        "answer": d.answer_value if numeric else d.answer,
        "answer_value": d.answer_value,
        "answer_interval": d.answer_interval,
        "scale": d.scale,
        "batch_id": d.batch_id,
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


def _num(value: Any) -> float | None:
    """A finite float, or None (strings, bools, NaN and missing values are not numbers here)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    v = float(value)
    return v if math.isfinite(v) else None


def _when(value: Any) -> datetime | None:
    try:
        return parse_as_of(value) if isinstance(value, str | datetime) else None
    except IntelInputError:
        return None


# --- normalised run objects (section 9.3) ------------------------------------------------------------
def _owner_title(owner_type: str, item: dict[str, Any]) -> str:
    if owner_type == "decision":
        title = item.get("question")
    elif owner_type == "trend":
        title = f"{item.get('series_id')} trend {item.get('direction')}"
    elif owner_type == "anomaly":
        title = f"{item.get('kind')}: {item.get('series_id') or item.get('entity')}"
    elif owner_type == "forecast":
        title = f"forecast {item.get('series_id')}"
    else:
        title = item.get("title")
    return str(title or item.get("id") or item.get("key") or "")[:300]


def evidence_owners(data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every (owner_type, object) of a result that can carry Evidence."""
    groups = (
        ("signal", data.get("signals")),
        ("trend", data.get("trends")),
        ("anomaly", data.get("anomalies")),
        ("forecast", (data.get("predictions") or {}).get("forecasts")),
        ("risk", data.get("risks")),
        ("decision", data.get("decisions")),
        ("warning", data.get("warnings")),
        ("action", data.get("actions")),
    )
    return [(owner, item) for owner, items in groups for item in items or [] if isinstance(item, dict)]


def evidence_rows(run_pk: int, data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for owner, item in evidence_owners(data):
        # warnings carry no id in the result: their dedup key identifies them
        owner_id = str(item.get("key") if owner == "warning" else item.get("id"))[:200]
        title = _owner_title(owner, item)
        for pos, ev in enumerate(item.get("evidence") or []):
            if not isinstance(ev, dict):
                continue
            value = ev.get("value")
            num = _num(value)
            if num is None and value is not None:
                text = json.dumps(value) if isinstance(value, bool | list | dict) else str(value)
            else:
                text = None
            rows.append(
                {
                    "run_id": run_pk,
                    "owner_type": owner,
                    "owner_id": owner_id,
                    "owner_title": title,
                    "position": pos,
                    "kind": str(ev.get("kind") or "")[:16],
                    "label": str(ev.get("label") or "")[:300],
                    "value_num": num,
                    "value_text": text,
                    "detail": None if ev.get("detail") is None else str(ev["detail"]),
                    "ref": _cut(ev.get("ref"), 200),
                }
            )
    return rows


def normalised_rows(run_pk: int, data: dict[str, Any]) -> dict[Any, list[dict[str, Any]]]:
    """Indexed key columns plus the full contract object (`payload`) per run object."""
    signals = [
        {
            "run_id": run_pk,
            "signal_id": str(x["id"])[:40],
            "dedup_key": str(x.get("dedup_key") or x["id"])[:200],
            "kind": str(x.get("kind") or "")[:32],
            "entity_type": _cut(x.get("entity_type"), 32),
            "entity": _cut(x.get("entity"), 200),
            "title": str(x.get("title") or x["id"])[:300],
            "value": _num(x.get("value")),
            "strength": _num(x.get("strength")),
            "direction": _cut(x.get("direction"), 8),
            "observed_at": _when(x.get("observed_at")),
            "payload": x,
        }
        for x in data.get("signals") or []
    ]
    trends = [
        {
            "run_id": run_pk,
            "trend_id": str(x["id"])[:40],
            "series_id": str(x.get("series_id") or "")[:200],
            "metric": _cut(x.get("metric"), 32),
            "entity_type": _cut(x.get("entity_type"), 32),
            "entity": _cut(x.get("entity"), 200),
            "direction": _cut(x.get("direction"), 8),
            "slope": _num(x.get("slope")),
            "p_value": _num(x.get("p_value")),
            "q_value": _num(x.get("q_value")),
            "evidence_strength": _num(x.get("evidence_strength")),
            "has_change_point": bool(x.get("change_point")),
            "payload": x,
        }
        for x in data.get("trends") or []
    ]
    anomalies = [
        {
            "run_id": run_pk,
            "anomaly_id": str(x["id"])[:40],
            "dedup_key": _cut(x.get("dedup_key"), 200),
            "kind": str(x.get("kind") or "")[:32],
            "entity_type": _cut(x.get("entity_type"), 32),
            "entity": _cut(x.get("entity"), 200),
            "series_id": _cut(x.get("series_id"), 200),
            "severity": _cut(x.get("severity"), 16),
            "score": _num(x.get("score")),
            "value": _num(x.get("value")),
            "detected_at": _when(x.get("detected_at")),
            "suppressed": bool(x.get("suppressed")),
            "payload": x,
        }
        for x in data.get("anomalies") or []
    ]
    forecasts = [
        {
            "run_id": run_pk,
            "forecast_id": str(x["id"])[:40],
            "series_id": str(x.get("series_id") or "")[:200],
            "metric": _cut(x.get("metric"), 32),
            "entity_type": _cut(x.get("entity_type"), 32),
            "entity": _cut(x.get("entity"), 200),
            "model": _cut(x.get("model"), 40),
            "horizon_months": x.get("horizon_months") if isinstance(x.get("horizon_months"), int) else None,
            "mase": _num((x.get("backtest") or {}).get("mase")),
            "coverage80": _num((x.get("backtest") or {}).get("coverage80")),
            "issued_at": _when(x.get("issued_at")),
            "payload": x,
        }
        for x in (data.get("predictions") or {}).get("forecasts") or []
    ]
    risks = [
        {
            "run_id": run_pk,
            "risk_id": str(x["id"])[:40],
            "key": f"risk:{x.get('kind')}:{x.get('entity')}"[:200],  # the warning key of this risk
            "kind": str(x.get("kind") or "")[:40],
            "entity_type": _cut(x.get("entity_type"), 32),
            "entity": _cut(x.get("entity"), 200),
            "title": str(x.get("title") or x["id"])[:300],
            "level": _cut(x.get("level"), 16),
            "score": _num(x.get("score")),
            "likelihood": _num(x.get("likelihood")),
            "impact": _num(x.get("impact")),
            "exposure": _num(x.get("exposure")),
            "confidence": _num(x.get("confidence")),
            "payload": x,
        }
        for x in data.get("risks") or []
    ]
    return {
        IntelSignalRow: signals,
        IntelTrendRow: trends,
        IntelAnomalyRow: anomalies,
        IntelForecastRow: forecasts,
        IntelRiskRow: risks,
        IntelEvidenceRow: evidence_rows(run_pk, data),
    }


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
        # one verdict (and one click) per member per movie: repeats cannot fake or bury a spike
        latest = latest_feedback()
        feedback = db.execute(select(latest.c.feedback, latest.c.created_at)).all()
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
            error = safe_error(exc, "loading inputs failed")
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
                error = safe_error(exc)
        if result is not None:
            t1 = time.perf_counter()
            try:
                data = result.to_dict()
                persisted = self._persist_success(db, run, data, now)
                run.stage_ms = {"load_inputs": load_ms, **data["run"]["stage_ms"]}
                run.stage_ms["persist"] = round(1000 * (time.perf_counter() - t1), 2)
                run.duration_ms = round(1000 * (time.perf_counter() - t0), 2)
                run.finished_at = datetime.now(UTC)
                self._audit_run(db, run, user, persisted)
                db.commit()
                self._record_persisted(persisted)
                self._remember(run.run_id, data, result)
            except Exception as exc:
                db.rollback()
                log.exception(
                    "persisting intelligence run failed", extra={"extra_fields": {"run_id": run.run_id}}
                )
                error = safe_error(exc, "persisting the result failed")
        if error is not None:
            run = db.get(IntelRun, run.id) or run
            run.status = "failed"
            run.error = error[:4000]
            run.finished_at = datetime.now(UTC)
            run.duration_ms = round(1000 * (time.perf_counter() - t0), 2)
            run.stage_ms = {"load_inputs": load_ms}
            self._audit_run(db, run, user, None)
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

    @staticmethod
    def _audit_run(db: Session, run: IntelRun, user: User | None, persisted: dict[str, int] | None) -> None:
        audit.record(
            db,
            "intel.run",
            user,
            "intel_run",
            run.run_id,
            {
                "trigger": run.trigger,
                "status": run.status,
                "requested_as_of": iso(run.requested_as_of),
                "as_of": iso(run.as_of),
                "duration_ms": run.duration_ms,
                "counts": (run.summary or {}).get("counts"),
                "persisted": persisted,
                "error": _cut(run.error, 500),
            },
        )

    @staticmethod
    def _record_persisted(persisted: dict[str, int]) -> None:
        for table, n in persisted.items():
            metrics.inc("intel_rows_persisted", table, n)
        metrics.set("intel_pipeline", "last_evidence_rows", persisted.get("intel_evidence", 0))
        metrics.observe("intel_evidence_rows_per_run", "all", persisted.get("intel_evidence", 0))

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

    def _persist_success(
        self, db: Session, run: IntelRun, data: dict[str, Any], now: datetime
    ) -> dict[str, int]:
        """Run row, decisions, normalised objects + evidence and warnings, in one transaction.
        Returns the number of rows written per table."""
        info = data["run"]
        run.status = "succeeded"
        run.as_of = parse_as_of(info["as_of"])
        run.data_version = _cut(info.get("data_version"), 120)
        run.model_version = _cut(info.get("model_version"), 80)
        run.summary = data.get("summary")
        run.result = data
        as_of = run.as_of or now
        for d in data.get("decisions", []):
            answer = d.get("answer")
            answer_value = _num(answer)  # score decisions (section 9.1) answer with a number
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
                    answer=_cut(answer, 64),
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
                    batch_id=_cut(d.get("batch_id"), 64),
                    answer_value=answer_value if d.get("kind") == "score" else None,
                    answer_interval=d.get("answer_interval"),
                    scale=d.get("scale"),
                    as_of=as_of,
                    created_at=now,
                )
            )
        persisted = {"intel_decisions": len(data.get("decisions", []))}
        for model, rows in normalised_rows(run.id, data).items():
            if rows:
                db.execute(insert(model), rows)
            persisted[model.__tablename__] = len(rows)
        self._upsert_warnings(db, run, data.get("warnings", []), now)
        return persisted

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
        audit.record(
            db,
            "warning.transition",
            user,
            "intel_warning",
            w.id,
            {"key": w.key, "from": w.status, "to": status, "note": note, "severity": w.severity},
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
