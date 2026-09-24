"""Retraining and model governance: snapshot -> candidate -> gate -> promotion / rollback.

docs/RETRAINING_AND_MODEL_GOVERNANCE.md. One job at a time, across processes: a PostgreSQL advisory
lock (``pg_try_advisory_lock``), or on SQLite a lease row in ``governance_locks``. The API runs jobs in
a worker thread; ``scripts/retrain.py`` runs them inline and hosts the scheduler loop.

The ML work is ``jev_ml.governance`` (no database there): this module reads app feedback into a
DataFrame, persists snapshots, jobs, lifecycle state and gate results, writes the audit log and swaps
the serving engine through the EngineHolder (load first, never swap to a broken model).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from jev_api.config import Settings
from jev_api.metrics import metrics
from jev_api.models import (
    Favorite,
    IntelDecision,
    IntelRun,
    ModelVersion,
    Rating,
    RecommendationFeedback,
    User,
    WatchHistory,
)
from jev_api.models.governance import DatasetSnapshot, GovernanceLock, ModelGovernance, TrainingJob
from jev_api.services import audit
from jev_api.services.ml import EngineHolder
from jev_api.services.sync import sync_model_versions
from jev_ml.governance import (
    GateConfig,
    build_snapshot,
    evaluate_candidate,
    promotion_blockers,
    train_candidate,
)
from jev_ml.governance.retrain import load_lineage
from jev_ml.governance.snapshot import APP_EVENT_COLUMNS, SEMANTICS_VERSION, load_snapshot_manifest
from jev_ml.paths import CONFIGS_DIR
from jev_ml.registry import active_version, load_manifest, read_registry, rollback_target, set_state

log = logging.getLogger(__name__)

LOCK_NAME = "jev-retrain"
PG_LOCK_KEY = 0x4A45_5652_5452  # "JEVRTR": the advisory lock id of the retrain job
# served-recommendation caches (every key also carries the engine version, so this only frees memory)
CACHE_PREFIXES = ("rec:", "sim:", "trend:", "meintel:", "strat:")
GATE_FILE = "gate.json"
HEADLINE_METRICS = ("ndcg@10", "recall@10", "precision@10", "hit_rate@10", "coverage@10")


class GovernanceError(Exception):
    """Base: ``status`` is the HTTP status the router maps it to."""

    status = 409

    def __init__(self, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.detail = detail if detail is not None else message


class BusyError(GovernanceError):
    status = 409


class NotFoundError(GovernanceError):
    status = 404


class BlockedError(GovernanceError):
    """Promotion refused: the gate has not passed for the current incumbent."""

    status = 409


class GovernanceHTTPError(HTTPException):
    """An HTTP error whose body keeps ``detail`` a string (like every other error) and carries the
    promotion blockers in their own field: ``{"detail": "...", "blockers": [...], "request_id"}``."""

    def __init__(self, status_code: int, detail: str, blockers: list[str] | None = None) -> None:
        super().__init__(status_code, detail)
        self.extra: dict[str, Any] = {"blockers": blockers} if blockers is not None else {}


def http_error(exc: GovernanceError) -> GovernanceHTTPError:
    blockers = [str(b) for b in exc.detail] if isinstance(exc.detail, list) else None
    return GovernanceHTTPError(exc.status, str(exc), blockers)


def utcnow() -> datetime:
    return datetime.now(UTC)


def aware(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes (stored as UTC)."""
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _epoch(dt: datetime | None) -> int | None:
    a = aware(dt)
    return None if a is None else int(a.timestamp())


def gate_config(settings: Settings, quick: bool) -> GateConfig:
    cfg = GateConfig(
        ndcg_margin=settings.governance_gate_ndcg_margin,
        recall_margin=settings.governance_gate_recall_margin,
        cold_start=settings.governance_gate_cold_start,
        cold_start_margin=settings.governance_gate_cold_start_margin,
        coverage_max_relative_drop=settings.governance_gate_coverage_max_drop,
        ece_margin=settings.governance_gate_ece_margin,
        auc_margin=settings.governance_gate_auc_margin,
        latency_p95_ms=settings.governance_gate_latency_p95_ms,
        latency_requests=settings.governance_gate_latency_requests,
        bootstrap_b=settings.governance_gate_bootstrap_b,
        permutations=settings.governance_gate_bootstrap_b,
        alpha=settings.governance_gate_alpha,
        max_users=settings.governance_gate_max_users,
    )
    return cfg.quick() if quick else cfg


def resolve_config(settings: Settings, name: str | None) -> Path | None:
    """A config name from the API (a file inside configs/) or the configured default."""
    if name is None:
        return settings.governance_config_path
    path = (CONFIGS_DIR / name).resolve()
    if path.parent != CONFIGS_DIR.resolve() or not path.is_file():
        raise NotFoundError(f"no config {name!r} in configs/")
    return path


# --- app feedback ----------
def read_app_events(db: Session) -> pd.DataFrame:
    """Every app signal as rows of APP_EVENT_COLUMNS (see jev_ml.governance.snapshot for semantics)."""
    rows: list[tuple[int, int, str, float | None, int | None]] = []
    for uid, mid, value, at in db.execute(
        select(Rating.user_id, Rating.movie_id, Rating.rating, Rating.updated_at)
    ):
        rows.append((uid, mid, "rating", float(value), _epoch(at)))
    for uid, mid, source, at in db.execute(
        select(Favorite.user_id, Favorite.movie_id, Favorite.source, Favorite.created_at)
    ):
        rows.append((uid, mid, "onboarding" if source == "onboarding" else "favorite", None, _epoch(at)))
    for uid, mid, at in db.execute(
        select(WatchHistory.user_id, WatchHistory.movie_id, WatchHistory.watched_at)
    ):
        rows.append((uid, mid, "watch", None, _epoch(at)))
    for uid, mid, kind, at in db.execute(
        select(
            RecommendationFeedback.user_id,
            RecommendationFeedback.movie_id,
            RecommendationFeedback.feedback,
            RecommendationFeedback.created_at,
        )
    ):
        rows.append((uid, mid, str(kind), None, _epoch(at)))
    df = pd.DataFrame(rows, columns=list(APP_EVENT_COLUMNS))
    return df.dropna(subset=["ts"]).astype({"ts": "int64"}) if len(df) else df


def db_watermark(db: Session) -> dict[str, Any]:
    """The newest row of every source (and of WS1's append-only events log when it exists)."""
    from sqlalchemy import func

    def newest(col: Any) -> str | None:
        v = aware(db.scalar(select(func.max(col))))
        return v.isoformat() if v else None

    out: dict[str, Any] = {
        "ratings": newest(Rating.updated_at),
        "favorites": newest(Favorite.created_at),
        "watch_history": newest(WatchHistory.watched_at),
        "recommendation_feedback": newest(RecommendationFeedback.created_at),
    }
    try:
        if inspect(db.get_bind()).has_table("events"):
            max_seq, max_at, n = db.execute(
                text("SELECT MAX(id), MAX(ingested_at), COUNT(*) FROM events")
            ).one()
            out["events"] = {
                "max_seq": max_seq,
                "max_ingested_at": str(max_at) if max_at else None,
                "rows": n,
            }
    except Exception as exc:  # the events log is WS1's; a schema change there must not block snapshots
        db.rollback()
        out["events"] = {"error": type(exc).__name__}
    return out


# --- the one-job-at-a-time lock ----------
class RetrainLock:
    """PostgreSQL: a session advisory lock on a dedicated AUTOCOMMIT connection (released by the server
    if the process dies). Other databases: a lease row that expires after the job timeout."""

    def __init__(self, engine: Engine, ttl_minutes: float, name: str = LOCK_NAME) -> None:
        self.engine = engine
        self.ttl = timedelta(minutes=ttl_minutes)
        self.name = name
        self.holder = f"{socket.gethostname()[:60]}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        self._conn: Connection | None = None
        self.held = False

    @property
    def is_postgres(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    def acquire(self) -> bool:
        if self.held:
            return True
        if self.is_postgres:
            conn = self.engine.connect().execution_options(isolation_level="AUTOCOMMIT")
            got = bool(conn.execute(text("SELECT pg_try_advisory_lock(:k)"), {"k": PG_LOCK_KEY}).scalar())
            if not got:
                conn.close()
                return False
            self._conn = conn
        else:
            with Session(self.engine) as s:
                now = utcnow()
                s.execute(
                    delete(GovernanceLock).where(
                        GovernanceLock.name == self.name, GovernanceLock.expires_at < now
                    )
                )
                s.add(
                    GovernanceLock(
                        name=self.name, holder=self.holder, acquired_at=now, expires_at=now + self.ttl
                    )
                )
                try:
                    s.commit()
                except IntegrityError:
                    s.rollback()
                    return False
        self.held = True
        return True

    def release(self) -> None:
        if not self.held:
            return
        try:
            if self._conn is not None:
                self._conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": PG_LOCK_KEY})
                self._conn.close()
                self._conn = None
            else:
                with Session(self.engine) as s:
                    s.execute(
                        delete(GovernanceLock).where(
                            GovernanceLock.name == self.name, GovernanceLock.holder == self.holder
                        )
                    )
                    s.commit()
        finally:
            self.held = False


# --- the latest retrain decision ----------
def latest_retrain_decision(db: Session) -> IntelDecision | None:
    """The ``retrain_model`` decision of the latest successful LIVE movie run (never a replay)."""
    run = db.scalar(
        select(IntelRun)
        .where(IntelRun.domain == "movie", IntelRun.mode == "live", IntelRun.status == "succeeded")
        .order_by(IntelRun.started_at.desc(), IntelRun.id.desc())
        .limit(1)
    )
    if run is None:
        return None
    return db.scalar(
        select(IntelDecision)
        .where(IntelDecision.run_id == run.run_id, IntelDecision.key == "retrain_model")
        .limit(1)
    )


def decision_says_retrain(dec: IntelDecision | None) -> bool:
    return dec is not None and dec.answer == "yes" and not dec.abstained


# --- snapshots ----------
def export_snapshot(
    db: Session, settings: Settings, cutoff: datetime | None = None, actor: User | str | None = None
) -> tuple[DatasetSnapshot, bool]:
    """Build (or reuse) the snapshot of the processed data + app feedback; register it. Returns the row
    and whether it is new. Audited as dataset.snapshot when new."""
    manifest = build_snapshot(
        settings.processed_dir,
        read_app_events(db),
        settings.governance_snapshots_dir,
        cutoff=cutoff,
        user_offset=settings.governance_app_user_offset,
        watermark=db_watermark(db),
        sources={"database": {"dialect": db.get_bind().dialect.name}},
    )
    row = db.scalar(select(DatasetSnapshot).where(DatasetSnapshot.snapshot_id == manifest["snapshot_id"]))
    new = row is None
    if row is None:
        name = actor.email if isinstance(actor, User) else (actor or audit.SYSTEM)
        row = DatasetSnapshot(
            snapshot_id=manifest["snapshot_id"],
            content_hash=manifest["content_hash"],
            base_dataset_version=manifest.get("base_dataset_version"),
            cutoff=datetime.fromisoformat(manifest["cutoff"]) if manifest.get("cutoff") else None,
            row_counts=manifest["row_counts"],
            sources=manifest["sources"],
            watermark=manifest.get("watermark") or {},
            semantics_version=SEMANTICS_VERSION,
            path=str(settings.governance_snapshots_dir / manifest["snapshot_id"]),
            created_by=str(name)[:320],
        )
        db.add(row)
        db.flush()
        audit.record(
            db,
            "dataset.snapshot",
            actor,
            "dataset_snapshot",
            row.snapshot_id,
            {
                "content_hash": row.content_hash,
                "row_counts": row.row_counts,
                "cutoff": manifest.get("cutoff"),
            },
        )
    db.commit()
    metrics.inc("governance", "snapshots_new" if new else "snapshots_reused")
    return row, new


def snapshot_dir_for(settings: Settings, version: str) -> Path:
    """The frozen data a version is gated on: its lineage snapshot, else (models trained before
    governance) the processed data when it is the dataset the version was trained on."""
    lineage = load_lineage(version, settings.models_dir) or {}
    sid = lineage.get("snapshot_id")
    if sid:
        d = settings.governance_snapshots_dir / str(sid)
        if (d / "manifest.json").exists():
            return d
        raise NotFoundError(
            f"snapshot {sid} of {version} is missing from {settings.governance_snapshots_dir}"
        )
    manifest = load_manifest(version, settings.models_dir)
    meta_path = settings.processed_dir / "dataset_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if meta.get("dataset_version") != manifest.get("dataset_version"):
        raise NotFoundError(
            f"{version} has no snapshot lineage and the processed data is a different dataset"
        )
    return settings.processed_dir


# --- governance rows ----------
def governance_row(db: Session, version: str) -> ModelGovernance:
    row = db.scalar(select(ModelGovernance).where(ModelGovernance.version == version))
    if row is None:
        sync_model_versions(db)
        row = db.scalar(select(ModelGovernance).where(ModelGovernance.version == version))
    if row is None:
        raise NotFoundError(f"unknown model version {version}")
    return row


def apply_lineage(row: ModelGovernance, lineage: dict[str, Any]) -> None:
    row.lineage = lineage
    row.snapshot_id = lineage.get("snapshot_id")
    row.job_id = lineage.get("job_id")
    row.decision_id = lineage.get("decision_id")
    row.config_hash = lineage.get("config_hash")
    row.git_commit = str(lineage["git_commit"])[:40] if lineage.get("git_commit") else None
    row.seed = lineage.get("seed")


def record_gate(settings: Settings, row: ModelGovernance, gate: dict[str, Any]) -> None:
    row.gate = gate
    row.gate_passed = bool(gate.get("passed"))
    row.gate_reasons = list(gate.get("reasons") or [])
    row.gated_against = gate.get("incumbent")
    row.gated_at = utcnow()
    path = settings.models_dir / row.version / GATE_FILE
    try:
        path.write_text(json.dumps(gate, indent=2, sort_keys=True))
        os.chmod(path, 0o644)
    except OSError:
        log.warning("could not write %s", path)


def gate_summary(gate: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, g in ((gate or {}).get("gates") or {}).items():
        numbers = {
            k: v
            for k, v in g.items()
            if k not in ("status", "reason") and (isinstance(v, int | float | str | bool) or v is None)
        }
        out[name] = {"status": g.get("status", "skipped"), "reason": g.get("reason"), "numbers": numbers}
    return out


def governed_model(
    db: Session, row: ModelGovernance, active: str | None, mv: ModelVersion | None
) -> dict[str, Any]:
    blockers = promotion_blockers(row.version, row.state, row.gate, active)
    # a fresh database mirrors lineage.json / gate.json from models/ (services.sync): the job id then
    # names a job of another database, so the entry is labelled as coming from the artifact files
    has_job = row.job_id is not None and (
        db.scalar(select(TrainingJob.id).where(TrainingJob.job_id == row.job_id)) is not None
    )
    metrics_ = {k: float(v) for k, v in ((mv.metrics if mv else {}) or {}).items() if k in HEADLINE_METRICS}
    return {
        "version": row.version,
        "state": row.state,
        "is_active": row.version == active,
        "source": "job" if has_job else "artifact",
        "model_version_id": mv.id if mv else None,
        "trained_at": aware(mv.trained_at) if mv else None,
        "dataset_version": mv.dataset_version if mv else None,
        "snapshot_id": row.snapshot_id,
        "job_id": row.job_id,
        "decision_id": row.decision_id,
        "metrics": metrics_,
        "gate_passed": row.gate_passed,
        "gate_reasons": row.gate_reasons or [],
        "gated_against": row.gated_against,
        "gated_at": aware(row.gated_at),
        "gates": gate_summary(row.gate),
        "promotable": not blockers,
        "blockers": blockers,
        "promoted_at": aware(row.promoted_at),
        "promoted_by": row.promoted_by,
        "forced": row.forced,
        "force_reason": row.force_reason,
    }


@dataclass
class _Actor:
    user_id: int | None
    name: str

    @classmethod
    def of(cls, actor: User | str | None) -> _Actor:
        if isinstance(actor, User):
            return cls(actor.id, actor.email)
        return cls(None, actor or audit.SYSTEM)

    def resolve(self, db: Session) -> User | str:
        if self.user_id is not None:
            user = db.get(User, self.user_id)
            if user is not None:
                return user
        return self.name


class GovernanceService:
    """Jobs (one at a time), promotion, rollback and the scheduler tick."""

    def __init__(
        self,
        settings: Settings,
        engines: EngineHolder,
        cache: Any = None,
        session_factory: sessionmaker[Session] | None = None,
        db_engine: Engine | None = None,
    ) -> None:
        from jev_api import db as dbmod

        self.settings = settings
        self.engines = engines
        self.cache = cache
        self.session_factory = session_factory or dbmod.SessionLocal
        self.db_engine = db_engine or dbmod.engine
        self._thread: threading.Thread | None = None
        self._local = threading.Lock()  # one submission at a time in this process
        self._swap = threading.Lock()  # promotions and rollbacks never interleave

    # --- jobs ----------
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(
        self,
        db: Session,
        *,
        kind: str = "retrain",
        trigger: str = "manual",
        actor: User | str | None = None,
        quick: bool | None = None,
        config: str | None = None,
        version: str | None = None,
        decision_id: str | None = None,
        background: bool = True,
    ) -> TrainingJob:
        """Queue a job and run it (in a worker thread, or inline). Raises BusyError when one is running
        here or in another process."""
        quick = self.settings.governance_quick_default if quick is None else quick
        config_path = resolve_config(self.settings, config) if kind == "retrain" else None
        if kind == "evaluate":
            if version is None:
                raise NotFoundError("evaluate needs a model version")
            governance_row(db, version)
        with self._local:
            if self.busy():
                raise BusyError("a training job is already running in this process")
            lock = RetrainLock(self.db_engine, self.settings.governance_job_timeout_minutes)
            if not lock.acquire():
                raise BusyError("a training job is already running (retrain lock held)")
            try:
                a = _Actor.of(actor)
                job = TrainingJob(
                    job_id=str(uuid.uuid4()),
                    kind=kind,
                    trigger=trigger,
                    status="queued",
                    quick=quick,
                    config_path=str(config_path) if config_path else None,
                    decision_id=decision_id,
                    model_version=version,
                    requested_by=a.name[:320],
                    steps=[],
                )
                db.add(job)
                db.commit()
            except Exception:
                lock.release()
                raise
            if background:
                self._thread = threading.Thread(
                    target=self._run, args=(job.job_id, lock, a), name=f"governance-{kind}", daemon=True
                )
                self._thread.start()
        if not background:
            self._run(job.job_id, lock, a)
            db.expire_all()
        return job

    def wait_idle(self, timeout: float = 600.0) -> bool:
        t = self._thread
        if t is not None:
            t.join(timeout)
        return not self.busy()

    def _run(self, job_id: str, lock: RetrainLock, actor: _Actor) -> None:
        t0 = time.perf_counter()
        try:
            with self.session_factory() as db:
                job = db.scalar(select(TrainingJob).where(TrainingJob.job_id == job_id))
                assert job is not None
                job.status, job.started_at = "running", utcnow()
                db.commit()
                try:
                    if job.kind == "retrain":
                        self._retrain(db, job, actor)
                    else:
                        self._evaluate_job(db, job, actor)
                    job.status = "succeeded"
                    metrics.inc("governance", f"jobs_{job.kind}_succeeded")
                except Exception as exc:
                    db.rollback()
                    log.exception("governance job %s failed", job_id)
                    job = db.scalar(select(TrainingJob).where(TrainingJob.job_id == job_id))
                    assert job is not None
                    job.status = "failed"
                    job.error = f"{type(exc).__name__}: {exc}"[:4000]
                    job.steps = [
                        *job.steps,
                        {
                            "step": "error",
                            "at": utcnow().isoformat(),
                            "trace": traceback.format_exc()[-1500:],
                        },
                    ]
                    metrics.inc("governance", f"jobs_{job.kind}_failed")
                job.finished_at = utcnow()
                job.duration_ms = round((time.perf_counter() - t0) * 1000, 1)
                db.commit()
        finally:
            lock.release()

    def _step(self, db: Session, job: TrainingJob, step: str, **info: Any) -> None:
        job.steps = [*job.steps, {"step": step, "at": utcnow().isoformat(), **info}]
        db.commit()

    def _retrain(self, db: Session, job: TrainingJob, actor: _Actor) -> None:
        s = self.settings
        who = actor.resolve(db)
        snap, new = export_snapshot(db, s, actor=who)
        job.snapshot_id = snap.snapshot_id
        self._step(
            db, job, "snapshot", snapshot_id=snap.snapshot_id, reused=not new, row_counts=snap.row_counts
        )
        incumbent = active_version(s.models_dir)
        job.incumbent_version = incumbent
        snap_dir = Path(snap.path)
        t = time.perf_counter()
        lineage = train_candidate(
            snap_dir,
            Path(job.config_path) if job.config_path else None,
            quick=job.quick,
            models_dir=s.models_dir,
            experiments_dir=s.experiments_dir,
            lineage_extra={
                "job_id": job.job_id,
                "trigger": job.trigger,
                "decision_id": job.decision_id,
                "requested_by": job.requested_by,
            },
        )
        version = lineage["version"]
        job.model_version = version
        self._step(db, job, "train", version=version, seconds=round(time.perf_counter() - t, 1))
        sync_model_versions(db)
        row = governance_row(db, version)
        apply_lineage(row, lineage)
        row.state = "candidate"
        audit.record(
            db,
            "model.retrain",
            who,
            "training_job",
            job.job_id,
            {
                "version": version,
                "trigger": job.trigger,
                "decision_id": job.decision_id,
                "snapshot_id": snap.snapshot_id,
                "quick": job.quick,
            },
        )
        audit.record(
            db,
            "model.register",
            who,
            "model_version",
            version,
            {
                "state": "candidate",
                "snapshot_id": snap.snapshot_id,
                "config_hash": lineage.get("config_hash"),
            },
        )
        db.commit()
        self._maybe_calibrate(db, job, version, incumbent, snap_dir)
        self._gate(db, job, version, incumbent, snap_dir, who)

    def _maybe_calibrate(
        self, db: Session, job: TrainingJob, version: str, incumbent: str | None, snap_dir: Path
    ) -> None:
        s = self.settings
        if not s.governance_calibrate or incumbent is None:
            return
        if not (s.models_dir / incumbent / "calibration.json").exists():
            return
        from jev_ml.calibration import calibrate_model

        t = time.perf_counter()
        try:
            cal = calibrate_model(version, s.models_dir, processed_dir=snap_dir)
            self._step(
                db,
                job,
                "calibrate",
                seconds=round(time.perf_counter() - t, 1),
                ece=(cal.get("headline") or {}).get("ece"),
                auc=(cal.get("headline") or {}).get("auc"),
            )
        except Exception as exc:  # the calibration gate then fails: the candidate has no calibration
            log.exception("calibrating %s failed", version)
            self._step(db, job, "calibrate", error=f"{type(exc).__name__}: {exc}"[:500])

    def _gate(
        self,
        db: Session,
        job: TrainingJob,
        version: str,
        incumbent: str | None,
        snap_dir: Path,
        who: User | str,
    ) -> dict[str, Any]:
        s = self.settings
        gate = evaluate_candidate(version, incumbent, snap_dir, gate_config(s, job.quick), s.models_dir)
        row = governance_row(db, version)
        record_gate(s, row, gate)
        job.gate_passed = bool(gate["passed"])
        self._step(
            db,
            job,
            "gate",
            passed=gate["passed"],
            incumbent=incumbent,
            reasons=gate["reasons"],
            seconds=gate.get("seconds"),
            ndcg_diff=(gate["gates"].get("ndcg@10") or {}).get("diff"),
            ndcg_ci_lo=(gate["gates"].get("ndcg@10") or {}).get("ci_lo"),
        )
        if not gate["passed"] and row.state == "candidate":
            set_state(version, "rejected", s.models_dir, reason="; ".join(gate["reasons"])[:500])
            row.state = "rejected"
            audit.record(
                db,
                "model.reject",
                who,
                "model_version",
                version,
                {
                    "incumbent": incumbent,
                    "reasons": gate["reasons"],
                    "job_id": job.job_id,
                },
            )
            metrics.inc("governance", "gate_rejected")
        elif gate["passed"]:
            metrics.inc("governance", "gate_passed")
        db.commit()
        if (
            gate["passed"]
            and s.governance_auto_promote
            and job.kind == "retrain"
            and row.state == "candidate"
        ):
            self.promote(db, version, audit.SYSTEM, auto=True)
            job.promoted = True
            self._step(db, job, "promote", version=version, auto=True)
        return gate

    def _evaluate_job(self, db: Session, job: TrainingJob, actor: _Actor) -> None:
        s = self.settings
        version = job.model_version
        assert version is not None
        who = actor.resolve(db)
        snap_dir = snapshot_dir_for(s, version)
        job.snapshot_id = (load_lineage(version, s.models_dir) or {}).get("snapshot_id")
        incumbent = active_version(s.models_dir)
        job.incumbent_version = incumbent
        self._step(db, job, "evaluate", version=version, incumbent=incumbent, data=str(snap_dir))
        self._gate(db, job, version, incumbent, snap_dir, who)

    # --- promotion and rollback ----------
    def _clear_caches(self) -> None:
        if self.cache is None:
            return
        for prefix in CACHE_PREFIXES:
            try:
                self.cache.delete_prefix(prefix)
            except Exception:
                log.exception("cache clear failed for %s", prefix)

    def _mirror_active(self, db: Session, active: str, previous: str | None) -> None:
        for mv in db.scalars(select(ModelVersion)).all():
            mv.is_active = mv.version == active
        if previous is not None and previous != active:
            prev = db.scalar(select(ModelGovernance).where(ModelGovernance.version == previous))
            if prev is not None:
                prev.state = "retired"
                prev.retired_at = utcnow()

    def promote(
        self,
        db: Session,
        version: str,
        actor: User | str | None,
        force: bool = False,
        reason: str | None = None,
        auto: bool = False,
    ) -> dict[str, Any]:
        """Make ``version`` serve. Without ``force`` its gate must have passed against the current
        active model. Every attempt is audited as model.promote (``detail.passed``)."""
        s = self.settings
        with self._swap:
            row = governance_row(db, version)
            current = active_version(s.models_dir)
            state = read_registry(s.models_dir)["states"].get(version, row.state)
            if version == current:
                raise BlockedError(
                    f"{version} is already the active model", [f"{version} is already the active model"]
                )
            blockers = promotion_blockers(version, state, row.gate, current)
            base = {
                "version": version,
                "previous": current,
                "forced": force,
                "reason": reason,
                "auto": auto,
                "gate_outcome": None if row.gate_passed is None else ("pass" if row.gate_passed else "fail"),
                "gated_against": row.gated_against,
            }
            if blockers and not force:
                audit.record(
                    db,
                    "model.promote",
                    actor,
                    "model_version",
                    version,
                    {**base, "promoted": False, "blockers": blockers},
                )
                db.commit()
                metrics.inc("governance", "promote_blocked")
                raise BlockedError("promotion blocked by the gate", blockers)
            name = actor.email if isinstance(actor, User) else (actor or audit.SYSTEM)
            try:
                self.engines.activate(version, action="promote", actor=name)  # load first
            except Exception as exc:
                audit.record(
                    db,
                    "model.promote",
                    actor,
                    "model_version",
                    version,
                    {**base, "promoted": False, "blockers": [f"artifact failed to load: {exc}"]},
                )
                db.commit()
                raise BlockedError(
                    "model artifacts failed to load", [f"artifact failed to load: {exc}"]
                ) from exc
            row.state = "active"
            row.promoted_at = utcnow()
            row.promoted_by = str(name)[:320]
            row.forced = bool(force and blockers)
            row.force_reason = reason[:500] if reason else None
            row.retired_at = None
            self._mirror_active(db, version, current)
            if row.job_id:
                job = db.scalar(select(TrainingJob).where(TrainingJob.job_id == row.job_id))
                if job is not None:
                    job.promoted = True
            audit.record(
                db,
                "model.promote",
                actor,
                "model_version",
                version,
                {
                    **base,
                    "promoted": True,
                    "gate_ok": not blockers,
                    "forced": bool(force and blockers),
                    "blockers": blockers,
                },
            )
            db.commit()
            self._clear_caches()
            metrics.inc("governance", "promotions_forced" if force and blockers else "promotions")
            return {
                "version": version,
                "previous": current,
                "action": "promote",
                "forced": bool(force and blockers),
                "gate_passed": row.gate_passed,
            }

    def rollback(self, db: Session, actor: User | str | None, reason: str | None = None) -> dict[str, Any]:
        """Restore the version that served before the active one (it was vetted by serving)."""
        s = self.settings
        with self._swap:
            current = active_version(s.models_dir)
            target = rollback_target(s.models_dir)
            if current is None or target is None:
                raise BlockedError("nothing to roll back to", ["no previous active version is recorded"])
            name = actor.email if isinstance(actor, User) else (actor or audit.SYSTEM)
            try:
                self.engines.activate(target, action="rollback", actor=name)
            except Exception as exc:
                raise BlockedError("the previous model failed to load", [str(exc)]) from exc
            sync_model_versions(db)
            row = governance_row(db, target)
            row.state = "active"
            row.retired_at = None
            self._mirror_active(db, target, current)
            audit.record(
                db,
                "model.rollback",
                actor,
                "model_version",
                target,
                {"from": current, "to": target, "reason": reason},
            )
            db.commit()
            self._clear_caches()
            metrics.inc("governance", "rollbacks")
            return {
                "version": target,
                "previous": current,
                "action": "rollback",
                "forced": False,
                "gate_passed": row.gate_passed,
            }

    # --- scheduler ----------
    def scheduler_tick(
        self, db: Session, now: datetime | None = None, background: bool = False
    ) -> dict[str, Any]:
        """One scheduler decision. Starts at most one job; returns what it did and why."""
        s = self.settings
        now = now or utcnow()
        if self.busy():
            return {"action": "skipped", "reason": "a job is running in this process"}
        last = db.scalar(
            select(TrainingJob)
            .where(TrainingJob.trigger.in_(("schedule", "decision")), TrainingJob.kind == "retrain")
            .order_by(TrainingJob.created_at.desc(), TrainingJob.id.desc())
            .limit(1)
        )
        spacing = timedelta(minutes=s.governance_schedule_interval_minutes)
        last_at = aware(last.created_at) if last is not None else None
        if last_at is not None and now - last_at < spacing:
            return {
                "action": "skipped",
                "reason": f"last automatic job at {last_at.isoformat()} (< interval)",
            }
        dec = latest_retrain_decision(db)
        yes = decision_says_retrain(dec)
        decision_id = dec.decision_id if dec is not None and yes else None
        if s.governance_schedule_require_decision:
            if not yes:
                return {
                    "action": "skipped",
                    "reason": "the latest live retrain_model decision is not yes",
                    "decision_id": dec.decision_id if dec is not None else None,
                }
            used = db.scalar(select(TrainingJob.id).where(TrainingJob.decision_id == decision_id).limit(1))
            if used is not None:
                return {
                    "action": "skipped",
                    "reason": "this decision was already acted on",
                    "decision_id": decision_id,
                }
            trigger = "decision"
        else:
            trigger = "decision" if yes else "schedule"
        try:
            job = self.submit(
                db,
                kind="retrain",
                trigger=trigger,
                actor=audit.SYSTEM,
                decision_id=decision_id,
                background=background,
            )
        except BusyError as exc:
            return {"action": "skipped", "reason": str(exc)}
        return {"action": "started", "job_id": job.job_id, "trigger": trigger, "decision_id": decision_id}

    def run_scheduler(self, stop: threading.Event, once: bool = False) -> None:
        """The loop (scripts/retrain.py schedule). Disabled unless JEV_GOVERNANCE_SCHEDULE_ENABLED."""
        if not self.settings.governance_schedule_enabled:
            log.warning("governance scheduler disabled (JEV_GOVERNANCE_SCHEDULE_ENABLED=false)")
            return
        while not stop.is_set():
            try:
                with self.session_factory() as db:
                    res = self.scheduler_tick(db)
                log.info("scheduler tick: %s", res)
            except Exception:
                log.exception("scheduler tick failed")
            if once:
                return
            stop.wait(self.settings.governance_schedule_poll_seconds)


_services_lock = threading.Lock()


def get_service(app: Any) -> GovernanceService:
    """The app's GovernanceService, created on first use (main.py is integrator-frozen)."""
    svc: GovernanceService | None = getattr(app.state, "governance", None)
    if svc is None:
        with _services_lock:
            svc = getattr(app.state, "governance", None)
            if svc is None:
                svc = GovernanceService(app.state.settings, app.state.engines, app.state.cache)
                app.state.governance = svc
    return svc


def snapshot_manifest(row: DatasetSnapshot) -> dict[str, Any] | None:
    try:
        return load_snapshot_manifest(Path(row.path))
    except (OSError, ValueError):
        return None
