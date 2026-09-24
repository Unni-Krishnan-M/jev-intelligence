"""Model governance (WS2, migration 0008): dataset snapshots, training jobs, per-version lifecycle
state with lineage and gate results, and the lease row that serialises retraining on SQLite.

docs/RETRAINING_AND_MODEL_GOVERNANCE.md. The files stay the source of truth for artifacts and for
which version serves (models/registry.json); ``model_governance.state`` mirrors it, and adds what only
the database keeps (gate results, who promoted, the job and decision behind a candidate).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from jev_api.models.base import Base, TimestampMixin, in_, utcnow
from jev_api.models.enums import (
    MODEL_STATES,
    TRAINING_JOB_KINDS,
    TRAINING_JOB_STATUSES,
    TRAINING_JOB_TRIGGERS,
)


class DatasetSnapshot(Base):
    """A versioned training snapshot under data/snapshots/<snapshot_id>/ (content-hashed)."""

    __tablename__ = "dataset_snapshots"
    __table_args__ = (
        UniqueConstraint("snapshot_id", name="uq_dataset_snapshots_snapshot_id"),
        Index("ix_dataset_snapshots_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    base_dataset_version: Mapped[str | None] = mapped_column(String(120))
    cutoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    row_counts: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    sources: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    watermark: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    semantics_version: Mapped[str] = mapped_column(String(40), nullable=False)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    created_by: Mapped[str] = mapped_column(String(320), default="system", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class TrainingJob(Base):
    """One retraining (snapshot -> train -> gate) or evaluation (gate only) job."""

    __tablename__ = "training_jobs"
    __table_args__ = (
        UniqueConstraint("job_id", name="uq_training_jobs_job_id"),
        CheckConstraint(in_("kind", TRAINING_JOB_KINDS), name="ck_training_job_kind"),
        CheckConstraint(in_("status", TRAINING_JOB_STATUSES), name="ck_training_job_status"),
        CheckConstraint(in_('"trigger"', TRAINING_JOB_TRIGGERS), name="ck_training_job_trigger"),
        Index("ix_training_jobs_status_created", "status", "created_at"),
        Index("ix_training_jobs_decision", "decision_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), default="retrain", nullable=False)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="queued", nullable=False)
    quick: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    config_path: Mapped[str | None] = mapped_column(String(500))
    decision_id: Mapped[str | None] = mapped_column(String(40))  # the retrain_model decision behind it
    snapshot_id: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(80))
    incumbent_version: Mapped[str | None] = mapped_column(String(80))
    gate_passed: Mapped[bool | None] = mapped_column(Boolean)
    promoted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    requested_by: Mapped[str] = mapped_column(String(320), default="system", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    steps: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)  # the logs summary
    error: Mapped[str | None] = mapped_column(Text)


class ModelGovernance(TimestampMixin, Base):
    """Lifecycle state, lineage and the latest gate result of one model version."""

    __tablename__ = "model_governance"
    __table_args__ = (
        UniqueConstraint("version", name="uq_model_governance_version"),
        CheckConstraint(in_("state", MODEL_STATES), name="ck_model_governance_state"),
        Index("ix_model_governance_state", "state"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(80), nullable=False)
    state: Mapped[str] = mapped_column(String(16), default="candidate", nullable=False)
    snapshot_id: Mapped[str | None] = mapped_column(String(64))
    job_id: Mapped[str | None] = mapped_column(String(36))
    decision_id: Mapped[str | None] = mapped_column(String(40))
    config_hash: Mapped[str | None] = mapped_column(String(64))
    git_commit: Mapped[str | None] = mapped_column(String(40))
    seed: Mapped[int | None] = mapped_column(Integer)
    lineage: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    gate: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    gate_passed: Mapped[bool | None] = mapped_column(Boolean)
    gate_reasons: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    gated_against: Mapped[str | None] = mapped_column(String(80))
    gated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    promoted_by: Mapped[str | None] = mapped_column(String(320))
    forced: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    force_reason: Mapped[str | None] = mapped_column(String(500))
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class GovernanceLock(Base):
    """A named lease (SQLite and other databases without advisory locks). PostgreSQL uses
    pg_try_advisory_lock instead; the table exists there too but stays empty."""

    __tablename__ = "governance_locks"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    holder: Mapped[str] = mapped_column(String(120), nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
