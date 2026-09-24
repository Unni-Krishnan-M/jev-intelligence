"""Intelligence layer tables (docs/intelligence.md, sections 5 and 9.3): runs, warnings and their events,
decisions, scenarios, operator feedback, the normalised run objects with their evidence, and offline
evaluation runs. Owned by WS4."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from jev_api.models.base import Base, TimestampMixin, domain_column, in_, nullable_in, utcnow
from jev_api.models.enums import (
    CONFIDENCE_KINDS,
    DECISION_KINDS,
    EVIDENCE_OWNERS,
    FEEDBACK_VERDICTS,
    INTEL_RUN_MODES,
    INTEL_RUN_STATUSES,
    INTEL_RUN_TRIGGERS,
    INTEL_SEVERITIES,
    TREND_DIRECTIONS,
    WARNING_OPEN_STATUSES,
    WARNING_STATUSES,
)

_OPEN_WARNING = in_("status", WARNING_OPEN_STATUSES)
_VERDICT_MATCHES_TARGET = " OR ".join(
    f"(target_type = '{t}' AND {in_('verdict', v)})" for t, v in FEEDBACK_VERDICTS.items()
)


class IntelRun(Base):
    """One pipeline run: parameters, status, timings, versions, summary and the full result JSON."""

    __tablename__ = "intel_runs"
    __table_args__ = (
        # quoted: TRIGGER is a keyword in SQL
        CheckConstraint(in_('"trigger"', INTEL_RUN_TRIGGERS), name="ck_intel_run_trigger"),
        CheckConstraint(in_("status", INTEL_RUN_STATUSES), name="ck_intel_run_status"),
        CheckConstraint(in_("mode", INTEL_RUN_MODES), name="ck_intel_run_mode"),
        Index("ix_intel_runs_status_started", "status", "started_at"),
        Index("ix_intel_runs_domain_started", "domain", "status", "started_at"),
        Index("ix_intel_runs_domain_mode_started", "domain", "mode", "status", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    domain: Mapped[str] = domain_column()
    trigger: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="running", nullable=False)
    # Phase 2 (migration 0006): "replay" when the run was asked for an explicit as_of (run_mode()).
    # Replays must never touch live state; "latest" reads mean the latest live run (WS4).
    mode: Mapped[str] = mapped_column(String(8), default="live", server_default="live", nullable=False)
    requested_as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # resolved by the pipeline
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None] = mapped_column(Float)
    pipeline_version: Mapped[str] = mapped_column(String(40), nullable=False)
    data_version: Mapped[str | None] = mapped_column(String(120))
    model_version: Mapped[str | None] = mapped_column(String(80))
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    stage_ms: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    # PipelineResult.to_dict() (~0.5 MB): deferred, so listing runs never loads it
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, deferred=True)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    # WS1 (migration 0007): the events the run read (max event id / ingested_at, as_of, knowledge time)
    event_watermark: Mapped[dict[str, Any] | None] = mapped_column(JSON)


def run_mode(requested_as_of: datetime | None) -> str:
    """The mode a new run is recorded with: an explicit as_of makes it a replay (P2.0 acceptance)."""
    return "live" if requested_as_of is None else "replay"


class IntelWarning(TimestampMixin, Base):
    """An early warning with a lifecycle. At most one open warning per (domain, key) (partial unique
    index; v1.2 made it per domain)."""

    __tablename__ = "intel_warnings"
    __table_args__ = (
        CheckConstraint(in_("severity", INTEL_SEVERITIES), name="ck_intel_warning_severity"),
        CheckConstraint(in_("status", WARNING_STATUSES), name="ck_intel_warning_status"),
        CheckConstraint(
            f"dismissed_severity IS NULL OR {in_('dismissed_severity', INTEL_SEVERITIES)}",
            name="ck_intel_warning_dismissed_severity",
        ),
        CheckConstraint("occurrences >= 1", name="ck_intel_warning_occurrences"),
        Index(
            "uq_intel_warnings_open_key",
            "domain",
            "key",
            unique=True,
            sqlite_where=text(_OPEN_WARNING),
            postgresql_where=text(_OPEN_WARNING),
        ),
        Index("ix_intel_warnings_key", "key"),
        Index("ix_intel_warnings_status_severity", "status", "severity"),
        Index("ix_intel_warnings_last_seen", "last_seen_at"),
        Index("ix_intel_warnings_domain_status", "domain", "status", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = domain_column()
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_kind: Mapped[str | None] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="new", nullable=False)
    trigger: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    recommended_action: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity: Mapped[str | None] = mapped_column(String(200))
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    occurrences: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    first_seen_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("intel_runs.run_id", ondelete="SET NULL")
    )
    last_seen_run_id: Mapped[str | None] = mapped_column(ForeignKey("intel_runs.run_id", ondelete="SET NULL"))
    reopened_from: Mapped[int | None] = mapped_column(ForeignKey("intel_warnings.id", ondelete="SET NULL"))
    # set on dismissal: the key stays suppressed until suppressed_until unless severity escalates
    dismissed_severity: Mapped[str | None] = mapped_column(String(16))
    suppressed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    # v1.2: the early_warning_level decision the warning is downstream of, and its answer
    decision_id: Mapped[str | None] = mapped_column(String(40))
    early_warning_level: Mapped[str | None] = mapped_column(String(16))

    events: Mapped[list[IntelWarningEvent]] = relationship(
        back_populates="warning", cascade="all, delete-orphan", order_by="IntelWarningEvent.id"
    )


class IntelWarningEvent(Base):
    """Audit trail: every status change of a warning (who, when, from -> to, note)."""

    __tablename__ = "intel_warning_events"
    __table_args__ = (
        CheckConstraint(
            f"from_status IS NULL OR {in_('from_status', WARNING_STATUSES)}", name="ck_intel_event_from"
        ),
        CheckConstraint(in_("to_status", WARNING_STATUSES), name="ck_intel_event_to"),
        Index("ix_intel_events_warning_at", "warning_id", "at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    warning_id: Mapped[int] = mapped_column(
        ForeignKey("intel_warnings.id", ondelete="CASCADE"), nullable=False
    )
    from_status: Mapped[str | None] = mapped_column(String(16))
    to_status: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String(320), nullable=False)  # "system" or the user's email
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("intel_runs.run_id", ondelete="SET NULL"))
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    warning: Mapped[IntelWarning] = relationship(back_populates="events")


class IntelDecision(Base):
    """Every decision of every run: flattened columns plus the policy's state/evidence as JSON."""

    __tablename__ = "intel_decisions"
    __table_args__ = (
        UniqueConstraint("run_id", "decision_id", name="uq_intel_decision_run"),
        CheckConstraint(in_("kind", DECISION_KINDS), name="ck_intel_decision_kind"),
        CheckConstraint(in_("confidence_kind", CONFIDENCE_KINDS), name="ck_intel_decision_confidence_kind"),
        Index("ix_intel_decisions_decision_id", "decision_id"),
        Index("ix_intel_decisions_key_created", "key", "created_at"),
        Index("ix_intel_decisions_batch", "batch_id"),
        Index("ix_intel_decisions_domain_key", "domain", "key", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("intel_runs.run_id", ondelete="CASCADE"), nullable=False)
    decision_id: Mapped[str] = mapped_column(String(40), nullable=False)  # contract id "dec-..."
    domain: Mapped[str] = domain_column()
    key: Mapped[str] = mapped_column(String(64), nullable=False)
    spec_id: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(40), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    options: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    answer: Mapped[str | None] = mapped_column(String(64))
    option_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    rationale: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    fallback_reason: Mapped[str | None] = mapped_column(Text)
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity: Mapped[str | None] = mapped_column(String(200))
    # v1.1 (section 9.1): multi-question batches and numeric "score" answers
    batch_id: Mapped[str | None] = mapped_column(String(64))
    answer_value: Mapped[float | None] = mapped_column(Float)  # score decisions; `answer` keeps str()
    answer_interval: Mapped[list[float] | None] = mapped_column(JSON)
    scale: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class IntelScenario(Base):
    """A saved what-if analysis (input spec + output)."""

    __tablename__ = "intel_scenarios"
    __table_args__ = (Index("ix_intel_scenarios_domain", "domain", "created_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    domain: Mapped[str] = domain_column()
    series_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    input: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    output: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False, index=True
    )


class IntelFeedback(Base):
    """Operator verdicts on decisions, warnings, actions and predictions."""

    __tablename__ = "intel_feedback"
    __table_args__ = (
        CheckConstraint(in_("target_type", tuple(FEEDBACK_VERDICTS)), name="ck_intel_feedback_target"),
        CheckConstraint(_VERDICT_MATCHES_TARGET, name="ck_intel_feedback_verdict"),
        Index("ix_intel_feedback_target", "target_type", "target_id"),
        Index("ix_intel_feedback_created", "created_at"),
        Index("ix_intel_feedback_domain", "domain", "target_type"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    domain: Mapped[str] = domain_column()
    target_type: Mapped[str] = mapped_column(String(16), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[str | None] = mapped_column(String(1000))
    outcome: Mapped[str | None] = mapped_column(String(500))
    actor: Mapped[str] = mapped_column(String(320), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("intel_runs.run_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


# --- v1.1: normalised run objects and evaluation runs (docs/intelligence.md, section 9.3) -------------
# Each run object keeps indexed key columns plus its full contract JSON in `payload`. `run_id` here is
# the integer intel_runs.id (CASCADE); the API reports the run's uuid. Kinds are an ML-owned, open
# vocabulary and carry no CHECK, so a new kind can never fail a run; severities/levels/directions do.


class IntelSignalRow(Base):
    __tablename__ = "intel_signals"
    __table_args__ = (
        UniqueConstraint("run_id", "signal_id", name="uq_intel_signal_run"),
        CheckConstraint(nullable_in("direction", TREND_DIRECTIONS), name="ck_intel_signal_direction"),
        Index("ix_intel_signals_run_kind", "run_id", "kind"),
        Index("ix_intel_signals_signal_id", "signal_id"),
        Index("ix_intel_signals_dedup_key", "dedup_key", "run_id"),
        Index("ix_intel_signals_entity", "entity_type", "entity"),
        Index("ix_intel_signals_strength", "strength"),
        Index("ix_intel_signals_observed", "observed_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("intel_runs.id", ondelete="CASCADE"), nullable=False)
    signal_id: Mapped[str] = mapped_column(String(40), nullable=False)
    dedup_key: Mapped[str] = mapped_column(String(200), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity: Mapped[str | None] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    strength: Mapped[float | None] = mapped_column(Float)
    direction: Mapped[str | None] = mapped_column(String(8))
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class IntelTrendRow(Base):
    __tablename__ = "intel_trends"
    __table_args__ = (
        UniqueConstraint("run_id", "trend_id", name="uq_intel_trend_run"),
        CheckConstraint(nullable_in("direction", TREND_DIRECTIONS), name="ck_intel_trend_direction"),
        Index("ix_intel_trends_run_direction", "run_id", "direction"),
        Index("ix_intel_trends_trend_id", "trend_id"),
        Index("ix_intel_trends_series", "series_id", "run_id"),
        Index("ix_intel_trends_entity", "entity_type", "entity"),
        Index("ix_intel_trends_p_value", "p_value"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("intel_runs.id", ondelete="CASCADE"), nullable=False)
    trend_id: Mapped[str] = mapped_column(String(40), nullable=False)
    series_id: Mapped[str] = mapped_column(String(200), nullable=False)
    metric: Mapped[str | None] = mapped_column(String(32))
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity: Mapped[str | None] = mapped_column(String(200))
    direction: Mapped[str | None] = mapped_column(String(8))
    slope: Mapped[float | None] = mapped_column(Float)
    p_value: Mapped[float | None] = mapped_column(Float)
    q_value: Mapped[float | None] = mapped_column(Float)
    evidence_strength: Mapped[float | None] = mapped_column(Float)
    has_change_point: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class IntelAnomalyRow(Base):
    __tablename__ = "intel_anomalies"
    __table_args__ = (
        UniqueConstraint("run_id", "anomaly_id", name="uq_intel_anomaly_run"),
        CheckConstraint(nullable_in("severity", INTEL_SEVERITIES), name="ck_intel_anomaly_severity"),
        Index("ix_intel_anomalies_run_kind", "run_id", "kind"),
        Index("ix_intel_anomalies_anomaly_id", "anomaly_id"),
        Index("ix_intel_anomalies_dedup_key", "dedup_key", "run_id"),
        Index("ix_intel_anomalies_series", "series_id"),
        Index("ix_intel_anomalies_entity", "entity_type", "entity"),
        Index("ix_intel_anomalies_severity", "severity"),
        Index("ix_intel_anomalies_score", "score"),
        Index("ix_intel_anomalies_detected", "detected_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("intel_runs.id", ondelete="CASCADE"), nullable=False)
    anomaly_id: Mapped[str] = mapped_column(String(40), nullable=False)
    dedup_key: Mapped[str | None] = mapped_column(String(200))
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity: Mapped[str | None] = mapped_column(String(200))
    series_id: Mapped[str | None] = mapped_column(String(200))
    severity: Mapped[str | None] = mapped_column(String(16))
    score: Mapped[float | None] = mapped_column(Float)
    value: Mapped[float | None] = mapped_column(Float)
    detected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    suppressed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class IntelForecastRow(Base):
    __tablename__ = "intel_forecasts"
    __table_args__ = (
        UniqueConstraint("run_id", "forecast_id", name="uq_intel_forecast_run"),
        Index("ix_intel_forecasts_forecast_id", "forecast_id"),
        Index("ix_intel_forecasts_series", "series_id", "run_id"),
        Index("ix_intel_forecasts_entity", "entity_type", "entity"),
        Index("ix_intel_forecasts_mase", "mase"),
        Index("ix_intel_forecasts_issued", "issued_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("intel_runs.id", ondelete="CASCADE"), nullable=False)
    forecast_id: Mapped[str] = mapped_column(String(40), nullable=False)
    series_id: Mapped[str] = mapped_column(String(200), nullable=False)
    metric: Mapped[str | None] = mapped_column(String(32))
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity: Mapped[str | None] = mapped_column(String(200))
    model: Mapped[str | None] = mapped_column(String(40))
    horizon_months: Mapped[int | None] = mapped_column(Integer)
    mase: Mapped[float | None] = mapped_column(Float)
    coverage80: Mapped[float | None] = mapped_column(Float)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class IntelRiskRow(Base):
    __tablename__ = "intel_risks"
    __table_args__ = (
        UniqueConstraint("run_id", "risk_id", name="uq_intel_risk_run"),
        CheckConstraint(nullable_in("level", INTEL_SEVERITIES), name="ck_intel_risk_level"),
        Index("ix_intel_risks_run_kind", "run_id", "kind"),
        Index("ix_intel_risks_risk_id", "risk_id"),
        Index("ix_intel_risks_key", "key", "run_id"),
        Index("ix_intel_risks_entity", "entity_type", "entity"),
        Index("ix_intel_risks_level", "level"),
        Index("ix_intel_risks_score", "score"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("intel_runs.id", ondelete="CASCADE"), nullable=False)
    risk_id: Mapped[str] = mapped_column(String(40), nullable=False)
    key: Mapped[str] = mapped_column(String(200), nullable=False)  # risk:<kind>:<entity>, the warning key
    kind: Mapped[str] = mapped_column(String(40), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity: Mapped[str | None] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    level: Mapped[str | None] = mapped_column(String(16))
    score: Mapped[float | None] = mapped_column(Float)
    likelihood: Mapped[float | None] = mapped_column(Float)
    impact: Mapped[float | None] = mapped_column(Float)
    exposure: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class IntelEvidenceRow(Base):
    """Every Evidence item of a run, polymorphic over its owner (owner_id = the owner's contract id;
    warnings, which carry no id in the result, use their dedup key)."""

    __tablename__ = "intel_evidence"
    __table_args__ = (
        CheckConstraint(in_("owner_type", EVIDENCE_OWNERS), name="ck_intel_evidence_owner"),
        Index("ix_intel_evidence_run_owner", "run_id", "owner_type"),
        Index("ix_intel_evidence_owner", "owner_type", "owner_id"),
        Index("ix_intel_evidence_kind", "kind"),
        Index("ix_intel_evidence_ref", "ref"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("intel_runs.id", ondelete="CASCADE"), nullable=False)
    owner_type: Mapped[str] = mapped_column(String(16), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(200), nullable=False)
    owner_title: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # order within the owner
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    label: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    value_num: Mapped[float | None] = mapped_column(Float)
    value_text: Mapped[str | None] = mapped_column(Text)
    detail: Mapped[str | None] = mapped_column(Text)
    ref: Mapped[str | None] = mapped_column(String(200))


class IntelEvaluationRun(Base):
    """One offline evaluation of the intelligence layer, synced from experiments/intel-eval-*/report.json."""

    __tablename__ = "intel_evaluation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_dir: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True)
    pipeline_version: Mapped[str | None] = mapped_column(String(40))
    data_version: Mapped[str | None] = mapped_column(String(120))
    as_of: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    headline: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    report_sha1: Mapped[str] = mapped_column(String(40), nullable=False)  # re-synced when the file changes
    report: Mapped[dict[str, Any]] = mapped_column(JSON, deferred=True, nullable=False)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
