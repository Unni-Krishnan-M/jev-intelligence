"""Online experiments (WS5, migration 0009, docs/EXPERIMENTATION.md).

Named ``ab_*`` so they never collide with the offline ``Experiment`` table (model_registry.py), which
keeps holding offline evaluation runs.

* ``AbExperiment``: one A/B test on a serving surface, with its lifecycle
  (draft -> running <-> paused -> stopped -> concluded), traffic allocation, hypothesis, primary metric,
  guardrails and the concluded result. At most one running or paused experiment per surface (a partial
  unique index, portable to SQLite and PostgreSQL).
* ``AbVariant``: an arm and its serving-config payload (hybrid overrides, model version, ...).
* ``AbAssignment``: the sticky (experiment, user) -> variant row. One per user per experiment.
* ``AbExposure``: one row per served list, including cache hits, inside or outside an experiment
  (experiment_id NULL). The exposure log the gap matrix asked for (P1 #6).
* ``AbOutcome``: an interaction attributed to an exposure within the experiment's window.
* ``MemberDecision``: every recommendation_strategy decision that was served, so decision ids on
  recommendation and feedback rows resolve after the cache expires (P1 #7).
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
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from jev_api.models.base import Base, TimestampMixin, in_, utcnow
from jev_api.models.enums import (
    AB_ACTIVE_STATUSES,
    AB_EXPERIMENT_STATUSES,
    AB_OUTCOME_KINDS,
    AB_OUTCOME_SOURCES,
    AB_PRIMARY_METRICS,
    AB_SURFACES,
    CONFIDENCE_KINDS,
)

# partial unique index: one running-or-paused experiment per surface
AB_ACTIVE_WHERE = in_("status", AB_ACTIVE_STATUSES)


class AbExperiment(TimestampMixin, Base):
    __tablename__ = "ab_experiments"
    __table_args__ = (
        CheckConstraint(in_("surface", AB_SURFACES), name="ck_ab_experiment_surface"),
        CheckConstraint(in_("status", AB_EXPERIMENT_STATUSES), name="ck_ab_experiment_status"),
        CheckConstraint(in_("primary_metric", AB_PRIMARY_METRICS), name="ck_ab_experiment_primary_metric"),
        Index(
            "uq_ab_experiments_active_surface",
            "surface",
            unique=True,
            sqlite_where=text(AB_ACTIVE_WHERE),
            postgresql_where=text(AB_ACTIVE_WHERE),
        ),
        Index("ix_ab_experiments_status", "status", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    surface: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="draft", nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, default="", nullable=False)
    primary_metric: Mapped[str] = mapped_column(String(32), nullable=False)
    # [{"metric": "negative_rate", "max_increase": 0.01}, {"metric": "latency_p95_ms", "max_ratio": 1.5}]
    guardrails: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    # share of eligible members enrolled, 0..100 (a ramp changes only this)
    traffic_percent: Mapped[float] = mapped_column(Float, default=100.0, nullable=False)
    # hashing salt, fixed at creation: sha256(salt ":" user_id) decides enrolment and the variant
    salt: Mapped[str] = mapped_column(String(64), nullable=False)
    attribution_window_hours: Mapped[float] = mapped_column(Float, default=24.0, nullable=False)
    # alpha, power, mde_relative, min_users_per_variant, data_source ("live" | "offline_replay")
    analysis: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)  # frozen at conclude
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    stopped_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    concluded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))

    variants: Mapped[list[AbVariant]] = relationship(
        back_populates="experiment",
        cascade="all, delete-orphan",
        order_by="AbVariant.position",
        lazy="selectin",
    )


class AbVariant(Base):
    __tablename__ = "ab_variants"
    __table_args__ = (UniqueConstraint("experiment_id", "name", name="uq_ab_variants_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("ab_experiments.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(40), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)  # the hash walks weights in this order
    is_control: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    description: Mapped[str] = mapped_column(String(300), default="", nullable=False)
    # {"hybrid_overrides": {...}, "model_version": "...", "strategy_decision": bool,
    #  "recency_half_life_days": float}
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    experiment: Mapped[AbExperiment] = relationship(back_populates="variants")


class AbAssignment(Base):
    """Sticky: written at a member's first request while enrolled, then reused for the experiment's
    life, whatever later ramps do to the traffic share."""

    __tablename__ = "ab_assignments"
    __table_args__ = (
        UniqueConstraint("experiment_id", "user_id", name="uq_ab_assignments_user"),
        Index("ix_ab_assignments_variant", "experiment_id", "variant"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("ab_experiments.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    variant_id: Mapped[int] = mapped_column(ForeignKey("ab_variants.id", ondelete="CASCADE"), nullable=False)
    variant: Mapped[str] = mapped_column(String(40), nullable=False)
    bucket: Mapped[int] = mapped_column(Integer, nullable=False)  # 0..9999, the enrolment hash
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AbExposure(Base):
    """One served list. request_id is fresh for every response, cache hits included; a cache hit points
    at the request that generated the list (source_request_id), whose recommendation rows it reuses."""

    __tablename__ = "ab_exposures"
    __table_args__ = (
        Index("ix_ab_exposures_experiment", "experiment_id", "variant", "served_at"),
        Index("ix_ab_exposures_user_served", "user_id", "served_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    source_request_id: Mapped[str | None] = mapped_column(String(36))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    surface: Mapped[str] = mapped_column(String(32), nullable=False)
    context: Mapped[str] = mapped_column(String(32), default="feed", nullable=False)
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("ab_experiments.id", ondelete="CASCADE"))
    variant: Mapped[str | None] = mapped_column(String(40))
    cached: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    decision_id: Mapped[str | None] = mapped_column(String(40))
    # [{"movie_id", "rank", "recommendation_id"}] in served order
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    n_items: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    served_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class AbOutcome(Base):
    """An interaction attributed to an exposure. rank is the item's served rank, NULL when the member
    interacted with an item that was not on the list (it still counts toward the ideal DCG)."""

    __tablename__ = "ab_outcomes"
    __table_args__ = (
        UniqueConstraint("exposure_id", "kind", "movie_id", name="uq_ab_outcomes_event"),
        CheckConstraint(in_("kind", AB_OUTCOME_KINDS), name="ck_ab_outcome_kind"),
        CheckConstraint(in_("source", AB_OUTCOME_SOURCES), name="ck_ab_outcome_source"),
        Index("ix_ab_outcomes_experiment", "experiment_id", "variant"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exposure_id: Mapped[int] = mapped_column(
        ForeignKey("ab_exposures.id", ondelete="CASCADE"), nullable=False
    )
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("ab_experiments.id", ondelete="CASCADE"), nullable=False
    )
    variant: Mapped[str] = mapped_column(String(40), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    rank: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16), default="live", nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class MemberDecision(Base):
    """A served recommendation_strategy decision. The decision id is stable for the same member and
    history, so rows are unique per (decision_id, state_hash): a new state under the same id (e.g. new
    feedback) is a new row, and resolving an id returns its newest state."""

    __tablename__ = "member_decisions"
    __table_args__ = (
        UniqueConstraint("decision_id", "state_hash", name="uq_member_decisions_state"),
        CheckConstraint(in_("confidence_kind", CONFIDENCE_KINDS), name="ck_member_decision_confidence_kind"),
        Index("ix_member_decisions_decision", "decision_id"),
        Index("ix_member_decisions_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[str] = mapped_column(String(40), nullable=False)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    spec_id: Mapped[str] = mapped_column(String(64), nullable=False)
    policy_version: Mapped[str | None] = mapped_column(String(64))
    answer: Mapped[str | None] = mapped_column(String(32))
    served_strategy: Mapped[str] = mapped_column(String(24), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    state_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    rationale: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    evidence: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # recommendations | me_intelligence
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
