"""ORM models. Portable types only (JSON, not JSONB), so the schema runs on PostgreSQL and SQLite."""

from __future__ import annotations

from datetime import UTC, datetime
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

from jev_api.db import Base


def utcnow() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # bumped on every taste-relevant event; part of the recommendation cache key
    profile_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    recommendation_prefs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    genre_preferences: Mapped[list[UserGenrePreference]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)


class Movie(TimestampMixin, Base):
    __tablename__ = "movies"

    # MovieLens movieId, so ids line up with model artifacts
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    runtime_min: Mapped[int | None] = mapped_column(Integer)
    directors: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    cast: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    countries: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    imdb_id: Mapped[str | None] = mapped_column(String(16))
    tmdb_id: Mapped[int | None] = mapped_column(Integer)
    n_ratings: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    mean_rating: Mapped[float | None] = mapped_column(Float)
    # lower-cased title + directors + cast, for simple portable search
    search_text: Mapped[str] = mapped_column(Text, default="", nullable=False)

    genres: Mapped[list[Genre]] = relationship(
        secondary="movie_genres", lazy="selectin", order_by="Genre.name"
    )


class MovieGenre(Base):
    __tablename__ = "movie_genres"

    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    genre_id: Mapped[int] = mapped_column(
        ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True, index=True
    )


class UserGenrePreference(Base):
    __tablename__ = "user_genre_preferences"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    genre_id: Mapped[int] = mapped_column(ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="genre_preferences")
    genre: Mapped[Genre] = relationship(lazy="joined")


class Rating(TimestampMixin, Base):
    __tablename__ = "ratings"
    __table_args__ = (
        UniqueConstraint("user_id", "movie_id", name="uq_rating_user_movie"),
        CheckConstraint("rating >= 0.5 AND rating <= 5.0", name="ck_rating_range"),
        Index("ix_ratings_user_updated", "user_id", "updated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id: Mapped[int] = mapped_column(
        ForeignKey("movies.id", ondelete="CASCADE"), nullable=False, index=True
    )
    rating: Mapped[float] = mapped_column(Float, nullable=False)

    movie: Mapped[Movie] = relationship(lazy="joined")


class WatchHistory(Base):
    __tablename__ = "watch_history"
    __table_args__ = (
        Index("ix_watch_user_time", "user_id", "watched_at"),
        Index("ix_watch_time", "watched_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    watched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    movie: Mapped[Movie] = relationship(lazy="joined")


class Favorite(Base):
    __tablename__ = "favorites"
    __table_args__ = (UniqueConstraint("user_id", "movie_id", name="uq_favorite_user_movie"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    source: Mapped[str] = mapped_column(String(16), default="user", nullable=False)  # user | onboarding
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    movie: Mapped[Movie] = relationship(lazy="joined")


class Recommendation(Base):
    """Every recommendation served, with score, reason and contributing signals."""

    __tablename__ = "recommendations"
    __table_args__ = (
        Index("ix_rec_user_created", "user_id", "created_at"),
        Index("ix_rec_request", "request_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    request_id: Mapped[str] = mapped_column(String(36), nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    context: Mapped[str] = mapped_column(String(32), default="feed", nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(String(300), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(32), nullable=False)
    signals: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    movie: Mapped[Movie] = relationship(lazy="joined")


class RecommendationFeedback(Base):
    __tablename__ = "recommendation_feedback"
    __table_args__ = (
        CheckConstraint("feedback IN ('like','dislike','not_interested','clicked')", name="ck_feedback_kind"),
        Index("ix_feedback_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    recommendation_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendations.id", ondelete="SET NULL"), index=True
    )
    feedback: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(120), nullable=False)
    training_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    training_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    artifact_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    experiment_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(120), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    training_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    split: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    metrics: Mapped[list[EvaluationMetric]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", lazy="selectin"
    )
    model_version: Mapped[ModelVersion | None] = relationship(lazy="joined")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    __table_args__ = (
        UniqueConstraint("experiment_id", "model_name", "protocol", "metric", "k", name="uq_metric"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(32), nullable=False)
    protocol: Mapped[str] = mapped_column(String(16), nullable=False)  # test | cold_start
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    k: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 0 = metric without a cut-off
    value: Mapped[float] = mapped_column(Float, nullable=False)

    experiment: Mapped[Experiment] = relationship(back_populates="metrics")


# --- intelligence layer (docs/intelligence.md, section 5) -------------------------------------------
# Enum values live here once; the CHECK constraints and the API validation both use them.
INTEL_RUN_TRIGGERS = ("startup", "manual", "script", "schedule")
INTEL_RUN_STATUSES = ("running", "succeeded", "failed")
INTEL_SEVERITIES = ("low", "medium", "high", "critical")
WARNING_STATUSES = ("new", "acknowledged", "investigating", "resolved", "dismissed")
WARNING_OPEN_STATUSES = ("new", "acknowledged", "investigating")
DECISION_KINDS = ("boolean", "choice", "score")
CONFIDENCE_KINDS = ("probability", "margin", "rule")
FEEDBACK_VERDICTS = {
    "decision": ("correct", "incorrect"),
    "warning": ("useful", "not_useful", "false_positive"),
    "action": ("useful", "not_useful"),
    "prediction": ("correct", "incorrect"),
}


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({','.join(repr(v) for v in values)})"


_OPEN_WARNING = _in("status", WARNING_OPEN_STATUSES)
_VERDICT_MATCHES_TARGET = " OR ".join(
    f"(target_type = '{t}' AND {_in('verdict', v)})" for t, v in FEEDBACK_VERDICTS.items()
)


class IntelRun(Base):
    """One pipeline run: parameters, status, timings, versions, summary and the full result JSON."""

    __tablename__ = "intel_runs"
    __table_args__ = (
        # quoted: TRIGGER is a keyword in SQL
        CheckConstraint(_in('"trigger"', INTEL_RUN_TRIGGERS), name="ck_intel_run_trigger"),
        CheckConstraint(_in("status", INTEL_RUN_STATUSES), name="ck_intel_run_status"),
        Index("ix_intel_runs_status_started", "status", "started_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    trigger: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="running", nullable=False)
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


class IntelWarning(TimestampMixin, Base):
    """An early warning with a lifecycle. At most one open warning per key (partial unique index)."""

    __tablename__ = "intel_warnings"
    __table_args__ = (
        CheckConstraint(_in("severity", INTEL_SEVERITIES), name="ck_intel_warning_severity"),
        CheckConstraint(_in("status", WARNING_STATUSES), name="ck_intel_warning_status"),
        CheckConstraint(
            f"dismissed_severity IS NULL OR {_in('dismissed_severity', INTEL_SEVERITIES)}",
            name="ck_intel_warning_dismissed_severity",
        ),
        CheckConstraint("occurrences >= 1", name="ck_intel_warning_occurrences"),
        Index(
            "uq_intel_warnings_open_key",
            "key",
            unique=True,
            sqlite_where=text(_OPEN_WARNING),
            postgresql_where=text(_OPEN_WARNING),
        ),
        Index("ix_intel_warnings_key", "key"),
        Index("ix_intel_warnings_status_severity", "status", "severity"),
        Index("ix_intel_warnings_last_seen", "last_seen_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(200), nullable=False)
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

    events: Mapped[list[IntelWarningEvent]] = relationship(
        back_populates="warning", cascade="all, delete-orphan", order_by="IntelWarningEvent.id"
    )


class IntelWarningEvent(Base):
    """Audit trail: every status change of a warning (who, when, from -> to, note)."""

    __tablename__ = "intel_warning_events"
    __table_args__ = (
        CheckConstraint(
            f"from_status IS NULL OR {_in('from_status', WARNING_STATUSES)}", name="ck_intel_event_from"
        ),
        CheckConstraint(_in("to_status", WARNING_STATUSES), name="ck_intel_event_to"),
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
        CheckConstraint(_in("kind", DECISION_KINDS), name="ck_intel_decision_kind"),
        CheckConstraint(_in("confidence_kind", CONFIDENCE_KINDS), name="ck_intel_decision_confidence_kind"),
        Index("ix_intel_decisions_decision_id", "decision_id"),
        Index("ix_intel_decisions_key_created", "key", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("intel_runs.run_id", ondelete="CASCADE"), nullable=False)
    decision_id: Mapped[str] = mapped_column(String(40), nullable=False)  # contract id "dec-..."
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
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class IntelScenario(Base):
    """A saved what-if analysis (input spec + output)."""

    __tablename__ = "intel_scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
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
        CheckConstraint(_in("target_type", tuple(FEEDBACK_VERDICTS)), name="ck_intel_feedback_target"),
        CheckConstraint(_VERDICT_MATCHES_TARGET, name="ck_intel_feedback_verdict"),
        Index("ix_intel_feedback_target", "target_type", "target_id"),
        Index("ix_intel_feedback_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_type: Mapped[str] = mapped_column(String(16), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[str | None] = mapped_column(String(1000))
    outcome: Mapped[str | None] = mapped_column(String(500))
    actor: Mapped[str] = mapped_column(String(320), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    run_id: Mapped[str | None] = mapped_column(ForeignKey("intel_runs.run_id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
