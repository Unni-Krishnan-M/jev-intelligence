"""Served recommendations (the exposure log), feedback on them, and member feedback on the decision layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from jev_api.models.base import Base, TimestampMixin, in_, nullable_in, utcnow
from jev_api.models.catalog import Movie
from jev_api.models.enums import (
    ME_FEEDBACK_TARGETS,
    ME_FEEDBACK_VERDICTS,
    REC_CONFIDENCE_KINDS,
    REC_FEEDBACK_KINDS,
)


class Recommendation(Base):
    """Every recommendation served, with score, reason and contributing signals."""

    __tablename__ = "recommendations"
    __table_args__ = (
        Index("ix_rec_user_created", "user_id", "created_at"),
        Index("ix_rec_request", "request_id"),
        Index("ix_rec_user_decision", "user_id", "decision_id"),
        Index("ix_rec_experiment", "experiment_id", "variant"),
        CheckConstraint(nullable_in("confidence_kind", REC_CONFIDENCE_KINDS), name="ck_rec_confidence_kind"),
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
    # calibrated P(rating >= 4) from models/<version>/calibration.json; null without a calibration
    confidence: Mapped[float | None] = mapped_column(Float)
    confidence_kind: Mapped[str | None] = mapped_column(String(16))
    # v1.2: the recommendation_strategy decision the list was served under (docs/platform.md, section 4)
    decision_id: Mapped[str | None] = mapped_column(String(40))
    strategy: Mapped[str | None] = mapped_column(String(24))
    # Phase 2 (0009): the online experiment and variant the list was served under (NULL outside one).
    # No foreign key: a lineage column, kept when an experiment is gone
    experiment_id: Mapped[int | None] = mapped_column(Integer)
    variant: Mapped[str | None] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    movie: Mapped[Movie] = relationship(lazy="joined")


# Partial unique indexes (migration 0004): one verdict (like / dislike / not_interested) and one click per
# user per served recommendation, or per user per movie for feedback given outside a recommendation.
# A click is an interaction, not a judgement, so it has its own slot and never replaces a verdict.
_WITH_REC, _NO_REC = "recommendation_id IS NOT NULL", "recommendation_id IS NULL"
_VERDICT, _CLICK = "feedback <> 'clicked'", "feedback = 'clicked'"
FEEDBACK_UNIQUE_INDEXES: tuple[tuple[str, str, str], ...] = (
    ("uq_feedback_verdict_rec", "recommendation_id", f"{_WITH_REC} AND {_VERDICT}"),
    ("uq_feedback_verdict_movie", "movie_id", f"{_NO_REC} AND {_VERDICT}"),
    ("uq_feedback_click_rec", "recommendation_id", f"{_WITH_REC} AND {_CLICK}"),
    ("uq_feedback_click_movie", "movie_id", f"{_NO_REC} AND {_CLICK}"),
)


class RecommendationFeedback(Base):
    """A member's latest verdict on a recommendation (or movie), plus at most one click row.
    POST /recommendations/feedback upserts; it never appends duplicates."""

    __tablename__ = "recommendation_feedback"
    __table_args__ = (
        CheckConstraint(in_("feedback", REC_FEEDBACK_KINDS), name="ck_feedback_kind"),
        Index("ix_feedback_user_created", "user_id", "created_at"),
        *(
            Index(name, "user_id", col, unique=True, sqlite_where=text(where), postgresql_where=text(where))
            for name, col, where in FEEDBACK_UNIQUE_INDEXES
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), nullable=False)
    recommendation_id: Mapped[int | None] = mapped_column(
        ForeignKey("recommendations.id", ondelete="SET NULL"), index=True
    )
    feedback: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class UserIntelFeedback(TimestampMixin, Base):
    """A member's verdict (accepted | rejected) on their recommendation-strategy decision or on a
    recommendation (POST /me/intelligence/feedback). One row per member per target: a repeat is a
    no-op and a new verdict replaces the old one. Kept apart from recommendation_feedback: it judges
    the decision layer, and must not silently exclude movies or move the taste profile."""

    __tablename__ = "user_intel_feedback"
    __table_args__ = (
        UniqueConstraint("user_id", "target_type", "target_id", name="uq_user_intel_feedback_target"),
        CheckConstraint(in_("target_type", ME_FEEDBACK_TARGETS), name="ck_user_intel_feedback_target"),
        CheckConstraint(in_("verdict", ME_FEEDBACK_VERDICTS), name="ck_user_intel_feedback_verdict"),
        Index("ix_user_intel_feedback_user_created", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    target_type: Mapped[str] = mapped_column(String(16), nullable=False)
    target_id: Mapped[str] = mapped_column(String(64), nullable=False)
    verdict: Mapped[str] = mapped_column(String(16), nullable=False)
    note: Mapped[str | None] = mapped_column(String(1000))
    # the strategy decision the target belongs to (the decision itself, or the one a recommendation was
    # served under), and the movie of a recommendation target
    decision_id: Mapped[str | None] = mapped_column(String(40))
    movie_id: Mapped[int | None] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"))
