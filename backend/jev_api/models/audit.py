"""The audit log (services/audit.py writes it; allowed actions: enums.AUDIT_ACTIONS)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from jev_api.models.base import Base, in_, utcnow
from jev_api.models.enums import AUDIT_ACTIONS


class AuditLog(Base):
    """Who did what, when (never secrets: see services/audit.py)."""

    __tablename__ = "audit_logs"
    __table_args__ = (
        CheckConstraint(in_("action", AUDIT_ACTIONS), name="ck_audit_action"),
        Index("ix_audit_logs_at", "at"),
        Index("ix_audit_logs_action_at", "action", "at"),
        Index("ix_audit_logs_actor_at", "actor", "at"),
        Index("ix_audit_logs_target", "target_type", "target_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    actor_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    actor: Mapped[str] = mapped_column(
        String(320), nullable=False
    )  # "system" or an email (attempted, on failure)
    action: Mapped[str] = mapped_column(String(40), nullable=False)
    target_type: Mapped[str | None] = mapped_column(String(32))
    target_id: Mapped[str | None] = mapped_column(String(200))
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64))
