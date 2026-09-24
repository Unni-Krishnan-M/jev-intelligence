"""Shared ORM building blocks: the declarative Base, timestamps and CHECK-constraint helpers.

Portable types only (JSON, not JSONB), so the schema runs on PostgreSQL and SQLite.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from jev_api.db import Base
from jev_api.models.enums import DEFAULT_DOMAIN

__all__ = ["Base", "TimestampMixin", "domain_column", "in_", "nullable_in", "utcnow"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )


def in_(column: str, values: tuple[str, ...]) -> str:
    """CHECK text `column IN ('a','b')`."""
    return f"{column} IN ({','.join(repr(v) for v in values)})"


def nullable_in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IS NULL OR {in_(column, values)}"


def domain_column() -> Mapped[str]:
    return mapped_column(String(64), default=DEFAULT_DOMAIN, server_default=DEFAULT_DOMAIN, nullable=False)
