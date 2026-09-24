"""Append-only event log (WS1: events and ingestion, migration 0007, docs/STREAMING_ARCHITECTURE.md).

- ``events``: one row per thing that happened. Never updated or deleted: migration 0007 installs
  BEFORE UPDATE / BEFORE DELETE triggers (SQLite and PostgreSQL) that abort the statement. ``id`` is the
  monotonically increasing sequence number; ``event_time`` is when it happened (valid time) and
  ``ingested_at`` when JEV learned about it (knowledge time).
- ``idempotency_keys``: the stored response of a request that carried an ``Idempotency-Key`` header, so
  a retry returns the original result.
- ``event_daily_counts``: incrementally maintained per-domain, per-day ingest counters (monitoring).

The current-state tables (ratings, favorites, watch_history, recommendation_feedback) are projections
of ``events`` maintained in the same transaction (jev_api.services.events).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from jev_api.models.base import Base, domain_column, in_, utcnow
from jev_api.models.enums import EVENT_TYPES

__all__ = ["EVENT_SCHEMA_VERSION", "Event", "EventDailyCount", "IdempotencyKey"]

EVENT_SCHEMA_VERSION = 1


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        CheckConstraint(in_("event_type", EVENT_TYPES), name="ck_event_type"),
        # NULL keys never collide (both PostgreSQL and SQLite treat NULLs as distinct)
        UniqueConstraint("event_id", name="uq_events_event_id"),
        UniqueConstraint("source", "idempotency_key", name="uq_events_source_key"),
        # as-of reads and replays: WHERE domain = ? AND event_time <= ? AND ingested_at <= ? ORDER BY id
        Index("ix_events_domain_time", "domain", "event_time", "id"),
        Index("ix_events_domain_ingested", "domain", "ingested_at"),
        # projection refolds: one member's events on one entity
        Index("ix_events_user_entity", "user_id", "entity_id", "event_type"),
        Index("ix_events_batch", "batch_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # the sequence number
    event_id: Mapped[str] = mapped_column(String(36), nullable=False)  # a UUID
    source: Mapped[str] = mapped_column(String(32), nullable=False)  # app | ingest | backfill
    domain: Mapped[str] = domain_column()
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    # no foreign key: the log outlives the mutable tables it describes (erasure is a retention job)
    user_id: Mapped[int | None] = mapped_column(Integer)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(200))
    batch_id: Mapped[str | None] = mapped_column(String(36))
    schema_version: Mapped[int] = mapped_column(
        Integer, default=EVENT_SCHEMA_VERSION, server_default=str(EVENT_SCHEMA_VERSION), nullable=False
    )


class IdempotencyKey(Base):
    """The first response to a request with an Idempotency-Key, replayed to every retry."""

    __tablename__ = "idempotency_keys"
    __table_args__ = (UniqueConstraint("scope", "key", name="uq_idempotency_scope_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    scope: Mapped[str] = mapped_column(String(64), nullable=False)  # "user:<id>"
    key: Mapped[str] = mapped_column(String(200), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)  # sha256(method, path, body)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class EventDailyCount(Base):
    """Per-domain, per-day ingest counters, upserted with every ingest (GET /events/health)."""

    __tablename__ = "event_daily_counts"

    domain: Mapped[str] = mapped_column(String(64), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    accepted: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    duplicates: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
    rejected: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)
