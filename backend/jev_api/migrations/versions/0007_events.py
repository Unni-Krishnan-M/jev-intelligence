"""Append-only event log with idempotency keys (owner: WS1 events and ingestion, docs/STREAMING_ARCHITECTURE.md)

- ``events``: the append-only log. BEFORE UPDATE / DELETE triggers (and TRUNCATE on PostgreSQL) abort any
  change to a stored row, on SQLite and PostgreSQL alike.
- ``idempotency_keys``: the stored first response of a request that carried an ``Idempotency-Key``.
- ``event_daily_counts``: per-domain, per-day ingest counters (accepted / duplicates / rejected).
- ``intel_runs.event_watermark``: the events an intelligence run read (lineage).

Backfill: every existing projection row (ratings, favorites, watch_history, recommendation_feedback)
becomes one event with ``source = 'backfill'``, ``event_time = ingested_at =`` the row's own timestamp
(``updated_at`` for a rating, whose ``created_at`` travels in the payload), and the idempotency key
``<table>:<row id>``. Folding the backfilled log therefore reproduces the projections exactly. The
downgrade drops the new tables and the column; the projections are untouched in both directions.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-24 18:00:00.000000
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# kept in sync with jev_api.models.enums.EVENT_TYPES (the CHECK parity test compares them)
_EVENT_TYPES = ("rating", "rating_removed", "watch", "favorite", "unfavorite", "rec_feedback", "observation")
_DEFAULT_DOMAIN = "movie"
_SCHEMA_VERSION = 1


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN ({','.join(repr(v) for v in values)})"


def _iso(dt: Any) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return str(dt.astimezone(UTC).isoformat())


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(36), nullable=False),
        sa.Column("source", sa.String(32), nullable=False),
        sa.Column("domain", sa.String(64), server_default=_DEFAULT_DOMAIN, nullable=False),
        sa.Column("event_type", sa.String(16), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("entity_id", sa.String(128), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("idempotency_key", sa.String(200), nullable=True),
        sa.Column("batch_id", sa.String(36), nullable=True),
        sa.Column("schema_version", sa.Integer(), server_default=str(_SCHEMA_VERSION), nullable=False),
        sa.CheckConstraint(_in("event_type", _EVENT_TYPES), name="ck_event_type"),
        sa.UniqueConstraint("event_id", name="uq_events_event_id"),
        sa.UniqueConstraint("source", "idempotency_key", name="uq_events_source_key"),
    )
    op.create_index("ix_events_domain_time", "events", ["domain", "event_time", "id"])
    op.create_index("ix_events_domain_ingested", "events", ["domain", "ingested_at"])
    op.create_index("ix_events_user_entity", "events", ["user_id", "entity_id", "event_type"])
    op.create_index("ix_events_batch", "events", ["batch_id"])

    op.create_table(
        "idempotency_keys",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("scope", sa.String(64), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("response", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("scope", "key", name="uq_idempotency_scope_key"),
    )
    op.create_table(
        "event_daily_counts",
        sa.Column("domain", sa.String(64), primary_key=True),
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("accepted", sa.Integer(), server_default="0", nullable=False),
        sa.Column("duplicates", sa.Integer(), server_default="0", nullable=False),
        sa.Column("rejected", sa.Integer(), server_default="0", nullable=False),
    )
    with op.batch_alter_table("intel_runs") as batch:
        batch.add_column(sa.Column("event_watermark", sa.JSON(), nullable=True))

    _backfill()
    _append_only_triggers()


def _backfill() -> None:
    """One backfill event per existing projection row, in event-time order."""
    bind = op.get_bind()
    md = sa.MetaData()
    ratings = sa.Table("ratings", md, autoload_with=bind)
    favorites = sa.Table("favorites", md, autoload_with=bind)
    watches = sa.Table("watch_history", md, autoload_with=bind)
    feedback = sa.Table("recommendation_feedback", md, autoload_with=bind)
    events = sa.Table("events", md, autoload_with=bind)
    rows: list[dict[str, Any]] = []

    def add(table: str, rid: int, etype: str, uid: int, mid: int, value: Any, payload: dict, t: Any) -> None:
        when = datetime.fromisoformat(t) if isinstance(t, str) else t
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        rows.append(
            {
                "event_id": str(uuid.uuid4()),
                "source": "backfill",
                "domain": _DEFAULT_DOMAIN,
                "event_type": etype,
                "user_id": uid,
                "entity_id": str(mid),
                "value": value,
                "payload": payload,
                "event_time": when,
                "ingested_at": when,
                "idempotency_key": f"{table}:{rid}",
                "batch_id": None,
                "schema_version": _SCHEMA_VERSION,
            }
        )

    for r in bind.execute(sa.select(ratings)).mappings():
        payload = {"created_at": _iso(r["created_at"])}
        add("ratings", r["id"], "rating", r["user_id"], r["movie_id"], r["rating"], payload, r["updated_at"])
    for r in bind.execute(sa.select(favorites)).mappings():
        payload = {"source": r["source"]}
        add("favorites", r["id"], "favorite", r["user_id"], r["movie_id"], None, payload, r["created_at"])
    for r in bind.execute(sa.select(watches)).mappings():
        add("watch_history", r["id"], "watch", r["user_id"], r["movie_id"], None, {}, r["watched_at"])
    for r in bind.execute(sa.select(feedback)).mappings():
        payload = {"feedback": r["feedback"], "recommendation_id": r["recommendation_id"]}
        add(
            "recommendation_feedback", r["id"], "rec_feedback", r["user_id"], r["movie_id"], None, payload,
            r["created_at"],
        )  # fmt: skip
    rows.sort(key=lambda e: (e["event_time"], e["idempotency_key"]))
    for i in range(0, len(rows), 1000):
        bind.execute(sa.insert(events), rows[i : i + 1000])


def _append_only_triggers() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute(
            "CREATE OR REPLACE FUNCTION jev_events_append_only() RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION 'events is append-only (% refused)', TG_OP; END $$"
        )
        op.execute(
            "CREATE TRIGGER events_append_only BEFORE UPDATE OR DELETE ON events "
            "FOR EACH ROW EXECUTE FUNCTION jev_events_append_only()"
        )
        op.execute(
            "CREATE TRIGGER events_no_truncate BEFORE TRUNCATE ON events "
            "FOR EACH STATEMENT EXECUTE FUNCTION jev_events_append_only()"
        )
    else:
        for verb in ("UPDATE", "DELETE"):
            op.execute(
                f"CREATE TRIGGER events_append_only_{verb.lower()} BEFORE {verb} ON events "
                f"BEGIN SELECT RAISE(ABORT, 'events is append-only ({verb} refused)'); END"
            )


def downgrade() -> None:
    postgres = op.get_bind().dialect.name == "postgresql"
    with op.batch_alter_table("intel_runs") as batch:
        batch.drop_column("event_watermark")
    op.drop_table("event_daily_counts")
    op.drop_table("idempotency_keys")
    op.drop_index("ix_events_batch", table_name="events")
    op.drop_index("ix_events_user_entity", table_name="events")
    op.drop_index("ix_events_domain_ingested", table_name="events")
    op.drop_index("ix_events_domain_time", table_name="events")
    op.drop_table("events")  # drops its triggers too
    if postgres:
        op.execute("DROP FUNCTION IF EXISTS jev_events_append_only()")
