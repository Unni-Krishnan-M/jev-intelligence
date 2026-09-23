"""intelligence layer: runs, warnings (+ events), decisions, scenarios, feedback

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-23 16:57:09.637812
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "intel_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("trigger", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("requested_as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("pipeline_version", sa.String(length=40), nullable=False),
        sa.Column("data_version", sa.String(length=120), nullable=True),
        sa.Column("model_version", sa.String(length=80), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=True),
        sa.Column("stage_ms", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("created_by_id", sa.Integer(), nullable=True),
        sa.CheckConstraint("status IN ('running','succeeded','failed')", name="ck_intel_run_status"),
        sa.CheckConstraint(
            "\"trigger\" IN ('startup','manual','script','schedule')", name="ck_intel_run_trigger"
        ),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id"),
    )
    with op.batch_alter_table("intel_runs", schema=None) as batch_op:
        batch_op.create_index("ix_intel_runs_status_started", ["status", "started_at"], unique=False)

    op.create_table(
        "intel_scenarios",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("series_id", sa.String(length=200), nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("input", sa.JSON(), nullable=False),
        sa.Column("output", sa.JSON(), nullable=False),
        sa.Column("created_by_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("intel_scenarios", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_intel_scenarios_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_intel_scenarios_series_id"), ["series_id"], unique=False)

    op.create_table(
        "intel_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=36), nullable=False),
        sa.Column("decision_id", sa.String(length=40), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("spec_id", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=40), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("answer", sa.String(length=64), nullable=True),
        sa.Column("option_scores", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confidence_kind", sa.String(length=16), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("rationale", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("abstained", sa.Boolean(), nullable=False),
        sa.Column("fallback_reason", sa.Text(), nullable=True),
        sa.Column("entity_type", sa.String(length=32), nullable=True),
        sa.Column("entity", sa.String(length=200), nullable=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "confidence_kind IN ('probability','margin','rule')", name="ck_intel_decision_confidence_kind"
        ),
        sa.CheckConstraint("kind IN ('boolean','choice','score')", name="ck_intel_decision_kind"),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.run_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "decision_id", name="uq_intel_decision_run"),
    )
    with op.batch_alter_table("intel_decisions", schema=None) as batch_op:
        batch_op.create_index("ix_intel_decisions_decision_id", ["decision_id"], unique=False)
        batch_op.create_index("ix_intel_decisions_key_created", ["key", "created_at"], unique=False)

    op.create_table(
        "intel_feedback",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("target_type", sa.String(length=16), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("note", sa.String(length=1000), nullable=True),
        sa.Column("outcome", sa.String(length=500), nullable=True),
        sa.Column("actor", sa.String(length=320), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "(target_type = 'decision' AND verdict IN ('correct','incorrect')) OR (target_type = 'warning' AND verdict IN ('useful','not_useful','false_positive')) OR (target_type = 'action' AND verdict IN ('useful','not_useful')) OR (target_type = 'prediction' AND verdict IN ('correct','incorrect'))",
            name="ck_intel_feedback_verdict",
        ),
        sa.CheckConstraint(
            "target_type IN ('decision','warning','action','prediction')", name="ck_intel_feedback_target"
        ),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.run_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("intel_feedback", schema=None) as batch_op:
        batch_op.create_index("ix_intel_feedback_created", ["created_at"], unique=False)
        batch_op.create_index("ix_intel_feedback_target", ["target_type", "target_id"], unique=False)

    op.create_table(
        "intel_warnings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confidence_kind", sa.String(length=16), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("trigger", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("recommended_action", sa.Text(), nullable=False),
        sa.Column("source", sa.JSON(), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=True),
        sa.Column("entity", sa.String(length=200), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("occurrences", sa.Integer(), nullable=False),
        sa.Column("first_seen_run_id", sa.String(length=36), nullable=True),
        sa.Column("last_seen_run_id", sa.String(length=36), nullable=True),
        sa.Column("reopened_from", sa.Integer(), nullable=True),
        sa.Column("dismissed_severity", sa.String(length=16), nullable=True),
        sa.Column("suppressed_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "dismissed_severity IS NULL OR dismissed_severity IN ('low','medium','high','critical')",
            name="ck_intel_warning_dismissed_severity",
        ),
        sa.CheckConstraint(
            "severity IN ('low','medium','high','critical')", name="ck_intel_warning_severity"
        ),
        sa.CheckConstraint(
            "status IN ('new','acknowledged','investigating','resolved','dismissed')",
            name="ck_intel_warning_status",
        ),
        sa.CheckConstraint("occurrences >= 1", name="ck_intel_warning_occurrences"),
        sa.ForeignKeyConstraint(["first_seen_run_id"], ["intel_runs.run_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["last_seen_run_id"], ["intel_runs.run_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["reopened_from"], ["intel_warnings.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("intel_warnings", schema=None) as batch_op:
        batch_op.create_index("ix_intel_warnings_key", ["key"], unique=False)
        batch_op.create_index("ix_intel_warnings_last_seen", ["last_seen_at"], unique=False)
        batch_op.create_index("ix_intel_warnings_status_severity", ["status", "severity"], unique=False)
        batch_op.create_index(
            "uq_intel_warnings_open_key",
            ["key"],
            unique=True,
            sqlite_where=sa.text("status IN ('new','acknowledged','investigating')"),
            postgresql_where=sa.text("status IN ('new','acknowledged','investigating')"),
        )

    op.create_table(
        "intel_warning_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("warning_id", sa.Integer(), nullable=False),
        sa.Column("from_status", sa.String(length=16), nullable=True),
        sa.Column("to_status", sa.String(length=16), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("actor", sa.String(length=320), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("run_id", sa.String(length=36), nullable=True),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "from_status IS NULL OR from_status IN ('new','acknowledged','investigating','resolved','dismissed')",
            name="ck_intel_event_from",
        ),
        sa.CheckConstraint(
            "to_status IN ('new','acknowledged','investigating','resolved','dismissed')",
            name="ck_intel_event_to",
        ),
        sa.ForeignKeyConstraint(["actor_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.run_id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["warning_id"], ["intel_warnings.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("intel_warning_events", schema=None) as batch_op:
        batch_op.create_index("ix_intel_events_warning_at", ["warning_id", "at"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("intel_warning_events", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_events_warning_at")

    op.drop_table("intel_warning_events")
    with op.batch_alter_table("intel_warnings", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_intel_warnings_open_key",
            sqlite_where=sa.text("status IN ('new','acknowledged','investigating')"),
            postgresql_where=sa.text("status IN ('new','acknowledged','investigating')"),
        )
        batch_op.drop_index("ix_intel_warnings_status_severity")
        batch_op.drop_index("ix_intel_warnings_last_seen")
        batch_op.drop_index("ix_intel_warnings_key")

    op.drop_table("intel_warnings")
    with op.batch_alter_table("intel_feedback", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_feedback_target")
        batch_op.drop_index("ix_intel_feedback_created")

    op.drop_table("intel_feedback")
    with op.batch_alter_table("intel_decisions", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_decisions_key_created")
        batch_op.drop_index("ix_intel_decisions_decision_id")

    op.drop_table("intel_decisions")
    with op.batch_alter_table("intel_scenarios", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_intel_scenarios_series_id"))
        batch_op.drop_index(batch_op.f("ix_intel_scenarios_created_at"))

    op.drop_table("intel_scenarios")
    with op.batch_alter_table("intel_runs", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_runs_status_started")

    op.drop_table("intel_runs")
