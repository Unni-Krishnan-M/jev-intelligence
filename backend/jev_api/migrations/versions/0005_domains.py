"""v1.2 platform: a `domain` on runs, warnings, decisions, scenarios and operator feedback; "one open
warning per key" becomes per (domain, key); the strategy decision behind served recommendations; member
feedback on strategy decisions and recommendations (docs/platform.md, sections 8 and 10)

Revision ID: 0005
Revises: 0004
Create Date: 2026-09-24 14:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# kept in sync with jev_api.models (a migration must not import the models)
_OPEN = "status IN ('new','acknowledged','investigating')"
_AUDIT_V11 = (
    "action IN ('intel.run','warning.transition','feedback.create','scenario.save','model.activate',"
    "'auth.login.success','auth.login.failure','auth.register')"
)
_AUDIT_V12 = (
    "action IN ('intel.run','warning.transition','feedback.create','scenario.save','model.activate',"
    "'auth.login.success','auth.login.failure','auth.register','me.feedback')"
)
# tables whose rows belong to one domain adapter. The normalised run objects (intel_signals, ...,
# intel_evidence) are not listed: each row is owned by exactly one run (run_id, ON DELETE CASCADE) and
# every query reaches them through that run, so a copy of the run's domain could only drift from it.
_DOMAIN_TABLES = ("intel_runs", "intel_warnings", "intel_decisions", "intel_scenarios", "intel_feedback")
_DOMAIN_INDEXES = {
    "intel_runs": ("ix_intel_runs_domain_started", ["domain", "status", "started_at"]),
    "intel_warnings": ("ix_intel_warnings_domain_status", ["domain", "status", "severity"]),
    "intel_decisions": ("ix_intel_decisions_domain_key", ["domain", "key", "created_at"]),
    "intel_scenarios": ("ix_intel_scenarios_domain", ["domain", "created_at"]),
    "intel_feedback": ("ix_intel_feedback_domain", ["domain", "target_type"]),
}

# downgrade: constant SQL (no interpolation) that removes every non-movie row, children first
_DROP_OTHER_DOMAINS = (
    "DELETE FROM intel_signals WHERE run_id IN (SELECT id FROM intel_runs WHERE domain <> 'movie')",
    "DELETE FROM intel_trends WHERE run_id IN (SELECT id FROM intel_runs WHERE domain <> 'movie')",
    "DELETE FROM intel_anomalies WHERE run_id IN (SELECT id FROM intel_runs WHERE domain <> 'movie')",
    "DELETE FROM intel_forecasts WHERE run_id IN (SELECT id FROM intel_runs WHERE domain <> 'movie')",
    "DELETE FROM intel_risks WHERE run_id IN (SELECT id FROM intel_runs WHERE domain <> 'movie')",
    "DELETE FROM intel_evidence WHERE run_id IN (SELECT id FROM intel_runs WHERE domain <> 'movie')",
    "DELETE FROM intel_warning_events WHERE warning_id IN "
    "(SELECT id FROM intel_warnings WHERE domain <> 'movie')",
    "UPDATE intel_warnings SET reopened_from = NULL WHERE reopened_from IN "
    "(SELECT id FROM intel_warnings WHERE domain <> 'movie')",
    "DELETE FROM intel_warnings WHERE domain <> 'movie'",
    "DELETE FROM intel_decisions WHERE domain <> 'movie'",
    "DELETE FROM intel_feedback WHERE domain <> 'movie'",
    "DELETE FROM intel_scenarios WHERE domain <> 'movie'",
    "UPDATE intel_warning_events SET run_id = NULL WHERE run_id IN "
    "(SELECT run_id FROM intel_runs WHERE domain <> 'movie')",
    "UPDATE intel_feedback SET run_id = NULL WHERE run_id IN "
    "(SELECT run_id FROM intel_runs WHERE domain <> 'movie')",
    "DELETE FROM intel_runs WHERE domain <> 'movie'",
)


def upgrade() -> None:
    for table in _DOMAIN_TABLES:
        with op.batch_alter_table(table, schema=None) as batch_op:
            # server_default backfills every existing row with 'movie' (the only domain before v1.2)
            batch_op.add_column(
                sa.Column("domain", sa.String(length=64), nullable=False, server_default="movie")
            )
            name, cols = _DOMAIN_INDEXES[table]
            batch_op.create_index(name, cols, unique=False)

    with op.batch_alter_table("intel_warnings", schema=None) as batch_op:
        batch_op.add_column(sa.Column("decision_id", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("early_warning_level", sa.String(length=16), nullable=True))
        batch_op.drop_index(
            "uq_intel_warnings_open_key", sqlite_where=sa.text(_OPEN), postgresql_where=sa.text(_OPEN)
        )
        batch_op.create_index(
            "uq_intel_warnings_open_key",
            ["domain", "key"],
            unique=True,
            sqlite_where=sa.text(_OPEN),
            postgresql_where=sa.text(_OPEN),
        )

    with op.batch_alter_table("recommendations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("decision_id", sa.String(length=40), nullable=True))
        batch_op.add_column(sa.Column("strategy", sa.String(length=24), nullable=True))
        batch_op.create_index("ix_rec_user_decision", ["user_id", "decision_id"], unique=False)

    op.create_table(
        "user_intel_feedback",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("target_type", sa.String(length=16), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column("verdict", sa.String(length=16), nullable=False),
        sa.Column("note", sa.String(length=1000), nullable=True),
        sa.Column("decision_id", sa.String(length=40), nullable=True),
        sa.Column("movie_id", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "target_type IN ('strategy','recommendation')", name="ck_user_intel_feedback_target"
        ),
        sa.CheckConstraint("verdict IN ('accepted','rejected')", name="ck_user_intel_feedback_verdict"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["movie_id"], ["movies.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "target_type", "target_id", name="uq_user_intel_feedback_target"),
    )
    with op.batch_alter_table("user_intel_feedback", schema=None) as batch_op:
        batch_op.create_index("ix_user_intel_feedback_user_created", ["user_id", "created_at"], unique=False)

    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.drop_constraint("ck_audit_action", type_="check")
        batch_op.create_check_constraint("ck_audit_action", _AUDIT_V12)


def downgrade() -> None:
    # v1.1 knows one domain: rows of every other domain are removed (lossy by design), movie rows stay
    op.execute("DELETE FROM audit_logs WHERE action = 'me.feedback'")
    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.drop_constraint("ck_audit_action", type_="check")
        batch_op.create_check_constraint("ck_audit_action", _AUDIT_V11)

    with op.batch_alter_table("user_intel_feedback", schema=None) as batch_op:
        batch_op.drop_index("ix_user_intel_feedback_user_created")
    op.drop_table("user_intel_feedback")

    with op.batch_alter_table("recommendations", schema=None) as batch_op:
        batch_op.drop_index("ix_rec_user_decision")
        batch_op.drop_column("strategy")
        batch_op.drop_column("decision_id")

    # children before parents (explicit: SQLite enforces ON DELETE only with PRAGMA foreign_keys)
    for statement in _DROP_OTHER_DOMAINS:
        op.execute(statement)

    with op.batch_alter_table("intel_warnings", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_intel_warnings_open_key", sqlite_where=sa.text(_OPEN), postgresql_where=sa.text(_OPEN)
        )
        batch_op.create_index(
            "uq_intel_warnings_open_key",
            ["key"],
            unique=True,
            sqlite_where=sa.text(_OPEN),
            postgresql_where=sa.text(_OPEN),
        )
        batch_op.drop_column("early_warning_level")
        batch_op.drop_column("decision_id")
    for table in reversed(_DOMAIN_TABLES):
        with op.batch_alter_table(table, schema=None) as batch_op:
            batch_op.drop_index(_DOMAIN_INDEXES[table][0])
            batch_op.drop_column("domain")
