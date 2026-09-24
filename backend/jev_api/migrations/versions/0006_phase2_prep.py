"""Phase 2 enablers: intel_runs.mode (live | replay), the "evidence" confidence kind on decisions, and every
Phase 2 audit action registered up front (docs/PHASE2_ARCHITECTURE_AUDIT.md, "Enabler — done")

Backfill of intel_runs.mode. A run is a replay when it was asked for an explicit as_of
(requested_as_of IS NOT NULL) AND its as_of lies before the domain's data end. The data end is not stored
anywhere, so it is inferred from the runs themselves: a run without an explicit as_of resolves as_of to the
data end (movie: the last MovieLens event; generic: now), so the domain's data end is the newest as_of of its
runs that had no explicit as_of. When a domain has no such run the data end cannot be inferred, and its
explicit-as_of runs stay "live" (the column default). An explicit as_of at or after the data end is an
analysis of the current data, so it also stays "live". New runs record their mode at creation
(jev_api.models.intel.run_mode: explicit as_of = replay).

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-24 18:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# kept in sync with jev_api.models.enums (a migration must not import the models; the CHECK parity test
# in tests/integration/test_migrations.py compares the migrated database with enums.py)
_AUDIT_V12_ACTIONS = (
    "intel.run",
    "warning.transition",
    "feedback.create",
    "scenario.save",
    "model.activate",
    "auth.login.success",
    "auth.login.failure",
    "auth.register",
    "me.feedback",
)
_AUDIT_PHASE2_ACTIONS = (
    "events.ingest",
    "events.replay",
    "model.register",
    "model.retrain",
    "model.promote",
    "model.reject",
    "model.rollback",
    "dataset.snapshot",
    "warning.auto_resolve",
    "experiment.create",
    "experiment.start",
    "experiment.ramp",
    "experiment.pause",
    "experiment.stop",
    "experiment.conclude",
    "token.create",
    "token.revoke",
    "auth.logout",
    "auth.revoke_all",
    "auth.password_change",
    "auth.login.throttled",
    "user.role_change",
    "retention.prune",
)


def _in(column: str, values: Sequence[str]) -> str:
    return f"{column} IN ({','.join(repr(v) for v in values)})"


_AUDIT_V12 = _in("action", _AUDIT_V12_ACTIONS)
_AUDIT_PHASE2 = _in("action", _AUDIT_V12_ACTIONS + _AUDIT_PHASE2_ACTIONS)
_DECISION_CK_V11 = "confidence_kind IN ('probability','margin','rule','interval')"
_DECISION_CK_PHASE2 = "confidence_kind IN ('probability','margin','rule','interval','evidence')"
_RUN_MODE_CK = "mode IN ('live','replay')"

# constant SQL: see the module docstring for the rule. COALESCE: a run that failed before resolving its
# as_of is judged by the as_of it asked for.
_BACKFILL_REPLAYS = """
UPDATE intel_runs SET mode = 'replay'
WHERE requested_as_of IS NOT NULL
  AND COALESCE(as_of, requested_as_of) < (
    SELECT MAX(ref.as_of) FROM intel_runs ref
    WHERE ref.domain = intel_runs.domain AND ref.requested_as_of IS NULL AND ref.as_of IS NOT NULL
  )
"""


def upgrade() -> None:
    with op.batch_alter_table("intel_runs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("mode", sa.String(length=8), nullable=False, server_default="live"))
        batch_op.create_check_constraint("ck_intel_run_mode", _RUN_MODE_CK)
        batch_op.create_index(
            "ix_intel_runs_domain_mode_started", ["domain", "mode", "status", "started_at"], unique=False
        )
    op.execute(_BACKFILL_REPLAYS)

    with op.batch_alter_table("intel_decisions", schema=None) as batch_op:
        batch_op.drop_constraint("ck_intel_decision_confidence_kind", type_="check")
        batch_op.create_check_constraint("ck_intel_decision_confidence_kind", _DECISION_CK_PHASE2)

    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.drop_constraint("ck_audit_action", type_="check")
        batch_op.create_check_constraint("ck_audit_action", _AUDIT_PHASE2)


def downgrade() -> None:
    # rows that use the new values go first, or the narrowed constraints would reject the table copy
    placeholders = ", ".join(f":a{i}" for i in range(len(_AUDIT_PHASE2_ACTIONS)))
    op.get_bind().execute(
        sa.text(f"DELETE FROM audit_logs WHERE action IN ({placeholders})"),  # noqa: S608 - bind names only
        {f"a{i}": action for i, action in enumerate(_AUDIT_PHASE2_ACTIONS)},
    )
    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.drop_constraint("ck_audit_action", type_="check")
        batch_op.create_check_constraint("ck_audit_action", _AUDIT_V12)

    op.execute("DELETE FROM intel_decisions WHERE confidence_kind = 'evidence'")
    with op.batch_alter_table("intel_decisions", schema=None) as batch_op:
        batch_op.drop_constraint("ck_intel_decision_confidence_kind", type_="check")
        batch_op.create_check_constraint("ck_intel_decision_confidence_kind", _DECISION_CK_V11)

    # the mode is dropped, the runs stay (a v1.2 schema cannot tell a replay from a live run)
    with op.batch_alter_table("intel_runs", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_runs_domain_mode_started")
        batch_op.drop_constraint("ck_intel_run_mode", type_="check")
        batch_op.drop_column("mode")
