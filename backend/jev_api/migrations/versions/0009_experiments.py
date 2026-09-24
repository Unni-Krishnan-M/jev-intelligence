"""Online experiments, arms and exposure columns (owner: WS5 serving and online experimentation):
ab_experiments, ab_variants, ab_assignments, ab_exposures (every served list, cache hits included),
ab_outcomes, member_decisions (persisted strategy decisions, gap P1 #7) and
recommendations.experiment_id / variant (docs/EXPERIMENTATION.md)

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-24 18:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# literal copies of jev_api.models.enums (a migration must not import the models; the parity test
# compares them)
_SURFACE = "surface IN ('recommendations')"
_STATUS = "status IN ('draft','running','paused','stopped','concluded')"
_ACTIVE = "status IN ('running','paused')"
_PRIMARY = (
    "primary_metric IN ('interaction_rate','positive_rate','rating_rate','feedback_rate','ndcg_at_10',"
    "'diversity','novelty')"
)
_OUTCOME_KIND = "kind IN ('click','like','dislike','not_interested','rating','watch','favorite')"
_OUTCOME_SOURCE = "source IN ('live','replay')"
_CONFIDENCE = "confidence_kind IN ('probability','margin','rule','interval','evidence')"


def _ts(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    op.create_table(
        "ab_experiments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("surface", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("primary_metric", sa.String(length=32), nullable=False),
        sa.Column("guardrails", sa.JSON(), nullable=False),
        sa.Column("traffic_percent", sa.Float(), nullable=False),
        sa.Column("salt", sa.String(length=64), nullable=False),
        sa.Column("attribution_window_hours", sa.Float(), nullable=False),
        sa.Column("analysis", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        _ts("started_at", True),
        _ts("stopped_at", True),
        _ts("concluded_at", True),
        sa.Column("created_by", sa.Integer(), nullable=True),
        _ts("created_at"),
        _ts("updated_at"),
        sa.CheckConstraint(_SURFACE, name="ck_ab_experiment_surface"),
        sa.CheckConstraint(_STATUS, name="ck_ab_experiment_status"),
        sa.CheckConstraint(_PRIMARY, name="ck_ab_experiment_primary_metric"),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key"),
    )
    with op.batch_alter_table("ab_experiments", schema=None) as batch_op:
        batch_op.create_index("ix_ab_experiments_status", ["status", "created_at"], unique=False)
        batch_op.create_index(
            "uq_ab_experiments_active_surface",
            ["surface"],
            unique=True,
            sqlite_where=sa.text(_ACTIVE),
            postgresql_where=sa.text(_ACTIVE),
        )

    op.create_table(
        "ab_variants",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=40), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("is_control", sa.Boolean(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=False),
        sa.Column("description", sa.String(length=300), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["experiment_id"], ["ab_experiments.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_id", "name", name="uq_ab_variants_name"),
    )

    op.create_table(
        "ab_assignments",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("variant_id", sa.Integer(), nullable=False),
        sa.Column("variant", sa.String(length=40), nullable=False),
        sa.Column("bucket", sa.Integer(), nullable=False),
        _ts("assigned_at"),
        sa.ForeignKeyConstraint(["experiment_id"], ["ab_experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["variant_id"], ["ab_variants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("experiment_id", "user_id", name="uq_ab_assignments_user"),
    )
    with op.batch_alter_table("ab_assignments", schema=None) as batch_op:
        batch_op.create_index("ix_ab_assignments_variant", ["experiment_id", "variant"], unique=False)

    op.create_table(
        "ab_exposures",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("source_request_id", sa.String(length=36), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("surface", sa.String(length=32), nullable=False),
        sa.Column("context", sa.String(length=32), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("variant", sa.String(length=40), nullable=True),
        sa.Column("cached", sa.Boolean(), nullable=False),
        sa.Column("model_version", sa.String(length=80), nullable=False),
        sa.Column("decision_id", sa.String(length=40), nullable=True),
        sa.Column("items", sa.JSON(), nullable=False),
        sa.Column("n_items", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        _ts("served_at"),
        sa.ForeignKeyConstraint(["experiment_id"], ["ab_experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("request_id"),
    )
    with op.batch_alter_table("ab_exposures", schema=None) as batch_op:
        batch_op.create_index(
            "ix_ab_exposures_experiment", ["experiment_id", "variant", "served_at"], unique=False
        )
        batch_op.create_index("ix_ab_exposures_user_served", ["user_id", "served_at"], unique=False)

    op.create_table(
        "ab_outcomes",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("exposure_id", sa.Integer(), nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=False),
        sa.Column("variant", sa.String(length=40), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("movie_id", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        _ts("occurred_at"),
        _ts("created_at"),
        sa.CheckConstraint(_OUTCOME_KIND, name="ck_ab_outcome_kind"),
        sa.CheckConstraint(_OUTCOME_SOURCE, name="ck_ab_outcome_source"),
        sa.ForeignKeyConstraint(["exposure_id"], ["ab_exposures.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["experiment_id"], ["ab_experiments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("exposure_id", "kind", "movie_id", name="uq_ab_outcomes_event"),
    )
    with op.batch_alter_table("ab_outcomes", schema=None) as batch_op:
        batch_op.create_index("ix_ab_outcomes_experiment", ["experiment_id", "variant"], unique=False)

    op.create_table(
        "member_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("decision_id", sa.String(length=40), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("spec_id", sa.String(length=64), nullable=False),
        sa.Column("policy_version", sa.String(length=64), nullable=True),
        sa.Column("answer", sa.String(length=32), nullable=True),
        sa.Column("served_strategy", sa.String(length=24), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("confidence_kind", sa.String(length=16), nullable=False),
        sa.Column("abstained", sa.Boolean(), nullable=False),
        sa.Column("state_hash", sa.String(length=64), nullable=False),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("rationale", sa.JSON(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("model_version", sa.String(length=80), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        _ts("created_at"),
        sa.CheckConstraint(_CONFIDENCE, name="ck_member_decision_confidence_kind"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("decision_id", "state_hash", name="uq_member_decisions_state"),
    )
    with op.batch_alter_table("member_decisions", schema=None) as batch_op:
        batch_op.create_index("ix_member_decisions_decision", ["decision_id"], unique=False)
        batch_op.create_index("ix_member_decisions_user_created", ["user_id", "created_at"], unique=False)

    with op.batch_alter_table("recommendations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("experiment_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("variant", sa.String(length=40), nullable=True))
        batch_op.create_index("ix_rec_experiment", ["experiment_id", "variant"], unique=False)


def downgrade() -> None:
    # lossy by design: experiment data and persisted decisions go; served recommendations stay
    with op.batch_alter_table("recommendations", schema=None) as batch_op:
        batch_op.drop_index("ix_rec_experiment")
        batch_op.drop_column("variant")
        batch_op.drop_column("experiment_id")
    with op.batch_alter_table("member_decisions", schema=None) as batch_op:
        batch_op.drop_index("ix_member_decisions_user_created")
        batch_op.drop_index("ix_member_decisions_decision")
    op.drop_table("member_decisions")
    with op.batch_alter_table("ab_outcomes", schema=None) as batch_op:
        batch_op.drop_index("ix_ab_outcomes_experiment")
    op.drop_table("ab_outcomes")
    with op.batch_alter_table("ab_exposures", schema=None) as batch_op:
        batch_op.drop_index("ix_ab_exposures_user_served")
        batch_op.drop_index("ix_ab_exposures_experiment")
    op.drop_table("ab_exposures")
    with op.batch_alter_table("ab_assignments", schema=None) as batch_op:
        batch_op.drop_index("ix_ab_assignments_variant")
    op.drop_table("ab_assignments")
    op.drop_table("ab_variants")
    with op.batch_alter_table("ab_experiments", schema=None) as batch_op:
        batch_op.drop_index(
            "uq_ab_experiments_active_surface",
            sqlite_where=sa.text(_ACTIVE),
            postgresql_where=sa.text(_ACTIVE),
        )
        batch_op.drop_index("ix_ab_experiments_status")
    op.drop_table("ab_experiments")
