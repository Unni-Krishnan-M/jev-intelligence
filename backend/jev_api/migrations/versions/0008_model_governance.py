"""Model governance: dataset snapshots, training jobs, per-version lifecycle state (candidate | active |
retired | rejected) with lineage and gate results, and the retrain lock lease (owner: WS2 retraining and
model governance; docs/RETRAINING_AND_MODEL_GOVERNANCE.md)

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-24 18:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# kept in sync with jev_api.models.enums (a migration must not import the models; the parity test checks)
_MODEL_STATES = "state IN ('candidate','active','retired','rejected')"
_JOB_KINDS = "kind IN ('retrain','evaluate')"
_JOB_STATUSES = "status IN ('queued','running','succeeded','failed')"
_JOB_TRIGGERS = "\"trigger\" IN ('manual','schedule','decision','cli')"


def _ts(name: str, nullable: bool = True) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    op.create_table(
        "dataset_snapshots",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("snapshot_id", sa.String(64), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("base_dataset_version", sa.String(120), nullable=True),
        _ts("cutoff"),
        sa.Column("row_counts", sa.JSON(), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=False),
        sa.Column("watermark", sa.JSON(), nullable=False),
        sa.Column("semantics_version", sa.String(40), nullable=False),
        sa.Column("path", sa.String(500), nullable=False),
        sa.Column("created_by", sa.String(320), nullable=False),
        _ts("created_at", nullable=False),
        sa.UniqueConstraint("snapshot_id", name="uq_dataset_snapshots_snapshot_id"),
    )
    op.create_index("ix_dataset_snapshots_created", "dataset_snapshots", ["created_at"])

    op.create_table(
        "training_jobs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("trigger", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("quick", sa.Boolean(), nullable=False),
        sa.Column("config_path", sa.String(500), nullable=True),
        sa.Column("decision_id", sa.String(40), nullable=True),
        sa.Column("snapshot_id", sa.String(64), nullable=True),
        sa.Column("model_version", sa.String(80), nullable=True),
        sa.Column("incumbent_version", sa.String(80), nullable=True),
        sa.Column("gate_passed", sa.Boolean(), nullable=True),
        sa.Column("promoted", sa.Boolean(), nullable=False),
        sa.Column("requested_by", sa.String(320), nullable=False),
        _ts("created_at", nullable=False),
        _ts("started_at"),
        _ts("finished_at"),
        sa.Column("duration_ms", sa.Float(), nullable=True),
        sa.Column("steps", sa.JSON(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.UniqueConstraint("job_id", name="uq_training_jobs_job_id"),
        sa.CheckConstraint(_JOB_KINDS, name="ck_training_job_kind"),
        sa.CheckConstraint(_JOB_STATUSES, name="ck_training_job_status"),
        sa.CheckConstraint(_JOB_TRIGGERS, name="ck_training_job_trigger"),
    )
    op.create_index("ix_training_jobs_status_created", "training_jobs", ["status", "created_at"])
    op.create_index("ix_training_jobs_decision", "training_jobs", ["decision_id"])

    op.create_table(
        "model_governance",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("version", sa.String(80), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("snapshot_id", sa.String(64), nullable=True),
        sa.Column("job_id", sa.String(36), nullable=True),
        sa.Column("decision_id", sa.String(40), nullable=True),
        sa.Column("config_hash", sa.String(64), nullable=True),
        sa.Column("git_commit", sa.String(40), nullable=True),
        sa.Column("seed", sa.Integer(), nullable=True),
        sa.Column("lineage", sa.JSON(), nullable=False),
        sa.Column("gate", sa.JSON(), nullable=True),
        sa.Column("gate_passed", sa.Boolean(), nullable=True),
        sa.Column("gate_reasons", sa.JSON(), nullable=False),
        sa.Column("gated_against", sa.String(80), nullable=True),
        _ts("gated_at"),
        _ts("promoted_at"),
        sa.Column("promoted_by", sa.String(320), nullable=True),
        sa.Column("forced", sa.Boolean(), nullable=False),
        sa.Column("force_reason", sa.String(500), nullable=True),
        _ts("retired_at"),
        _ts("created_at", nullable=False),
        _ts("updated_at", nullable=False),
        sa.UniqueConstraint("version", name="uq_model_governance_version"),
        sa.CheckConstraint(_MODEL_STATES, name="ck_model_governance_state"),
    )
    op.create_index("ix_model_governance_state", "model_governance", ["state"])

    op.create_table(
        "governance_locks",
        sa.Column("name", sa.String(64), primary_key=True),
        sa.Column("holder", sa.String(120), nullable=False),
        _ts("acquired_at", nullable=False),
        _ts("expires_at", nullable=False),
    )


def downgrade() -> None:
    op.drop_table("governance_locks")
    op.drop_index("ix_model_governance_state", table_name="model_governance")
    op.drop_table("model_governance")
    op.drop_index("ix_training_jobs_decision", table_name="training_jobs")
    op.drop_index("ix_training_jobs_status_created", table_name="training_jobs")
    op.drop_table("training_jobs")
    op.drop_index("ix_dataset_snapshots_created", table_name="dataset_snapshots")
    op.drop_table("dataset_snapshots")
