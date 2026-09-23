"""v1.1: normalised intel run objects + evidence, audit log, evaluation runs, score decisions,
recommendation confidence (docs/intelligence.md, section 9.3)

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-23 20:27:35.512601
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# confidence_kind gains "interval" (score decisions: nominal coverage of answer_interval)
_DECISION_CK = "confidence_kind IN ('probability','margin','rule')"
_DECISION_CK_V11 = "confidence_kind IN ('probability','margin','rule','interval')"
_REC_CK = "confidence_kind IS NULL OR confidence_kind IN ('probability')"

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "intel_evaluation_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_dir", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("pipeline_version", sa.String(length=40), nullable=True),
        sa.Column("data_version", sa.String(length=120), nullable=True),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=True),
        sa.Column("headline", sa.JSON(), nullable=False),
        sa.Column("report_sha1", sa.String(length=40), nullable=False),
        sa.Column("report", sa.JSON(), nullable=False),
        sa.Column("synced_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_dir"),
    )
    with op.batch_alter_table("intel_evaluation_runs", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_intel_evaluation_runs_created_at"), ["created_at"], unique=False)

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("actor", sa.String(length=320), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=True),
        sa.Column("target_id", sa.String(length=200), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("request_id", sa.String(length=64), nullable=True),
        sa.CheckConstraint(
            "action IN ('intel.run','warning.transition','feedback.create','scenario.save','model.activate','auth.login.success','auth.login.failure','auth.register')",
            name="ck_audit_action",
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.create_index("ix_audit_logs_action_at", ["action", "at"], unique=False)
        batch_op.create_index("ix_audit_logs_actor_at", ["actor", "at"], unique=False)
        batch_op.create_index("ix_audit_logs_at", ["at"], unique=False)
        batch_op.create_index("ix_audit_logs_target", ["target_type", "target_id"], unique=False)

    op.create_table(
        "intel_anomalies",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("anomaly_id", sa.String(length=40), nullable=False),
        sa.Column("dedup_key", sa.String(length=200), nullable=True),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=True),
        sa.Column("entity", sa.String(length=200), nullable=True),
        sa.Column("series_id", sa.String(length=200), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("suppressed", sa.Boolean(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "severity IS NULL OR severity IN ('low','medium','high','critical')",
            name="ck_intel_anomaly_severity",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "anomaly_id", name="uq_intel_anomaly_run"),
    )
    with op.batch_alter_table("intel_anomalies", schema=None) as batch_op:
        batch_op.create_index("ix_intel_anomalies_anomaly_id", ["anomaly_id"], unique=False)
        batch_op.create_index("ix_intel_anomalies_dedup_key", ["dedup_key", "run_id"], unique=False)
        batch_op.create_index("ix_intel_anomalies_detected", ["detected_at"], unique=False)
        batch_op.create_index("ix_intel_anomalies_entity", ["entity_type", "entity"], unique=False)
        batch_op.create_index("ix_intel_anomalies_run_kind", ["run_id", "kind"], unique=False)
        batch_op.create_index("ix_intel_anomalies_score", ["score"], unique=False)
        batch_op.create_index("ix_intel_anomalies_series", ["series_id"], unique=False)
        batch_op.create_index("ix_intel_anomalies_severity", ["severity"], unique=False)

    op.create_table(
        "intel_evidence",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("owner_type", sa.String(length=16), nullable=False),
        sa.Column("owner_id", sa.String(length=200), nullable=False),
        sa.Column("owner_title", sa.String(length=300), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("label", sa.String(length=300), nullable=False),
        sa.Column("value_num", sa.Float(), nullable=True),
        sa.Column("value_text", sa.Text(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("ref", sa.String(length=200), nullable=True),
        sa.CheckConstraint(
            "owner_type IN ('signal','trend','anomaly','forecast','risk','decision','warning','action')",
            name="ck_intel_evidence_owner",
        ),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("intel_evidence", schema=None) as batch_op:
        batch_op.create_index("ix_intel_evidence_kind", ["kind"], unique=False)
        batch_op.create_index("ix_intel_evidence_owner", ["owner_type", "owner_id"], unique=False)
        batch_op.create_index("ix_intel_evidence_ref", ["ref"], unique=False)
        batch_op.create_index("ix_intel_evidence_run_owner", ["run_id", "owner_type"], unique=False)

    op.create_table(
        "intel_forecasts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("forecast_id", sa.String(length=40), nullable=False),
        sa.Column("series_id", sa.String(length=200), nullable=False),
        sa.Column("metric", sa.String(length=32), nullable=True),
        sa.Column("entity_type", sa.String(length=32), nullable=True),
        sa.Column("entity", sa.String(length=200), nullable=True),
        sa.Column("model", sa.String(length=40), nullable=True),
        sa.Column("horizon_months", sa.Integer(), nullable=True),
        sa.Column("mase", sa.Float(), nullable=True),
        sa.Column("coverage80", sa.Float(), nullable=True),
        sa.Column("issued_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "forecast_id", name="uq_intel_forecast_run"),
    )
    with op.batch_alter_table("intel_forecasts", schema=None) as batch_op:
        batch_op.create_index("ix_intel_forecasts_entity", ["entity_type", "entity"], unique=False)
        batch_op.create_index("ix_intel_forecasts_forecast_id", ["forecast_id"], unique=False)
        batch_op.create_index("ix_intel_forecasts_issued", ["issued_at"], unique=False)
        batch_op.create_index("ix_intel_forecasts_mase", ["mase"], unique=False)
        batch_op.create_index("ix_intel_forecasts_series", ["series_id", "run_id"], unique=False)

    op.create_table(
        "intel_risks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("risk_id", sa.String(length=40), nullable=False),
        sa.Column("key", sa.String(length=200), nullable=False),
        sa.Column("kind", sa.String(length=40), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=True),
        sa.Column("entity", sa.String(length=200), nullable=True),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("level", sa.String(length=16), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("likelihood", sa.Float(), nullable=True),
        sa.Column("impact", sa.Float(), nullable=True),
        sa.Column("exposure", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "level IS NULL OR level IN ('low','medium','high','critical')", name="ck_intel_risk_level"
        ),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "risk_id", name="uq_intel_risk_run"),
    )
    with op.batch_alter_table("intel_risks", schema=None) as batch_op:
        batch_op.create_index("ix_intel_risks_entity", ["entity_type", "entity"], unique=False)
        batch_op.create_index("ix_intel_risks_key", ["key", "run_id"], unique=False)
        batch_op.create_index("ix_intel_risks_level", ["level"], unique=False)
        batch_op.create_index("ix_intel_risks_risk_id", ["risk_id"], unique=False)
        batch_op.create_index("ix_intel_risks_run_kind", ["run_id", "kind"], unique=False)
        batch_op.create_index("ix_intel_risks_score", ["score"], unique=False)

    op.create_table(
        "intel_signals",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("signal_id", sa.String(length=40), nullable=False),
        sa.Column("dedup_key", sa.String(length=200), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=True),
        sa.Column("entity", sa.String(length=200), nullable=True),
        sa.Column("title", sa.String(length=300), nullable=False),
        sa.Column("value", sa.Float(), nullable=True),
        sa.Column("strength", sa.Float(), nullable=True),
        sa.Column("direction", sa.String(length=8), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "direction IS NULL OR direction IN ('up','down','flat')", name="ck_intel_signal_direction"
        ),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "signal_id", name="uq_intel_signal_run"),
    )
    with op.batch_alter_table("intel_signals", schema=None) as batch_op:
        batch_op.create_index("ix_intel_signals_dedup_key", ["dedup_key", "run_id"], unique=False)
        batch_op.create_index("ix_intel_signals_entity", ["entity_type", "entity"], unique=False)
        batch_op.create_index("ix_intel_signals_observed", ["observed_at"], unique=False)
        batch_op.create_index("ix_intel_signals_run_kind", ["run_id", "kind"], unique=False)
        batch_op.create_index("ix_intel_signals_signal_id", ["signal_id"], unique=False)
        batch_op.create_index("ix_intel_signals_strength", ["strength"], unique=False)

    op.create_table(
        "intel_trends",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.Integer(), nullable=False),
        sa.Column("trend_id", sa.String(length=40), nullable=False),
        sa.Column("series_id", sa.String(length=200), nullable=False),
        sa.Column("metric", sa.String(length=32), nullable=True),
        sa.Column("entity_type", sa.String(length=32), nullable=True),
        sa.Column("entity", sa.String(length=200), nullable=True),
        sa.Column("direction", sa.String(length=8), nullable=True),
        sa.Column("slope", sa.Float(), nullable=True),
        sa.Column("p_value", sa.Float(), nullable=True),
        sa.Column("q_value", sa.Float(), nullable=True),
        sa.Column("evidence_strength", sa.Float(), nullable=True),
        sa.Column("has_change_point", sa.Boolean(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.CheckConstraint(
            "direction IS NULL OR direction IN ('up','down','flat')", name="ck_intel_trend_direction"
        ),
        sa.ForeignKeyConstraint(["run_id"], ["intel_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "trend_id", name="uq_intel_trend_run"),
    )
    with op.batch_alter_table("intel_trends", schema=None) as batch_op:
        batch_op.create_index("ix_intel_trends_entity", ["entity_type", "entity"], unique=False)
        batch_op.create_index("ix_intel_trends_p_value", ["p_value"], unique=False)
        batch_op.create_index("ix_intel_trends_run_direction", ["run_id", "direction"], unique=False)
        batch_op.create_index("ix_intel_trends_series", ["series_id", "run_id"], unique=False)
        batch_op.create_index("ix_intel_trends_trend_id", ["trend_id"], unique=False)

    with op.batch_alter_table("intel_decisions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("batch_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("answer_value", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("answer_interval", sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column("scale", sa.JSON(), nullable=True))
        batch_op.create_index("ix_intel_decisions_batch", ["batch_id"], unique=False)
        batch_op.drop_constraint("ck_intel_decision_confidence_kind", type_="check")
        batch_op.create_check_constraint("ck_intel_decision_confidence_kind", _DECISION_CK_V11)

    with op.batch_alter_table("recommendations", schema=None) as batch_op:
        batch_op.add_column(sa.Column("confidence", sa.Float(), nullable=True))
        batch_op.add_column(sa.Column("confidence_kind", sa.String(length=16), nullable=True))
        batch_op.create_check_constraint("ck_rec_confidence_kind", _REC_CK)


def downgrade() -> None:
    with op.batch_alter_table("recommendations", schema=None) as batch_op:
        batch_op.drop_constraint("ck_rec_confidence_kind", type_="check")
        batch_op.drop_column("confidence_kind")
        batch_op.drop_column("confidence")

    # score decisions do not fit the v1.0 schema: drop them (and interval confidences) before narrowing
    op.execute("DELETE FROM intel_decisions WHERE kind = 'score' OR confidence_kind = 'interval'")
    with op.batch_alter_table("intel_decisions", schema=None) as batch_op:
        batch_op.drop_constraint("ck_intel_decision_confidence_kind", type_="check")
        batch_op.create_check_constraint("ck_intel_decision_confidence_kind", _DECISION_CK)
        batch_op.drop_index("ix_intel_decisions_batch")
        batch_op.drop_column("scale")
        batch_op.drop_column("answer_interval")
        batch_op.drop_column("answer_value")
        batch_op.drop_column("batch_id")

    with op.batch_alter_table("intel_trends", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_trends_trend_id")
        batch_op.drop_index("ix_intel_trends_series")
        batch_op.drop_index("ix_intel_trends_run_direction")
        batch_op.drop_index("ix_intel_trends_p_value")
        batch_op.drop_index("ix_intel_trends_entity")

    op.drop_table("intel_trends")
    with op.batch_alter_table("intel_signals", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_signals_strength")
        batch_op.drop_index("ix_intel_signals_signal_id")
        batch_op.drop_index("ix_intel_signals_run_kind")
        batch_op.drop_index("ix_intel_signals_observed")
        batch_op.drop_index("ix_intel_signals_entity")
        batch_op.drop_index("ix_intel_signals_dedup_key")

    op.drop_table("intel_signals")
    with op.batch_alter_table("intel_risks", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_risks_score")
        batch_op.drop_index("ix_intel_risks_run_kind")
        batch_op.drop_index("ix_intel_risks_risk_id")
        batch_op.drop_index("ix_intel_risks_level")
        batch_op.drop_index("ix_intel_risks_key")
        batch_op.drop_index("ix_intel_risks_entity")

    op.drop_table("intel_risks")
    with op.batch_alter_table("intel_forecasts", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_forecasts_series")
        batch_op.drop_index("ix_intel_forecasts_mase")
        batch_op.drop_index("ix_intel_forecasts_issued")
        batch_op.drop_index("ix_intel_forecasts_forecast_id")
        batch_op.drop_index("ix_intel_forecasts_entity")

    op.drop_table("intel_forecasts")
    with op.batch_alter_table("intel_evidence", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_evidence_run_owner")
        batch_op.drop_index("ix_intel_evidence_ref")
        batch_op.drop_index("ix_intel_evidence_owner")
        batch_op.drop_index("ix_intel_evidence_kind")

    op.drop_table("intel_evidence")
    with op.batch_alter_table("intel_anomalies", schema=None) as batch_op:
        batch_op.drop_index("ix_intel_anomalies_severity")
        batch_op.drop_index("ix_intel_anomalies_series")
        batch_op.drop_index("ix_intel_anomalies_score")
        batch_op.drop_index("ix_intel_anomalies_run_kind")
        batch_op.drop_index("ix_intel_anomalies_entity")
        batch_op.drop_index("ix_intel_anomalies_detected")
        batch_op.drop_index("ix_intel_anomalies_dedup_key")
        batch_op.drop_index("ix_intel_anomalies_anomaly_id")

    op.drop_table("intel_anomalies")
    with op.batch_alter_table("audit_logs", schema=None) as batch_op:
        batch_op.drop_index("ix_audit_logs_target")
        batch_op.drop_index("ix_audit_logs_at")
        batch_op.drop_index("ix_audit_logs_actor_at")
        batch_op.drop_index("ix_audit_logs_action_at")

    op.drop_table("audit_logs")
    with op.batch_alter_table("intel_evaluation_runs", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_intel_evaluation_runs_created_at"))

    op.drop_table("intel_evaluation_runs")
