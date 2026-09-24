"""Service tokens and token revocation (owner: integrator, security hardening;
docs/SECURITY_AUDIT_PHASE2.md): users.token_version, service_tokens (hashed, scoped, expiring,
revocable machine credentials) and revoked_tokens (the jti denylist written by logout)

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-24 18:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _ts(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    # existing sessions carry no `ver` claim and are rejected after the upgrade: members sign in once more
    with op.batch_alter_table("users") as batch:
        batch.add_column(sa.Column("token_version", sa.Integer(), server_default="0", nullable=False))

    op.create_table(
        "service_tokens",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(80), nullable=False),
        sa.Column("token_prefix", sa.String(16), nullable=False),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=False),
        sa.Column(
            "created_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("created_by", sa.String(320), nullable=False),
        _ts("created_at"),
        _ts("expires_at"),
        _ts("revoked_at", nullable=True),
        sa.Column("revoked_by", sa.String(320), nullable=True),
        _ts("last_used_at", nullable=True),
        sa.UniqueConstraint("token_hash", name="uq_service_tokens_token_hash"),
    )

    op.create_table(
        "revoked_tokens",
        sa.Column("jti", sa.String(64), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        _ts("revoked_at"),
        _ts("expires_at"),
    )
    op.create_index("ix_revoked_tokens_user_id", "revoked_tokens", ["user_id"])
    op.create_index("ix_revoked_tokens_expires_at", "revoked_tokens", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_revoked_tokens_expires_at", table_name="revoked_tokens")
    op.drop_index("ix_revoked_tokens_user_id", table_name="revoked_tokens")
    op.drop_table("revoked_tokens")
    op.drop_table("service_tokens")
    with op.batch_alter_table("users") as batch:
        batch.drop_column("token_version")
