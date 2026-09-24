"""Append-only event log with idempotency keys (owner: WS1 events and ingestion). Empty stub reserved by the Phase 2 enabler: the owning workstream fills
upgrade() and downgrade() in place (docs/PHASE2_ARCHITECTURE_AUDIT.md, "Enabler — done")

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-24 18:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
