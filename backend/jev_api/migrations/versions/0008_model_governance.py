"""Model governance: candidates, promotions, rollbacks, dataset snapshots (owner: WS2 retraining and model governance). Empty stub reserved by the Phase 2 enabler: the owning workstream fills
upgrade() and downgrade() in place (docs/PHASE2_ARCHITECTURE_AUDIT.md, "Enabler — done")

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-24 18:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
