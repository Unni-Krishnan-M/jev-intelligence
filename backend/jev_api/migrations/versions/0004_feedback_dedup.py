"""recommendation feedback: drop duplicates, then one verdict and one click per user per recommendation
(or per movie when no recommendation is attached)

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-24 10:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# kept in sync with jev_api.models.FEEDBACK_UNIQUE_INDEXES (a migration must not import the models)
_WITH_REC, _NO_REC = "recommendation_id IS NOT NULL", "recommendation_id IS NULL"
_VERDICT, _CLICK = "feedback <> 'clicked'", "feedback = 'clicked'"
_INDEXES = (
    ("uq_feedback_verdict_rec", "recommendation_id", f"{_WITH_REC} AND {_VERDICT}"),
    ("uq_feedback_verdict_movie", "movie_id", f"{_NO_REC} AND {_VERDICT}"),
    ("uq_feedback_click_rec", "recommendation_id", f"{_WITH_REC} AND {_CLICK}"),
    ("uq_feedback_click_movie", "movie_id", f"{_NO_REC} AND {_CLICK}"),
)

# A row is a duplicate when another row has the same key and is newer (created_at, then id). The key
# is (user, recommendation) or, without a recommendation, (user, movie), each split into verdicts and
# clicks. Constant SQL that runs unchanged on SQLite and PostgreSQL; the newest row per key survives.
_DELETE_DUPLICATES = """
DELETE FROM recommendation_feedback WHERE id IN (
  SELECT f.id FROM recommendation_feedback f
  JOIN recommendation_feedback g
    ON g.user_id = f.user_id
   AND g.id <> f.id
   AND (CASE WHEN g.feedback = 'clicked' THEN 1 ELSE 0 END)
     = (CASE WHEN f.feedback = 'clicked' THEN 1 ELSE 0 END)
   AND ((f.recommendation_id IS NOT NULL AND g.recommendation_id = f.recommendation_id)
     OR (f.recommendation_id IS NULL AND g.recommendation_id IS NULL AND g.movie_id = f.movie_id))
   AND (g.created_at > f.created_at OR (g.created_at = f.created_at AND g.id > f.id))
)
"""


def upgrade() -> None:
    op.execute(_DELETE_DUPLICATES)
    with op.batch_alter_table("recommendation_feedback", schema=None) as batch_op:
        for name, col, where in _INDEXES:
            batch_op.create_index(
                name,
                ["user_id", col],
                unique=True,
                sqlite_where=sa.text(where),
                postgresql_where=sa.text(where),
            )


def downgrade() -> None:
    # the removed duplicates are not restored (they carried no information beyond the newest row)
    with op.batch_alter_table("recommendation_feedback", schema=None) as batch_op:
        for name, _col, where in reversed(_INDEXES):
            batch_op.drop_index(name, sqlite_where=sa.text(where), postgresql_where=sa.text(where))
