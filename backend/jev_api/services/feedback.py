"""Recommendation feedback: one verdict per member per recommendation, and counts that stay honest.

Writes are upserts (migration 0004 enforces the keys with partial unique indexes):

* a verdict (like / dislike / not_interested) replaces the member's earlier verdict on the same
  recommendation, or on the same movie when no recommendation is attached;
* a click is an interaction, not a judgement: it has its own slot per key and never replaces a verdict.
  Repeated clicks are a no-op.

Reads that aggregate feedback (intel monitoring, the live-feedback input of the pipeline, admin stats)
go through `latest_feedback()`, which keeps only the newest verdict and the newest click per member
per movie. That also covers a member who collects many recommendation ids for one movie (every served
list creates new ones), so no single member can move a rate by repeating themselves.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, case, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from jev_api.models import RecommendationFeedback, User, utcnow

CLICK = "clicked"
VERDICTS = ("like", "dislike", "not_interested")
NEGATIVE = frozenset({"dislike", "not_interested"})  # these exclude the movie from recommendations


def upsert_feedback(
    db: Session, user: User, movie_id: int, kind: str, recommendation_id: int | None
) -> RecommendationFeedback:
    """Record `kind` as the member's current feedback for the key and commit. Returns the row."""
    fb = RecommendationFeedback
    for attempt in range(2):
        same_slot = (fb.feedback == CLICK) if kind == CLICK else (fb.feedback != CLICK)
        q = select(fb).where(fb.user_id == user.id, same_slot)
        if recommendation_id is not None:
            q = q.where(fb.recommendation_id == recommendation_id)
        else:
            q = q.where(fb.recommendation_id.is_(None), fb.movie_id == movie_id)
        row = db.scalars(q.order_by(fb.created_at.desc(), fb.id.desc())).first()
        previous = row.feedback if row is not None else None
        if row is None:
            row = fb(user_id=user.id, movie_id=movie_id, feedback=kind, recommendation_id=recommendation_id)
            db.add(row)
        elif previous != kind:
            row.feedback = kind
            row.created_at = utcnow()  # the verdict changed now
        if (previous in NEGATIVE) != (kind in NEGATIVE):
            user.profile_version += 1  # the exclusion list changed: cached recommendations are stale
        try:
            db.commit()
        except IntegrityError:  # a concurrent request inserted the same key first: update that row
            db.rollback()
            if attempt:
                raise
            continue
        db.refresh(row)
        return row
    raise AssertionError("unreachable")


def latest_feedback() -> Any:
    """Subquery of feedback rows, at most one verdict and one click per (user, movie): the newest."""
    fb = RecommendationFeedback
    is_click = case((fb.feedback == CLICK, 1), else_=0)
    ranked = select(
        fb.id,
        fb.user_id,
        fb.movie_id,
        fb.recommendation_id,
        fb.feedback,
        fb.created_at,
        func.row_number()
        .over(partition_by=(fb.user_id, fb.movie_id, is_click), order_by=(fb.created_at.desc(), fb.id.desc()))
        .label("rn"),
    ).subquery("ranked_feedback")
    return (
        select(
            ranked.c.id,
            ranked.c.user_id,
            ranked.c.movie_id,
            ranked.c.recommendation_id,
            ranked.c.feedback,
            ranked.c.created_at,
        )
        .where(ranked.c.rn == 1)
        .subquery("latest_feedback")
    )


def feedback_counts() -> Select[Any]:
    """feedback kind -> number of distinct (user, movie) verdicts / clicks."""
    latest = latest_feedback()
    return select(latest.c.feedback, func.count()).group_by(latest.c.feedback)
