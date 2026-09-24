"""The append-only event log: appends, projections, idempotency, bitemporal reads, replay, the refresher.

docs/STREAMING_ARCHITECTURE.md is the design document; the rules in short:

* **Append, then project.** Every member interaction is appended to ``events`` first. The current-state
  tables (ratings, favorites, watch_history, recommendation_feedback) are *projections* of the log,
  written in the same transaction. A projection row is recomputed by folding the key's events in
  ``(event_time, id)`` order (``refold_*``), so an event that arrives late (an older ``event_time``)
  lands where it belongs, and a full rebuild from the log (``fold_log``) equals the live tables.
* **Idempotency.** A request with an ``Idempotency-Key`` stores its first response
  (``idempotency_keys``); a retry with the same key returns it and writes nothing. Events carry the key
  too (unique per ``(source, idempotency_key)``). Without a key, state-setting writes (rate, unrate,
  favorite, feedback) are safe to retry because folding absorbs repeats; a watch is additive, so an
  identical keyless watch within ``events_watch_dedupe_seconds`` is treated as a retry.
* **Bitemporal reads.** Intelligence inputs see an event when ``event_time <= as_of`` (it had happened)
  AND ``ingested_at <= knowledge_time`` (JEV knew about it). Live: both are now. Replay: knowledge_time
  defaults to as_of, so a replay never sees an event that arrived after its date.
* **Near real time.** Every accepted append marks its domain dirty; ``EventRefresher`` runs one debounced
  live intelligence run per dirty domain, through the service's existing run lock.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from jev_api.config import Settings
from jev_api.metrics import metrics
from jev_api.models import (
    DEFAULT_DOMAIN,
    Favorite,
    IntelRun,
    Rating,
    Recommendation,
    RecommendationFeedback,
    User,
    WatchHistory,
    utcnow,
)
from jev_api.models.events import EVENT_SCHEMA_VERSION, Event, EventDailyCount, IdempotencyKey
from jev_ml.core.quality import to_epoch
from jev_ml.domains.generic.adapter import GenericAdapter

log = logging.getLogger(__name__)

APP = "app"  # member writes (UI endpoints and POST /events)
INGEST = "ingest"  # POST /intel/domains/{key}/observations
BACKFILL = "backfill"  # migration 0007: the projection rows that existed before the log

CLICK = "clicked"
NEGATIVE = frozenset({"dislike", "not_interested"})
RATING_TYPES = ("rating", "rating_removed")
FAVORITE_TYPES = ("favorite", "unfavorite")
MEMBER_EVENT_TYPES = ("rating", "rating_removed", "watch", "favorite", "unfavorite", "rec_feedback")


class IdempotencyMismatchError(Exception):
    """The Idempotency-Key was already used for a different request (HTTP 422)."""


def aware(dt: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; they are stored as UTC."""
    if dt is None:
        return None
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def iso(dt: datetime | None) -> str | None:
    return None if dt is None else _utc(dt).isoformat()


def member_key(user_id: int, key: str) -> str:
    """An event's idempotency key for a member-supplied key: scoped to the member."""
    return f"u{user_id}:{key}"


def fingerprint(method: str, path: str, body: Any) -> str:
    raw = json.dumps([method.upper(), path, body], sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


# --- appends ---------------------------------------------------------------------------------------------
def append(
    db: Session,
    *,
    source: str,
    event_type: str,
    entity_id: str,
    event_time: datetime,
    ingested_at: datetime | None = None,
    domain: str = DEFAULT_DOMAIN,
    user_id: int | None = None,
    value: float | None = None,
    payload: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    batch_id: str | None = None,
) -> Event:
    """Append one event (flushed, so its id is assigned). The caller commits."""
    ev = Event(
        event_id=str(uuid.uuid4()),
        source=source,
        domain=domain,
        event_type=event_type,
        user_id=user_id,
        entity_id=entity_id,
        value=value,
        payload=payload or {},
        event_time=_utc(event_time),
        ingested_at=_utc(ingested_at or utcnow()),
        idempotency_key=idempotency_key,
        batch_id=batch_id,
        schema_version=EVENT_SCHEMA_VERSION,
    )
    db.add(ev)
    db.flush()
    metrics.inc("events_ingested", f"{domain}:{event_type}")
    return ev


def count_ingest(
    db: Session, domain: str, day: date, accepted: int = 0, duplicates: int = 0, rejected: int = 0
) -> None:
    """Incremental per-domain/day counters (an upsert in the caller's transaction)."""
    if not (accepted or duplicates or rejected):
        return
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        ins: Any = pg_insert(EventDailyCount)
    else:
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        ins = sqlite_insert(EventDailyCount)
    t = EventDailyCount.__table__
    stmt = ins.values(domain=domain, day=day, accepted=accepted, duplicates=duplicates, rejected=rejected)
    stmt = stmt.on_conflict_do_update(
        index_elements=["domain", "day"],
        set_={
            "accepted": t.c.accepted + stmt.excluded.accepted,
            "duplicates": t.c.duplicates + stmt.excluded.duplicates,
            "rejected": t.c.rejected + stmt.excluded.rejected,
        },
    )
    db.execute(stmt)
    if duplicates:
        metrics.inc("events_duplicates", domain, duplicates)
    if rejected:
        metrics.inc("events_rejected", domain, rejected)


# --- folds (pure: the same code maintains the projections, rebuilds them and reads them as of) -----------
class EventRow(NamedTuple):
    id: int
    event_type: str
    user_id: int | None
    entity_id: str
    value: float | None
    payload: dict[str, Any]
    event_time: datetime


@dataclass(frozen=True)
class RatingState:
    rating: float
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class FavoriteState:
    source: str
    created_at: datetime


@dataclass(frozen=True)
class FeedbackState:
    feedback: str
    created_at: datetime
    first_event: int  # tie-break of the newest-verdict rule (row id order in the live table)


def _order(events: Iterable[EventRow]) -> list[EventRow]:
    return sorted(events, key=lambda e: (_utc(e.event_time), e.id))


def _payload_time(payload: dict[str, Any], name: str) -> datetime | None:
    raw = payload.get(name)
    return _utc(datetime.fromisoformat(raw)) if isinstance(raw, str) else None


def fold_rating(events: Iterable[EventRow]) -> RatingState | None:
    """rating sets the value (updated_at moves only when the value changes, as an in-place UPDATE of an
    unchanged value never did); rating_removed deletes it. A backfill event carries created_at."""
    state: RatingState | None = None
    for e in _order(events):
        t = _utc(e.event_time)
        if e.event_type == "rating_removed":
            state = None
        elif e.event_type == "rating" and e.value is not None:
            if state is None:
                state = RatingState(float(e.value), _payload_time(e.payload, "created_at") or t, t)
            elif state.rating != float(e.value):
                state = RatingState(float(e.value), state.created_at, t)
    return state


def fold_favorite(events: Iterable[EventRow]) -> FavoriteState | None:
    """favorite adds (a repeat keeps the first source and time); unfavorite removes."""
    state: FavoriteState | None = None
    for e in _order(events):
        if e.event_type == "unfavorite":
            state = None
        elif e.event_type == "favorite" and state is None:
            state = FavoriteState(str(e.payload.get("source") or "user"), _utc(e.event_time))
    return state


FeedbackSlot = tuple[int | None, bool]  # (recommendation_id, is_click) within one (user, movie)


def fold_feedback(events: Iterable[EventRow]) -> dict[FeedbackSlot, FeedbackState]:
    """The upsert of services/feedback.py as a fold: one verdict slot and one click slot per
    recommendation (or per movie without one); a changed verdict replaces the old one and moves
    created_at, a repeat changes nothing."""
    slots: dict[FeedbackSlot, FeedbackState] = {}
    for e in _order(events):
        kind = e.payload.get("feedback")
        if e.event_type != "rec_feedback" or not isinstance(kind, str):
            continue
        rid = e.payload.get("recommendation_id")
        slot = (int(rid) if rid is not None else None, kind == CLICK)
        prev = slots.get(slot)
        if prev is None:
            slots[slot] = FeedbackState(kind, _utc(e.event_time), e.id)
        elif prev.feedback != kind:
            slots[slot] = FeedbackState(kind, _utc(e.event_time), prev.first_event)
    return slots


# --- projection maintenance -------------------------------------------------------------------------------
def _row(e: Event) -> EventRow:
    return EventRow(e.id, e.event_type, e.user_id, e.entity_id, e.value, e.payload or {}, e.event_time)


def _key_events(db: Session, user_id: int, movie_id: int, types: Sequence[str]) -> list[EventRow]:
    rows = db.scalars(
        select(Event).where(
            Event.user_id == user_id,
            Event.entity_id == str(movie_id),
            Event.event_type.in_(types),
            Event.domain == DEFAULT_DOMAIN,
        )
    ).all()
    return [_row(e) for e in rows]


def refold_rating(db: Session, user_id: int, movie_id: int) -> Rating | None:
    state = fold_rating(_key_events(db, user_id, movie_id, RATING_TYPES))
    row = db.scalar(select(Rating).where(Rating.user_id == user_id, Rating.movie_id == movie_id))
    if state is None:
        if row is not None:
            db.delete(row)
        return None
    if row is None:
        row = Rating(user_id=user_id, movie_id=movie_id)
        db.add(row)
    changed = False
    if row.rating != state.rating:
        row.rating, changed = state.rating, True
    if aware(row.created_at) != state.created_at:
        row.created_at, changed = state.created_at, True
    if changed or aware(row.updated_at) != state.updated_at:
        # always in the UPDATE's SET clause, else TimestampMixin's onupdate stamps the wall clock
        row.updated_at = state.updated_at
        flag_modified(row, "updated_at")
    return row


def refold_favorite(db: Session, user_id: int, movie_id: int) -> Favorite | None:
    state = fold_favorite(_key_events(db, user_id, movie_id, FAVORITE_TYPES))
    row = db.scalar(select(Favorite).where(Favorite.user_id == user_id, Favorite.movie_id == movie_id))
    if state is None:
        if row is not None:
            db.delete(row)
        return None
    if row is None:
        row = Favorite(user_id=user_id, movie_id=movie_id, source=state.source, created_at=state.created_at)
        db.add(row)
    else:
        if row.source != state.source:
            row.source = state.source
        if aware(row.created_at) != state.created_at:
            row.created_at = state.created_at
    return row


def project_watch(db: Session, ev: Event) -> WatchHistory:
    """A watch is additive: one projection row per event (order-independent)."""
    row = WatchHistory(user_id=ev.user_id, movie_id=int(ev.entity_id), watched_at=ev.event_time)
    db.add(row)
    return row


def refold_feedback(db: Session, user_id: int, movie_id: int) -> dict[FeedbackSlot, RecommendationFeedback]:
    """Recompute every feedback slot of (user, movie) from the log; rows are updated in place."""
    slots = fold_feedback(_key_events(db, user_id, movie_id, ("rec_feedback",)))
    fb = RecommendationFeedback
    existing: dict[FeedbackSlot, RecommendationFeedback] = {}
    for r in db.scalars(
        select(fb).where(fb.user_id == user_id, fb.movie_id == movie_id).order_by(fb.created_at, fb.id)
    ):
        existing.setdefault((r.recommendation_id, r.feedback == CLICK), r)
    out: dict[FeedbackSlot, RecommendationFeedback] = {}
    for slot, state in sorted(slots.items(), key=lambda kv: kv[1].first_event):
        row = existing.get(slot)
        if row is None:
            row = fb(
                user_id=user_id,
                movie_id=movie_id,
                recommendation_id=slot[0],
                feedback=state.feedback,
                created_at=state.created_at,
            )
            db.add(row)
        else:
            if row.feedback != state.feedback:
                row.feedback = state.feedback
            if aware(row.created_at) != state.created_at:
                row.created_at = state.created_at
        out[slot] = row
    db.flush()
    return out


# --- member interactions (the movie write endpoints and POST /events share this) -------------------------
@dataclass
class Applied:
    event: Event
    profile_changed: bool  # the member's recommendations are stale (the caller bumps profile_version)


def apply_member_event(
    db: Session,
    user: User,
    event_type: str,
    movie_id: int,
    *,
    value: float | None = None,
    feedback: str | None = None,
    recommendation_id: int | None = None,
    event_time: datetime | None = None,
    key: str | None = None,
    batch_id: str | None = None,
    now: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> Applied:
    """Append one member event and update its projection in the same transaction (no commit)."""
    now = now or utcnow()
    payload: dict[str, Any] = dict(extra or {})
    if event_type == "rec_feedback":
        payload.update({"feedback": feedback, "recommendation_id": recommendation_id})
    elif event_type == "favorite":
        payload.setdefault("source", "user")
    before_negative = False
    if event_type == "rec_feedback":
        before_negative = _slot_negative(db, user.id, movie_id, recommendation_id, feedback)
    ev = append(
        db,
        source=APP,
        event_type=event_type,
        entity_id=str(movie_id),
        user_id=user.id,
        value=value,
        payload=payload,
        event_time=event_time or now,
        ingested_at=now,
        idempotency_key=member_key(user.id, key) if key else None,
        batch_id=batch_id,
    )
    changed = True
    if event_type in RATING_TYPES:
        refold_rating(db, user.id, movie_id)
    elif event_type in FAVORITE_TYPES:
        refold_favorite(db, user.id, movie_id)
    elif event_type == "watch":
        project_watch(db, ev)
    elif event_type == "rec_feedback":
        refold_feedback(db, user.id, movie_id)
        after_negative = _slot_negative(db, user.id, movie_id, recommendation_id, feedback)
        changed = before_negative != after_negative  # only the exclusion list matters to recommendations
    count_ingest(db, DEFAULT_DOMAIN, _utc(now).date(), accepted=1)
    return Applied(ev, changed)


def _slot_negative(
    db: Session, user_id: int, movie_id: int, recommendation_id: int | None, kind: str | None
) -> bool:
    """Is the verdict slot that `kind` writes to currently negative?"""
    if kind == CLICK:
        return False
    fb = RecommendationFeedback
    q = select(fb.feedback).where(fb.user_id == user_id, fb.feedback != CLICK)
    if recommendation_id is not None:
        q = q.where(fb.recommendation_id == recommendation_id)
    else:
        q = q.where(fb.recommendation_id.is_(None), fb.movie_id == movie_id)
    current = db.scalars(q.order_by(fb.created_at.desc(), fb.id.desc())).first()
    return current in NEGATIVE


def recent_watch(db: Session, user_id: int, movie_id: int, window_s: float, now: datetime) -> bool:
    """The keyless watch retry rule: an identical watch appended within the window."""
    if window_s <= 0:
        return False
    since = _utc(now) - timedelta(seconds=window_s)
    hit = db.scalar(
        select(Event.id)
        .where(
            Event.user_id == user_id,
            Event.entity_id == str(movie_id),
            Event.event_type == "watch",
            Event.domain == DEFAULT_DOMAIN,
            Event.source == APP,
            Event.ingested_at >= since,
        )
        .limit(1)
    )
    return hit is not None


# --- request-level idempotency (Idempotency-Key header) ---------------------------------------------------
def user_scope(user: User) -> str:
    return f"user:{user.id}"


def stored_response(db: Session, scope: str, key: str, fp: str) -> dict[str, Any] | None:
    row = db.scalar(select(IdempotencyKey).where(IdempotencyKey.scope == scope, IdempotencyKey.key == key))
    if row is None:
        return None
    if row.fingerprint != fp:
        raise IdempotencyMismatchError("this Idempotency-Key was already used for a different request")
    return dict(row.response)


def idempotent(
    db: Session,
    scope: str,
    key: str | None,
    fp: str,
    work: Callable[[], dict[str, Any]],
    domain: str = DEFAULT_DOMAIN,
    status_code: int = 200,
) -> tuple[dict[str, Any], bool]:
    """Run `work` (appends and projections, no commit) and commit it together with the stored response.
    Returns (response, replayed). A retry with the same key returns the first response and writes
    nothing; a concurrent duplicate that loses the race is answered the same way."""
    if key is not None:
        hit = stored_response(db, scope, key, fp)
        if hit is not None:
            _count_duplicate(db, domain)
            return hit, True
    for attempt in range(2):
        try:
            response = work()
            if key is not None:
                db.add(
                    IdempotencyKey(
                        scope=scope,
                        key=key,
                        fingerprint=fp,
                        status_code=status_code,
                        response=json.loads(json.dumps(response, default=str)),
                        created_at=utcnow(),
                    )
                )
            db.commit()
            return response, False
        except IntegrityError:
            db.rollback()
            if key is not None:
                hit = stored_response(db, scope, key, fp)
                if hit is not None:
                    _count_duplicate(db, domain)
                    return hit, True
            if attempt:
                raise
    raise AssertionError("unreachable")


def _count_duplicate(db: Session, domain: str) -> None:
    count_ingest(db, domain, utcnow().date(), duplicates=1)
    db.commit()


# --- rebuild and verification -------------------------------------------------------------------------------
@dataclass
class Projections:
    ratings: dict[tuple[int, int], RatingState] = field(default_factory=dict)
    favorites: dict[tuple[int, int], FavoriteState] = field(default_factory=dict)
    watches: Counter[tuple[int, int, datetime]] = field(default_factory=Counter)
    # (user, movie, recommendation_id, is_click) -> state
    feedback: dict[tuple[int, int, int | None, bool], FeedbackState] = field(default_factory=dict)


def _window(q: Any, domain: str, as_of: datetime | None, knowledge: datetime | None, upto: int | None) -> Any:
    q = q.where(Event.domain == domain)
    if as_of is not None:
        q = q.where(Event.event_time <= _utc(as_of))
    if knowledge is not None:
        q = q.where(Event.ingested_at <= _utc(knowledge))
    if upto is not None:
        q = q.where(Event.id <= upto)
    return q


def read_events(
    db: Session,
    domain: str = DEFAULT_DOMAIN,
    types: Sequence[str] | None = None,
    as_of: datetime | None = None,
    knowledge: datetime | None = None,
    upto: int | None = None,
) -> list[EventRow]:
    q = select(
        Event.id,
        Event.event_type,
        Event.user_id,
        Event.entity_id,
        Event.value,
        Event.payload,
        Event.event_time,
    )
    if types is not None:
        q = q.where(Event.event_type.in_(types))
    rows = db.execute(_window(q, domain, as_of, knowledge, upto).order_by(Event.id)).all()
    return [EventRow(r[0], r[1], r[2], r[3], r[4], r[5] or {}, r[6]) for r in rows]


def fold_log(
    db: Session, as_of: datetime | None = None, knowledge: datetime | None = None, upto: int | None = None
) -> Projections:
    """Every movie-domain projection, folded from the log (optionally as of a bitemporal cut)."""
    groups: dict[tuple[str, int, str], list[EventRow]] = {}
    out = Projections()
    for e in read_events(db, DEFAULT_DOMAIN, MEMBER_EVENT_TYPES, as_of, knowledge, upto):
        if e.user_id is None:
            continue
        if e.event_type == "watch":
            out.watches[(e.user_id, int(e.entity_id), _utc(e.event_time))] += 1
            continue
        family = (
            "rating"
            if e.event_type in RATING_TYPES
            else "favorite"
            if e.event_type in FAVORITE_TYPES
            else "fb"
        )
        groups.setdefault((family, e.user_id, e.entity_id), []).append(e)
    for (family, uid, ent), evs in groups.items():
        mid = int(ent)
        if family == "rating":
            r = fold_rating(evs)
            if r is not None:
                out.ratings[(uid, mid)] = r
        elif family == "favorite":
            f = fold_favorite(evs)
            if f is not None:
                out.favorites[(uid, mid)] = f
        else:
            for (rid, click), s in fold_feedback(evs).items():
                out.feedback[(uid, mid, rid, click)] = s
    return out


def live_projections(db: Session) -> Projections:
    out = Projections()
    for uid, mid, rating, c, u in db.execute(
        select(Rating.user_id, Rating.movie_id, Rating.rating, Rating.created_at, Rating.updated_at)
    ):
        out.ratings[(uid, mid)] = RatingState(float(rating), _utc(c), _utc(u))
    for uid, mid, src, c in db.execute(
        select(Favorite.user_id, Favorite.movie_id, Favorite.source, Favorite.created_at)
    ):
        out.favorites[(uid, mid)] = FavoriteState(src, _utc(c))
    for uid, mid, t in db.execute(
        select(WatchHistory.user_id, WatchHistory.movie_id, WatchHistory.watched_at)
    ):
        out.watches[(uid, mid, _utc(t))] += 1
    fb = RecommendationFeedback
    for rid_, uid, mid, rid, kind, c in db.execute(
        select(fb.id, fb.user_id, fb.movie_id, fb.recommendation_id, fb.feedback, fb.created_at)
    ):
        out.feedback[(uid, mid, rid, kind == CLICK)] = FeedbackState(kind, _utc(c), rid_)
    return out


def _dict_diff(
    folded: dict[Any, Any], live: dict[Any, Any], cmp: Callable[[Any, Any], bool]
) -> dict[str, Any]:
    missing = [k for k in folded if k not in live]
    extra = [k for k in live if k not in folded]
    changed = [k for k in folded if k in live and not cmp(folded[k], live[k])]
    return {
        "missing": len(missing),
        "extra": len(extra),
        "changed": len(changed),
        "sample": [str(k) for k in (missing + extra + changed)[:5]],
    }


def diff_projections(folded: Projections, live: Projections) -> dict[str, Any]:
    """Per projection: rows the log implies but the table lacks (missing), rows without events (extra)
    and rows whose values differ (changed). All zero = the log reproduces the tables exactly."""
    fb_cmp: Callable[[Any, Any], bool] = lambda a, b: (a.feedback, a.created_at) == (b.feedback, b.created_at)  # noqa: E731
    watch_missing = folded.watches - live.watches
    watch_extra = live.watches - folded.watches
    out: dict[str, Any] = {
        "ratings": _dict_diff(folded.ratings, live.ratings, lambda a, b: a == b),
        "favorites": _dict_diff(folded.favorites, live.favorites, lambda a, b: a == b),
        "recommendation_feedback": _dict_diff(folded.feedback, live.feedback, fb_cmp),
        "watch_history": {
            "missing": sum(watch_missing.values()),
            "extra": sum(watch_extra.values()),
            "changed": 0,
            "sample": [str(k) for k in list(watch_missing)[:3] + list(watch_extra)[:2]],
        },
    }
    out["consistent"] = all(v["missing"] == v["extra"] == v["changed"] == 0 for v in out.values())
    return out


def verify_projections(db: Session) -> dict[str, Any]:
    return diff_projections(fold_log(db), live_projections(db))


def rebuild_projections(db: Session) -> dict[str, Any]:
    """Rewrite every movie-domain projection from the log (admin replay; the caller commits). Rows are
    updated in place by key, so ids survive for unchanged rows. Returns the diff found before."""
    folded, live = fold_log(db), live_projections(db)
    before = diff_projections(folded, live)
    if before["consistent"]:
        return before
    keys = {(u, m) for (u, m) in folded.ratings} | {(u, m) for (u, m) in live.ratings}
    for u, m in keys:
        refold_rating(db, u, m)
    keys = {(u, m) for (u, m) in folded.favorites} | {(u, m) for (u, m) in live.favorites}
    for u, m in keys:
        refold_favorite(db, u, m)
    for u, m in {(k[0], k[1]) for k in folded.feedback} | {(k[0], k[1]) for k in live.feedback}:
        refold_feedback(db, u, m)
        # feedback rows without any event (written outside the log) are removed
        slots = {(k[2], k[3]) for k in folded.feedback if k[0] == u and k[1] == m}
        for row in db.scalars(
            select(RecommendationFeedback).where(
                RecommendationFeedback.user_id == u, RecommendationFeedback.movie_id == m
            )
        ):
            if (row.recommendation_id, row.feedback == CLICK) not in slots:
                db.delete(row)
    for (u, m, t), n in (live.watches - folded.watches).items():
        rows = db.scalars(
            select(WatchHistory)
            .where(WatchHistory.user_id == u, WatchHistory.movie_id == m)
            .order_by(WatchHistory.id.desc())
        ).all()
        for extra_row in [r for r in rows if _utc(r.watched_at) == t][:n]:
            db.delete(extra_row)
    for (u, m, t), n in (folded.watches - live.watches).items():
        for _ in range(n):
            db.add(WatchHistory(user_id=u, movie_id=m, watched_at=t))
    db.flush()
    return before


# --- bitemporal reads for the intelligence layer ------------------------------------------------------------
def cut(
    as_of: datetime | None, now: datetime, knowledge_time: datetime | None = None
) -> tuple[datetime, datetime]:
    """(valid-time cut, knowledge-time cut). Live: (now, now). Replay: (as_of, knowledge_time or as_of)."""
    valid = _utc(as_of) if as_of is not None else _utc(now)
    knowledge = (
        _utc(knowledge_time) if knowledge_time is not None else valid if as_of is not None else _utc(now)
    )
    return valid, knowledge


def watermark(
    db: Session, domain: str, as_of: datetime | None, now: datetime, knowledge_time: datetime | None = None
) -> dict[str, Any]:
    """The newest event visible to a run of `domain` at this cut (recorded on the run for lineage)."""
    valid, knowledge = cut(as_of, now, knowledge_time)
    q = select(
        func.max(Event.id), func.max(Event.ingested_at), func.max(Event.event_time), func.count(Event.id)
    )
    max_id, max_ing, max_evt, n = db.execute(_window(q, domain, valid, knowledge, None)).one()
    return {
        "domain": domain,
        "as_of": iso(valid),
        "knowledge_time": iso(knowledge),
        "max_event_id": max_id,
        "max_ingested_at": iso(aware(max_ing)),
        "max_event_time": iso(aware(max_evt)),
        "n_events": int(n or 0),
    }


def app_frames(
    db: Session, as_of: datetime | None, now: datetime, knowledge_time: datetime | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """(app_ratings, app_feedback, app_served, watermark) for a movie run, read bitemporally from the log.
    Live runs reproduce the projections; a replay folds only what had happened and was known by then."""
    wm = watermark(db, DEFAULT_DOMAIN, as_of, now, knowledge_time)
    valid, knowledge = cut(as_of, now, knowledge_time)
    upto = wm["max_event_id"] or 0
    folded = fold_log(db, valid, knowledge, upto)
    ratings = pd.DataFrame(
        [(u, m, s.rating, s.updated_at) for (u, m), s in sorted(folded.ratings.items())],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    )
    # one verdict (and one click) per member per movie: the newest (services/feedback.latest_feedback)
    latest: dict[tuple[int, int, bool], FeedbackState] = {}
    for (u, m, _rid, click), s in folded.feedback.items():
        k = (u, m, click)
        cur = latest.get(k)
        if cur is None or (s.created_at, s.first_event) > (cur.created_at, cur.first_event):
            latest[k] = s
    feedback = pd.DataFrame(
        [(s.feedback, s.created_at) for _, s in sorted(latest.items())], columns=["feedback", "timestamp"]
    )
    q = select(Recommendation.movie_id, Recommendation.created_at)
    if as_of is not None:  # a replay sees only what had been served by then
        q = q.where(Recommendation.created_at <= valid)
    served = pd.DataFrame(db.execute(q).all(), columns=["movie_id", "timestamp"])
    return ratings, feedback, served, wm


# --- generic-domain observations ----------------------------------------------------------------------------
def observation_key(domain: str, key: str) -> str:
    return f"{domain}:{key}"


def observation_frame(
    db: Session, domain: str, as_of: datetime | None, now: datetime, knowledge_time: datetime | None = None
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Pushed observations visible at the cut, newest per (entity, event_time) (a re-sent value for the
    same period is a revision), plus the watermark."""
    wm = watermark(db, domain, as_of, now, knowledge_time)
    valid, knowledge = cut(as_of, now, knowledge_time)
    rows = read_events(db, domain, ("observation",), valid, knowledge, wm["max_event_id"] or 0)
    latest: dict[tuple[str, datetime], EventRow] = {}
    for e in rows:  # id order: a later ingest wins
        latest[(e.entity_id, _utc(e.event_time))] = e
    frame = pd.DataFrame(
        [
            {
                "event": e.id,
                "entity": e.entity_id,
                "event_time": _utc(e.event_time),
                "value": e.value,
                "attributes": e.payload.get("attributes") or {},
            }
            for e in sorted(latest.values(), key=lambda r: r.id)
        ],
        columns=["event", "entity", "event_time", "value", "attributes"],
    )
    return frame, wm


class ObservedAdapter(GenericAdapter):
    """A generic adapter over the configured file plus pushed observations (the ``frame=`` hook).

    Pushed rows follow the file's row template of their entity (entity type, groups), replace a file row
    for the same (period, entity), and are *published when ingested*: the availability lag models a
    release delay for file data, whereas a pushed row's knowledge time is its ``ingested_at``, already
    applied by the bitemporal read. Relies on GenericAdapter's ``_raw`` / ``_lag_days`` / ``_provenance``
    (tests/integration/test_events_ingest.py guards that contract)."""

    def __init__(
        self, base: GenericAdapter, frame: pd.DataFrame, n_file: int, n_pushed: int, max_event: int | None
    ) -> None:
        super().__init__(base.cfg, frame=frame)
        self._base = base
        self._n_file = n_file
        self._n_pushed = n_pushed
        self._max_event = max_event

    def _lag_days(self, entity: pd.Series, entity_type: pd.Series) -> np.ndarray:
        lag = np.array(super()._lag_days(entity, entity_type), dtype=float, copy=True)
        lag[self._n_file :] = 0.0
        return lag

    def _provenance(self) -> dict[str, Any]:
        base = self._base._provenance()
        pushed = self.frame.iloc[self._n_file :] if self.frame is not None else pd.DataFrame()
        h = hashlib.sha256(str(base.get("checksum")).encode())
        h.update(pd.util.hash_pandas_object(pushed.astype(str), index=False).to_numpy().tobytes())
        detail = base.get("detail") or self.cfg.source.get("file")
        return {
            "checksum": f"sha256:{h.hexdigest()}",
            "detail": f"{detail}; + {self._n_pushed} pushed observations (events up to id {self._max_event})",
        }


def domain_entities(adapter: GenericAdapter) -> dict[str, dict[Any, Any]]:
    """entity name -> its newest file row (the template of pushed rows)."""
    raw = adapter._raw()
    c = adapter.cfg.columns
    order = to_epoch(raw[c["timestamp"]]).to_numpy()
    newest = raw.iloc[np.argsort(order, kind="stable")].groupby(raw[c["entity"]].astype(str)).tail(1)
    return {str(row[c["entity"]]): row.to_dict() for _, row in newest.iterrows()}


def adapter_with_observations(
    base: GenericAdapter, obs: pd.DataFrame, max_event: int | None
) -> GenericAdapter:
    if not len(obs):
        return base
    c = base.cfg.columns
    raw = base._raw().reset_index(drop=True)
    ts_col, ent_col, val_col = c["timestamp"], c["entity"], c["value"]
    templates = domain_entities(base)
    numeric_ts = pd.api.types.is_numeric_dtype(raw[ts_col])
    pushed: list[dict[str, Any]] = []
    records: list[dict[Any, Any]] = obs.to_dict("records")
    for r in records:
        tpl = templates.get(str(r["entity"]))
        if tpl is None:  # validated at ingest; the dataset may have changed since
            continue
        row = dict(tpl)
        t = _utc(r["event_time"])
        row[ts_col] = (
            t.timestamp()
            if numeric_ts
            else (t.strftime("%Y-%m-%d") if t.time() == datetime.min.time() else t.isoformat())
        )
        row[val_col] = float(r["value"]) if r["value"] is not None else math.nan
        for k, v in (r["attributes"] or {}).items():
            row[k] = v
        pushed.append(row)
    if not pushed:
        return base
    new = pd.DataFrame(pushed, columns=raw.columns)
    file_ts = (
        pd.to_datetime(raw[ts_col], utc=True, errors="coerce")
        if not numeric_ts
        else pd.to_datetime(raw[ts_col], unit="s", utc=True)
    )
    new_ts = (
        pd.to_datetime(new[ts_col], utc=True, errors="coerce")
        if not numeric_ts
        else pd.to_datetime(new[ts_col], unit="s", utc=True)
    )
    replaced = set(zip(new[ent_col].astype(str), new_ts, strict=True))
    keep = [(e, t) not in replaced for e, t in zip(raw[ent_col].astype(str), file_ts, strict=True)]
    kept = raw.loc[keep]
    frame = pd.concat([kept, new], ignore_index=True)
    return ObservedAdapter(base, frame, n_file=len(kept), n_pushed=len(new), max_event=max_event)


# --- monitoring ---------------------------------------------------------------------------------------------
def health(
    db: Session, now: datetime | None = None, refresher: EventRefresher | None = None
) -> dict[str, Any]:
    now = _utc(now or utcnow())
    today = now.date()
    week = today - timedelta(days=6)
    per: dict[str, dict[str, Any]] = {}
    for domain, n, max_id, max_evt, max_ing in db.execute(
        select(
            Event.domain,
            func.count(Event.id),
            func.max(Event.id),
            func.max(Event.event_time),
            func.max(Event.ingested_at),
        ).group_by(Event.domain)
    ):
        evt, ing = aware(max_evt), aware(max_ing)
        per[domain] = {
            "domain": domain,
            "events": int(n),
            "max_event_id": max_id,
            "max_event_time": iso(evt),
            "max_ingested_at": iso(ing),
            "event_time_lag_s": round((now - evt).total_seconds(), 1) if evt else None,
            "ingest_lag_s": round((now - ing).total_seconds(), 1) if ing else None,
        }
    for domain, day, acc, dup, rej in db.execute(
        select(
            EventDailyCount.domain,
            EventDailyCount.day,
            EventDailyCount.accepted,
            EventDailyCount.duplicates,
            EventDailyCount.rejected,
        ).where(EventDailyCount.day >= week)
    ):
        d = per.setdefault(domain, {"domain": domain, "events": 0, "max_event_id": None})
        for window, include in (("today", day == today), ("last_7_days", True)):
            if include:
                agg = d.setdefault(window, {"accepted": 0, "duplicates": 0, "rejected": 0})
                agg["accepted"] += acc
                agg["duplicates"] += dup
                agg["rejected"] += rej
    for domain, d in per.items():
        for window in ("today", "last_7_days"):
            d.setdefault(window, {"accepted": 0, "duplicates": 0, "rejected": 0})
        run = db.scalar(
            select(IntelRun)
            .where(IntelRun.domain == domain, IntelRun.mode == "live", IntelRun.status == "succeeded")
            .order_by(IntelRun.started_at.desc(), IntelRun.id.desc())
            .limit(1)
        )
        wm = run.event_watermark if run is not None else None
        seen = (wm or {}).get("max_event_id") or 0
        d["last_live_run"] = (
            None
            if run is None
            else {"run_id": run.run_id, "trigger": run.trigger, "finished_at": iso(aware(run.finished_at))}
        )
        d["watermark"] = wm
        d["events_since_last_run"] = int(
            db.scalar(select(func.count(Event.id)).where(Event.domain == domain, Event.id > seen)) or 0
        )
        if refresher is not None:
            d["refresh"] = refresher.state(domain)
    return {
        "generated_at": iso(now),
        "domains": sorted(per.values(), key=lambda d: d["domain"]),
        "refresher": refresher.summary() if refresher is not None else {"enabled": False},
    }


# --- the debounced near-real-time refresher -----------------------------------------------------------------
def refresh_enabled(settings: Settings) -> bool:
    """events_refresh_enabled, or auto (None): on, except under pytest or JEV_ENV=test."""
    if settings.events_refresh_enabled is not None:
        return settings.events_refresh_enabled
    return settings.env != "test" and "PYTEST_CURRENT_TEST" not in os.environ


class EventRefresher:
    """Trailing-edge debounce per domain: a domain marked dirty is refreshed once it has been quiet for
    `debounce` seconds, or `max_delay` seconds after it first became dirty under steady traffic.

    `run(domain)` performs one live intelligence run and may raise `busy` (another run holds the service
    lock): the domain then stays dirty and is retried on the next tick. One worker thread per process;
    `tick()` is also callable directly (tests, scripts)."""

    def __init__(
        self,
        run: Callable[[str], Any],
        debounce: float,
        max_delay: float,
        enabled: bool = True,
        busy: tuple[type[BaseException], ...] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._run = run
        self.debounce = float(debounce)
        self.max_delay = max(float(max_delay), float(debounce))
        self.enabled = enabled
        self._busy = busy
        self._clock = clock
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._first: dict[str, float] = {}
        self._last: dict[str, float] = {}
        self._done: dict[str, dict[str, Any]] = {}

    def mark_dirty(self, domain: str) -> None:
        now = self._clock()
        with self._lock:
            self._first.setdefault(domain, now)
            self._last[domain] = now
            start = self.enabled and self._thread is None
            if start:
                self._thread = threading.Thread(target=self._loop, name="events-refresher", daemon=True)
        metrics.inc("events_refresh", "marked_dirty")
        if start and self._thread is not None:
            self._thread.start()
        self._wake.set()

    def due(self, now: float | None = None) -> list[str]:
        now = self._clock() if now is None else now
        with self._lock:
            return sorted(
                d
                for d, last in self._last.items()
                if now - last >= self.debounce or now - self._first[d] >= self.max_delay
            )

    def next_due_in(self, now: float | None = None) -> float | None:
        now = self._clock() if now is None else now
        with self._lock:
            waits = [
                min(self.debounce - (now - last), self.max_delay - (now - self._first[d]))
                for d, last in self._last.items()
            ]
        return max(0.0, min(waits)) if waits else None

    def tick(self, now: float | None = None) -> list[str]:
        """Refresh every due domain; returns those that ran (successfully or not)."""
        ran: list[str] = []
        for domain in self.due(now):
            with self._lock:
                first = self._first.pop(domain, None)
                self._last.pop(domain, None)
            try:
                result = self._run(domain)
            except self._busy:
                with self._lock:  # still dirty: retried after another debounce period
                    self._first.setdefault(domain, first if first is not None else self._clock())
                    self._last.setdefault(domain, self._clock())
                metrics.inc("events_refresh", "busy")
                continue
            except Exception as exc:
                log.exception(
                    "event-driven intelligence refresh failed", extra={"extra_fields": {"domain": domain}}
                )
                self._done[domain] = {"at": iso(utcnow()), "status": "error", "error": type(exc).__name__}
                metrics.inc("events_refresh", "errors")
                ran.append(domain)
                continue
            status = getattr(result, "status", None)
            self._done[domain] = {
                "at": iso(utcnow()),
                "status": status,
                "run_id": getattr(result, "run_id", None),
            }
            metrics.inc("events_refresh", f"runs_{status or 'unknown'}")
            metrics.set("events_refresh", "last_domain", domain)
            ran.append(domain)
        return ran

    def _loop(self) -> None:
        while not self._stop.is_set():
            wait = self.next_due_in()
            self._wake.wait(timeout=wait if wait is not None else 60.0)
            self._wake.clear()
            if self._stop.is_set():
                break
            try:
                self.tick()
            except Exception:  # never let the worker die
                log.exception("events refresher tick failed")

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def state(self, domain: str) -> dict[str, Any]:
        with self._lock:
            dirty = domain in self._last
        return {"dirty": dirty, "last_refresh": self._done.get(domain)}

    def summary(self) -> dict[str, Any]:
        with self._lock:
            dirty = sorted(self._last)
        return {
            "enabled": self.enabled,
            "debounce_seconds": self.debounce,
            "max_delay_seconds": self.max_delay,
            "dirty_domains": dirty,
            "running": self._thread is not None and self._thread.is_alive(),
        }


_refresher_lock = threading.Lock()


def refresher_for(app: Any) -> EventRefresher:
    """The app's refresher, created on first use (main.py is frozen; nothing starts it at boot: the
    worker thread starts on the first dirty mark)."""
    with _refresher_lock:
        existing = getattr(app.state, "event_refresher", None)
        if isinstance(existing, EventRefresher):
            return existing
        from jev_api.services.intel import IntelBusyError

        settings: Settings = app.state.settings
        intel = app.state.intel
        r = EventRefresher(
            run=lambda domain: intel.run("schedule", domain=domain),
            debounce=settings.events_refresh_debounce_seconds,
            max_delay=settings.events_refresh_max_delay_seconds,
            enabled=refresh_enabled(settings),
            busy=(IntelBusyError,),
        )
        app.state.event_refresher = r
        return r
