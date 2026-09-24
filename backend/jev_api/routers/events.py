"""Event ingestion, replay and monitoring (WS1, docs/STREAMING_ARCHITECTURE.md).

- ``POST /events``: a member posts their own interaction events (batch, idempotent per item and, with an
  ``Idempotency-Key`` header, per request).
- ``POST /intel/domains/{key}/observations``: observations for a generic domain, validated against its
  config. Admin for now; the dependency is pluggable (``require_observation_writer``) so service tokens
  can be accepted later without touching the handler.
- ``POST /events/replay`` (admin): verify, or rebuild, the projections from the log.
- ``GET /events/health`` (admin): counts, duplicates, rejections, lag, watermark and refresh state.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from jev_api.deps import DB, AdminUser, CurrentUser, OptionalUser, ServiceTokenDep, Writer, writer_for
from jev_api.models import DEFAULT_DOMAIN, Movie, Recommendation, User, utcnow
from jev_api.models.events import Event
from jev_api.schemas.events import (
    IDEMPOTENCY_KEY_PATTERN,
    EventBatchIn,
    EventHealthOut,
    IngestResultOut,
    MemberEventIn,
    ObservationBatchIn,
    ObservationIn,
    ReplayIn,
    ReplayOut,
)
from jev_api.services import audit, events
from jev_api.services.intel import DOMAIN_PATTERN, DomainUnavailableError, UnknownDomainError

router = APIRouter(tags=["events"])

IdempotencyKeyHeader = Annotated[
    str | None,
    Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=200,
        pattern=IDEMPOTENCY_KEY_PATTERN,
        description="optional: a retry with the same key returns the first response and writes nothing",
    ),
]


def require_observation_writer(request: Request, user: OptionalUser, token: ServiceTokenDep) -> Writer:
    """Who may push observations: a service token with scope ``events:ingest:<key>`` (or
    ``events:ingest:*``), or an admin (docs/SECURITY_AUDIT_PHASE2.md). Members get 403. The key is read
    raw so that authentication comes first; the handler validates it (an invalid key matches no scope)."""
    key = str(request.path_params.get("key", ""))
    return writer_for(f"events:ingest:{key}", user, token)


ObservationWriter = Annotated[Writer, Depends(require_observation_writer)]


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _item(index: int, status_: str, ev: Event | None = None, reason: str | None = None) -> dict[str, Any]:
    return {
        "index": index,
        "status": status_,
        "event_id": ev.event_id if ev is not None else None,
        "seq": ev.id if ev is not None else None,
        "reason": reason,
    }


def _existing(db: Session, source: str, keys: list[str]) -> dict[str, Event]:
    out: dict[str, Event] = {}
    for i in range(0, len(keys), 500):
        chunk = keys[i : i + 500]
        for ev in db.scalars(select(Event).where(Event.source == source, Event.idempotency_key.in_(chunk))):
            if ev.idempotency_key is not None:
                out[ev.idempotency_key] = ev
    return out


def _time_problem(t: datetime, now: datetime, skew_s: float, max_age_days: float | None) -> str | None:
    if t > now + timedelta(seconds=skew_s):
        return "event_time is in the future"
    if max_age_days is not None and t < now - timedelta(days=max_age_days):
        return f"event_time is more than {max_age_days:g} days old"
    return None


# --- POST /events -------------------------------------------------------------------------------------------
def _same_member_event(ev: Event, e: MemberEventIn) -> bool:
    payload = ev.payload or {}
    same = (
        ev.event_type == e.event_type
        and ev.entity_id == str(e.movie_id)
        and ev.value == e.value
        and payload.get("feedback") == e.feedback
        and payload.get("recommendation_id") == e.recommendation_id
    )
    if e.event_time is not None:
        same = same and _utc(ev.event_time) == _utc(e.event_time)
    return same


def _member_batch(db: Session, user: User, items: list[MemberEventIn], request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    now = utcnow()
    batch_id = str(uuid.uuid4())
    keys = [events.member_key(user.id, e.idempotency_key) for e in items]
    existing = _existing(db, events.APP, keys)
    movie_ids = sorted({e.movie_id for e in items})
    movies = set(db.scalars(select(Movie.id).where(Movie.id.in_(movie_ids))))
    rec_ids = sorted({e.recommendation_id for e in items if e.recommendation_id is not None})
    recs = {r.id: r for r in db.scalars(select(Recommendation).where(Recommendation.id.in_(rec_ids)))}
    out: list[dict[str, Any]] = []
    stale = False
    for i, e in enumerate(items):
        prior = existing.get(keys[i])
        if prior is not None:
            if _same_member_event(prior, e):
                out.append(_item(i, "duplicate", prior))
            else:
                out.append(_item(i, "rejected", reason="idempotency_key already used for a different event"))
            continue
        t = _utc(e.event_time) if e.event_time is not None else now
        reason = _time_problem(t, now, settings.events_max_future_skew_seconds, settings.events_max_age_days)
        if e.movie_id not in movies:
            reason = "movie not found"
        elif e.recommendation_id is not None:
            rec = recs.get(e.recommendation_id)
            if rec is None or rec.user_id != user.id or rec.movie_id != e.movie_id:
                reason = "recommendation not found"
        if reason is not None:
            out.append(_item(i, "rejected", reason=reason))
            continue
        applied = events.apply_member_event(
            db,
            user,
            e.event_type,
            e.movie_id,
            value=e.value,
            feedback=e.feedback,
            recommendation_id=e.recommendation_id,
            event_time=t,
            key=e.idempotency_key,
            batch_id=batch_id,
            now=now,
        )
        existing[keys[i]] = applied.event  # a repeat inside the same batch is a duplicate
        stale = stale or applied.profile_changed
        out.append(_item(i, "accepted", applied.event))
    if stale:
        user.profile_version += 1  # cached recommendations are stale
    return _summary(db, batch_id, DEFAULT_DOMAIN, out, now)


def _summary(
    db: Session, batch_id: str, domain: str, items: list[dict[str, Any]], now: datetime
) -> dict[str, Any]:
    n = {s: sum(1 for it in items if it["status"] == s) for s in ("accepted", "duplicate", "rejected")}
    events.count_ingest(db, domain, now.date(), duplicates=n["duplicate"], rejected=n["rejected"])
    return {
        "batch_id": batch_id,
        "domain": domain,
        "accepted": n["accepted"],
        "duplicates": n["duplicate"],
        "rejected": n["rejected"],
        "items": items,
    }


@router.post("/events", response_model=IngestResultOut)
def ingest_events(
    body: EventBatchIn, user: CurrentUser, db: DB, request: Request, key: IdempotencyKeyHeader = None
) -> dict[str, Any]:
    """Your own interaction events. Each item is accepted, a duplicate (same idempotency_key and the
    same content: the original event is returned) or rejected with a reason. A malformed batch is a
    422 as a whole; an item for another member is a 403."""
    cap = request.app.state.settings.events_batch_max
    if len(body.events) > cap:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"at most {cap} events per request")
    if any(e.user_id is not None and e.user_id != user.id for e in body.events):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "events can only be posted for your own account")
    fp = events.fingerprint("POST", "/events", body.model_dump(mode="json"))
    try:
        response, replayed = events.idempotent(
            db, events.user_scope(user), key, fp, lambda: _member_batch(db, user, body.events, request)
        )
    except events.IdempotencyMismatchError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    if response["accepted"] and not replayed:
        events.refresher_for(request.app).mark_dirty(DEFAULT_DOMAIN)
    return response


# --- POST /intel/domains/{key}/observations -----------------------------------------------------------------
def _observation_problem(
    o: ObservationIn, t: datetime, cfg: Any, entities: dict[str, Any], allowed: set[str], required: set[str]
) -> str | None:
    if o.entity not in entities:
        return f"unknown entity {o.entity!r} for this domain"
    if cfg.value_range is not None and not cfg.value_range[0] <= o.value <= cfg.value_range[1]:
        return f"value outside the domain's range [{cfg.value_range[0]:g}, {cfg.value_range[1]:g}]"
    if t.time() != datetime.min.time():
        return "event_time must be a period start at midnight UTC"
    if cfg.frequency == "month" and t.day != 1:
        return "event_time must be the first day of the month (monthly domain)"
    unknown = sorted(set(o.attributes) - allowed)
    if unknown:
        return f"unknown attributes {unknown}; allowed: {sorted(allowed)}"
    missing = sorted(required - set(o.attributes))
    if missing:
        return f"attributes {missing} are required for this domain"
    return None


@router.post("/intel/domains/{key}/observations", response_model=IngestResultOut)
def ingest_observations(
    key: Annotated[str, Path(min_length=1, max_length=64, pattern=DOMAIN_PATTERN)],
    body: ObservationBatchIn,
    user: ObservationWriter,
    db: DB,
    request: Request,
    idem: IdempotencyKeyHeader = None,
) -> dict[str, Any]:
    """Observations for a generic domain (entity, value, period). They enter the domain's next run
    (the debounced refresher starts a live run) and are replayable bitemporally."""
    settings = request.app.state.settings
    svc = request.app.state.intel
    if not key.startswith("generic:"):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "observations are accepted for generic domains only"
        )
    try:
        svc.require_available(key)
    except UnknownDomainError as exc:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"unknown domain {key!r}; see GET /intel/domains"
        ) from exc
    except DomainUnavailableError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, f"domain {exc.key!r} is unavailable: {exc.reason}"
        ) from exc
    cap = settings.events_observations_batch_max
    if len(body.observations) > cap:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"at most {cap} observations per request")
    adapter = svc.adapter(key)
    cfg = adapter.cfg
    entities = events.domain_entities(adapter)
    cal = cfg.calendar_check.get("column") if cfg.calendar_check else None
    allowed = set(cfg.columns.get("groups") or []) | ({cal} if cal else set())
    required = {cal} if cal else set()
    fp = events.fingerprint("POST", f"/intel/domains/{key}/observations", body.model_dump(mode="json"))

    def work() -> dict[str, Any]:
        now = utcnow()
        batch_id = str(uuid.uuid4())
        keys = [events.observation_key(key, o.idempotency_key) for o in body.observations]
        existing = _existing(db, events.INGEST, keys)
        items: list[dict[str, Any]] = []
        for i, o in enumerate(body.observations):
            t = _utc(o.event_time)
            prior = existing.get(keys[i])
            if prior is not None:
                same = (prior.entity_id, prior.value, _utc(prior.event_time)) == (o.entity, o.value, t)
                items.append(
                    _item(i, "duplicate", prior)
                    if same
                    else _item(
                        i, "rejected", reason="idempotency_key already used for a different observation"
                    )
                )
                continue
            reason = _time_problem(t, now, settings.events_max_future_skew_seconds, None)
            reason = reason or _observation_problem(o, t, cfg, entities, allowed, required)
            if reason is not None:
                items.append(_item(i, "rejected", reason=reason))
                continue
            ev = events.append(
                db,
                source=events.INGEST,
                domain=key,
                event_type="observation",
                entity_id=o.entity,
                value=o.value,
                payload={"attributes": o.attributes} if o.attributes else {},
                event_time=t,
                ingested_at=now,
                idempotency_key=keys[i],
                batch_id=batch_id,
            )
            existing[keys[i]] = ev
            items.append(_item(i, "accepted", ev))
        n_acc = sum(1 for it in items if it["status"] == "accepted")
        events.count_ingest(db, key, now.date(), accepted=n_acc)
        out = _summary(db, batch_id, key, items, now)
        audit.record(
            db,
            "events.ingest",
            user.actor,
            "domain",
            key,
            {k: out[k] for k in ("batch_id", "accepted", "duplicates", "rejected")},
        )
        return out

    try:
        response, replayed = events.idempotent(db, user.scope, idem, fp, work, domain=key)
    except events.IdempotencyMismatchError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    if response["accepted"] and not replayed:
        events.refresher_for(request.app).mark_dirty(key)
    return response


# --- replay and health (admin) ------------------------------------------------------------------------------
@router.post("/events/replay", response_model=ReplayOut)
def replay_projections(body: ReplayIn, user: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    """Fold the log and compare it with the live projections; with apply=true, rewrite them."""
    if not body.apply:
        diff = events.verify_projections(db)
        return {
            "applied": False,
            "consistent_before": diff["consistent"],
            "consistent_after": diff["consistent"],
            "diff": diff,
        }
    before = events.rebuild_projections(db)
    after = events.verify_projections(db)
    audit.record(
        db,
        "events.replay",
        user,
        "projections",
        "movie",
        {"consistent_before": before["consistent"], "consistent_after": after["consistent"], "diff": before},
    )
    db.commit()
    if not before["consistent"]:
        request.app.state.cache.delete_prefix("rec:")  # served lists may rest on the rewritten rows
    return {
        "applied": True,
        "consistent_before": before["consistent"],
        "consistent_after": after["consistent"],
        "diff": before,
    }


@router.get("/events/health", response_model=EventHealthOut)
def events_health(_: AdminUser, db: DB, request: Request) -> dict[str, Any]:
    return events.health(db, utcnow(), events.refresher_for(request.app))
