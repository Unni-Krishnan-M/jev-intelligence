"""WS1 events (docs/STREAMING_ARCHITECTURE.md): the append-only log behind the member write endpoints.

- Idempotency-Key on rate / unrate / favorite / watch / POST /events: a retry returns the same response
  and appends nothing; a key reused for another request is a 422.
- The keyless watch retry rule (identical watch within events_watch_dedupe_seconds).
- POST /events: validation (422), per-item accepted / duplicate / rejected, preserved event times,
  and no posting for another member (403).
- Projections: the live tables equal a fold of the log; POST /events/replay verifies and repairs.
- Bitemporal reads: a replay at an earlier knowledge time never sees a late-arriving event.
- The events watermark is recorded on every intelligence run.
- The log is append-only at the database level (triggers from migration 0007).
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from conftest import admin_headers, register
from sqlalchemy import func, select, text
from sqlalchemy.exc import DatabaseError
from test_intel_api import synthetic_frames

from jev_ml.intel import PipelineInputs


def _uid(client, headers) -> int:
    return client.get("/users/me", headers=headers).json()["id"]


def _events(**where) -> list:
    from jev_api.db import SessionLocal
    from jev_api.models.events import Event

    with SessionLocal() as db:
        q = select(Event)
        for k, v in where.items():
            q = q.where(getattr(Event, k) == v)
        return list(db.scalars(q.order_by(Event.id)))


def _count(model, **where) -> int:
    from jev_api.db import SessionLocal

    with SessionLocal() as db:
        q = select(func.count()).select_from(model)
        for k, v in where.items():
            q = q.where(getattr(model, k) == v)
        return int(db.scalar(q) or 0)


def _ev(event_type: str, movie_id: int, key: str, **fields) -> dict:
    return {"event_type": event_type, "movie_id": movie_id, "idempotency_key": key, **fields}


@pytest.fixture
def member(client):
    headers = register(client, f"ev-{uuid.uuid4().hex[:8]}@example.com")
    return headers, _uid(client, headers)


# --- Idempotency-Key on the member endpoints -------------------------------------------------------------
def test_rate_with_idempotency_key_ten_times_is_one_event(client, member):
    headers, uid = member
    h = {**headers, "Idempotency-Key": "rate-once-1"}
    responses = [client.post("/movies/3/rate", headers=h, json={"rating": 4.5}) for _ in range(10)]
    assert all(r.status_code == 200 for r in responses)
    assert len({r.text for r in responses}) == 1  # identical responses, including profile_version
    assert len(_events(user_id=uid, event_type="rating")) == 1
    from jev_api.models import Rating

    assert _count(Rating, user_id=uid, movie_id=3) == 1
    # the key is spent: another request with it is refused, not silently replayed
    r = client.post("/movies/3/rate", headers=h, json={"rating": 2.0})
    assert r.status_code == 422, r.text
    assert "different request" in r.json()["detail"]


def test_every_member_write_is_idempotent_with_a_key(client, member):
    from jev_api.models import Favorite, WatchHistory

    headers, uid = member
    calls = [
        ("post", "/movies/5/favorite", {"favorite": True}),
        ("post", "/movies/5/watch", None),
        ("post", "/movies/5/rate", {"rating": 3.0}),
        ("delete", "/movies/5/rate", None),
    ]
    for i, (method, path, body) in enumerate(calls):
        h = {**headers, "Idempotency-Key": f"k-{i}"}
        first = client.request(method, path, headers=h, json=body)
        assert first.status_code == 200, first.text
        for _ in range(3):
            again = client.request(method, path, headers=h, json=body)
            assert again.json() == first.json()
    assert [e.event_type for e in _events(user_id=uid)] == ["favorite", "watch", "rating", "rating_removed"]
    assert _count(WatchHistory, user_id=uid) == 1
    assert _count(Favorite, user_id=uid) == 1
    detail = client.get("/movies/5", headers=headers).json()
    assert detail["user_rating"] is None and detail["is_favorite"] and detail["watched"]


def test_responses_without_a_key_are_unchanged(client, member):
    headers, uid = member
    r1 = client.post("/movies/7/rate", headers=headers, json={"rating": 4.0}).json()
    r2 = client.post("/movies/7/rate", headers=headers, json={"rating": 4.0}).json()
    assert r1["rating"] == r2["rating"] == 4.0
    assert r2["profile_version"] == r1["profile_version"] + 1  # as before Phase 2
    assert client.delete("/movies/99999/rate", headers=headers).status_code == 200  # unknown: no-op
    assert client.post("/movies/99999/rate", headers=headers, json={"rating": 4.0}).status_code == 404
    fav = client.post("/movies/7/favorite", headers=headers).json()
    assert fav["is_favorite"] is True and set(fav) == {
        "movie_id",
        "rating",
        "is_favorite",
        "watched",
        "profile_version",
    }
    # a keyless re-rate with the same value appends an event but the projection does not move
    from jev_api.db import SessionLocal
    from jev_api.models import Rating

    with SessionLocal() as db:
        row = db.scalar(select(Rating).where(Rating.user_id == uid, Rating.movie_id == 7))
        assert row is not None and row.rating == 4.0
        assert row.created_at == row.updated_at


def test_invalid_idempotency_key_is_422(client, member):
    headers, _ = member
    r = client.post("/movies/3/rate", headers={**headers, "Idempotency-Key": "has space"}, json={"rating": 4})
    assert r.status_code == 422


# --- the keyless watch retry rule --------------------------------------------------------------------------
def test_keyless_watch_retry_within_the_window_is_deduplicated(client, member):
    from jev_api.models import WatchHistory

    headers, uid = member
    first = client.post("/movies/9/watch", headers=headers).json()
    retry = client.post("/movies/9/watch", headers=headers).json()
    assert retry == {**first}  # same state, same profile_version: nothing new happened
    assert _count(WatchHistory, user_id=uid, movie_id=9) == 1
    assert len(_events(user_id=uid, event_type="watch")) == 1
    # a different movie is not a retry, and an explicit new key is a new watch
    client.post("/movies/10/watch", headers=headers)
    client.post("/movies/9/watch", headers={**headers, "Idempotency-Key": "rewatch-9"})
    assert _count(WatchHistory, user_id=uid, movie_id=9) == 2
    assert _count(WatchHistory, user_id=uid) == 3


def test_keyless_watch_after_the_window_is_a_new_watch(client, member):
    from jev_api.models import WatchHistory

    headers, uid = member
    settings = client.app.state.settings
    original = settings.events_watch_dedupe_seconds
    settings.events_watch_dedupe_seconds = 0
    try:
        client.post("/movies/11/watch", headers=headers)
        client.post("/movies/11/watch", headers=headers)
    finally:
        settings.events_watch_dedupe_seconds = original
    assert _count(WatchHistory, user_id=uid, movie_id=11) == 2


# --- POST /events ------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "item",
    [
        {"event_type": "rating", "movie_id": 1, "idempotency_key": "a"},  # rating without value
        {"event_type": "rating", "movie_id": 1, "value": 4.2, "idempotency_key": "a"},  # not a half step
        {"event_type": "watch", "movie_id": 1, "value": 3, "idempotency_key": "a"},  # value on a watch
        {"event_type": "rec_feedback", "movie_id": 1, "idempotency_key": "a"},  # no feedback
        {"event_type": "observation", "movie_id": 1, "idempotency_key": "a"},  # not a member type
        {"event_type": "watch", "movie_id": 1},  # no key
        {"event_type": "watch", "movie_id": 1, "idempotency_key": "a", "surprise": 1},  # extra field
        {"event_type": "watch", "movie_id": -1, "idempotency_key": "a"},
    ],
)
def test_post_events_validation_is_422(client, member, item):
    headers, uid = member
    r = client.post("/events", headers=headers, json={"events": [item]})
    assert r.status_code == 422, r.text
    assert _events(user_id=uid) == []


def test_post_events_empty_and_oversized_batches_are_422(client, member):
    headers, _ = member
    assert client.post("/events", headers=headers, json={"events": []}).status_code == 422
    cap = client.app.state.settings.events_batch_max
    batch = [{"event_type": "watch", "movie_id": 1, "idempotency_key": f"w{i}"} for i in range(cap + 1)]
    r = client.post("/events", headers=headers, json={"events": batch})
    assert r.status_code == 422 and "at most" in r.json()["detail"]


def test_post_events_requires_login_and_only_for_yourself(client, member):
    headers, uid = member
    item = {"event_type": "watch", "movie_id": 1, "idempotency_key": "x"}
    assert client.post("/events", json={"events": [item]}).status_code == 401
    other = register(client, f"ev-other-{uuid.uuid4().hex[:6]}@example.com")
    other_id = _uid(client, other)
    r = client.post("/events", headers=headers, json={"events": [{**item, "user_id": other_id}]})
    assert r.status_code == 403
    assert _events(user_id=other_id) == [] and _events(user_id=uid) == []
    ok = client.post("/events", headers=headers, json={"events": [{**item, "user_id": uid}]})
    assert ok.status_code == 200 and ok.json()["accepted"] == 1


def test_post_events_per_item_results_and_preserved_timestamps(client, member):
    from jev_api.models import Rating, WatchHistory

    headers, uid = member
    t_rate = datetime.now(UTC).replace(microsecond=0) - timedelta(days=2)
    t_watch = t_rate + timedelta(hours=1)
    batch = [
        _ev("rating", 12, "r1", value=3.5, event_time=t_rate.isoformat()),
        {"event_type": "watch", "movie_id": 12, "idempotency_key": "w1", "event_time": t_watch.isoformat()},
        {"event_type": "watch", "movie_id": 12, "idempotency_key": "w1", "event_time": t_watch.isoformat()},
        {"event_type": "watch", "movie_id": 99999, "idempotency_key": "w2"},
        _ev("watch", 12, "w3", event_time="2999-01-01T00:00:00Z"),
        _ev("watch", 12, "w4", event_time="2000-01-01T00:00:00Z"),
        _ev("rec_feedback", 12, "f1", feedback="like", recommendation_id=2**31 - 1),
    ]  # fmt: skip
    r = client.post("/events", headers=headers, json={"events": batch})
    assert r.status_code == 200, r.text
    out = r.json()
    assert (out["accepted"], out["duplicates"], out["rejected"]) == (2, 1, 4)
    status = [i["status"] for i in out["items"]]
    assert status == ["accepted", "accepted", "duplicate", "rejected", "rejected", "rejected", "rejected"]
    reasons = [i["reason"] for i in out["items"]]
    assert reasons[3] == "movie not found"
    assert reasons[4] == "event_time is in the future"
    assert "days old" in reasons[5]
    assert reasons[6] == "recommendation not found"
    assert out["items"][2]["event_id"] == out["items"][1]["event_id"]  # the original event
    # event time is preserved in the log and in the projections; ingested_at is the server's clock
    evs = _events(user_id=uid)
    assert [e.event_type for e in evs] == ["rating", "watch"]
    assert evs[0].event_time.replace(tzinfo=UTC) == t_rate
    assert evs[0].ingested_at.replace(tzinfo=UTC) > t_rate + timedelta(days=1)
    from jev_api.db import SessionLocal

    with SessionLocal() as db:
        rating = db.scalar(select(Rating).where(Rating.user_id == uid))
        watch = db.scalar(select(WatchHistory).where(WatchHistory.user_id == uid))
        assert rating.updated_at.replace(tzinfo=UTC) == t_rate
        assert watch.watched_at.replace(tzinfo=UTC) == t_watch
    # the whole batch again: every item is now a duplicate (or still rejected); nothing is appended
    again = client.post("/events", headers=headers, json={"events": batch}).json()
    assert (again["accepted"], again["duplicates"], again["rejected"]) == (0, 3, 4)
    assert len(_events(user_id=uid)) == 2
    # the same key for a different event is rejected, not merged
    clash = [{"event_type": "rating", "movie_id": 12, "value": 1.0, "idempotency_key": "r1"}]
    item = client.post("/events", headers=headers, json={"events": clash}).json()["items"][0]
    assert item["status"] == "rejected" and "different event" in item["reason"]


def test_post_events_with_request_key_returns_identical_responses(client, member):
    headers, uid = member
    h = {**headers, "Idempotency-Key": "batch-1"}
    body = {"events": [{"event_type": "favorite", "movie_id": 14, "idempotency_key": "fav-14"}]}
    responses = [client.post("/events", headers=h, json=body) for _ in range(10)]
    assert len({r.text for r in responses}) == 1
    assert responses[0].json()["accepted"] == 1
    assert len(_events(user_id=uid)) == 1


def test_late_event_lands_in_event_time_order(client, member):
    """An older event that arrives after a newer one does not overwrite it (fold order = event time)."""
    headers, uid = member
    now = datetime.now(UTC)
    new = {
        "event_type": "rating",
        "movie_id": 15,
        "value": 5.0,
        "idempotency_key": "new",
        "event_time": now.isoformat(),
    }
    old = _ev("rating", 15, "old", value=1.0, event_time=(now - timedelta(hours=3)).isoformat())
    client.post("/events", headers=headers, json={"events": [new]})
    client.post("/events", headers=headers, json={"events": [old]})
    assert client.get("/movies/15", headers=headers).json()["user_rating"] == 5.0
    assert len(_events(user_id=uid, entity_id="15")) == 2


# --- projections --------------------------------------------------------------------------------------------
def test_projection_rebuild_equals_live_tables(client, member):
    headers, uid = member
    for path, body in [
        ("/movies/20/rate", {"rating": 2.0}),
        ("/movies/20/rate", {"rating": 4.5}),
        ("/movies/21/rate", {"rating": 3.0}),
        ("/movies/20/favorite", {"favorite": True}),
        ("/movies/21/favorite", {"favorite": True}),
        ("/movies/21/favorite", {"favorite": False}),
        ("/movies/22/watch", None),
    ]:
        assert client.post(path, headers=headers, json=body).status_code == 200
    assert client.delete("/movies/21/rate", headers=headers).status_code == 200
    items = client.get("/recommendations?limit=2", headers=headers).json()["items"]
    for kind in ("like", "dislike", "clicked", "clicked"):
        body = {
            "movie_id": items[0]["movie_id"],
            "recommendation_id": items[0]["recommendation_id"],
            "feedback": kind,
        }
        assert client.post("/recommendations/feedback", headers=headers, json=body).status_code == 201
    body = {"movie_id": items[1]["movie_id"], "feedback": "not_interested"}
    assert client.post("/recommendations/feedback", headers=headers, json=body).status_code == 201
    onboard = client.post(
        "/users/me/onboarding", headers=headers, json={"genres": ["Drama"], "movie_ids": [23, 24]}
    )
    assert onboard.status_code == 200, onboard.text

    from jev_api.db import SessionLocal
    from jev_api.services import events

    with SessionLocal() as db:
        folded, live = events.fold_log(db), events.live_projections(db)
        diff = events.diff_projections(folded, live)
        assert diff["consistent"], diff
        mine = {k: v for k, v in live.ratings.items() if k[0] == uid}
        assert mine == {k: v for k, v in folded.ratings.items() if k[0] == uid}
        assert {k for k in live.favorites if k[0] == uid} == {(uid, 20), (uid, 23), (uid, 24)}
        assert live.favorites[(uid, 23)].source == "onboarding"
    admin = admin_headers(client)
    r = client.post("/events/replay", headers=admin, json={})
    assert r.status_code == 200 and r.json()["consistent_before"] is True
    assert client.post("/events/replay", headers=headers, json={}).status_code == 403


def test_replay_repairs_a_drifted_projection(client, member):
    from jev_api.db import SessionLocal
    from jev_api.models import Rating, WatchHistory

    headers, uid = member
    client.post("/movies/25/rate", headers=headers, json={"rating": 4.0})
    client.post("/movies/26/watch", headers=headers)
    with SessionLocal() as db:  # simulate drift: a projection edited outside the log
        db.execute(Rating.__table__.update().where(Rating.user_id == uid).values(rating=1.0))
        db.add(WatchHistory(user_id=uid, movie_id=27))
        db.commit()
    admin = admin_headers(client)
    check = client.post("/events/replay", headers=admin, json={}).json()
    assert check["consistent_before"] is False and check["applied"] is False
    assert check["diff"]["ratings"]["changed"] >= 1 and check["diff"]["watch_history"]["extra"] >= 1
    fixed = client.post("/events/replay", headers=admin, json={"apply": True}).json()
    assert fixed["applied"] and fixed["consistent_after"] is True
    assert client.get("/movies/25", headers=headers).json()["user_rating"] == 4.0
    assert _count(WatchHistory, user_id=uid, movie_id=27) == 0
    from jev_api.models import AuditLog

    assert _count(AuditLog, action="events.replay") >= 1


# --- bitemporal reads ---------------------------------------------------------------------------------------
def test_replay_never_sees_late_arriving_events(client, member):
    """event_time <= as_of AND ingested_at <= knowledge_time (default: as_of)."""
    from jev_api.db import SessionLocal
    from jev_api.models import User
    from jev_api.services import events

    _, uid = member
    as_of = datetime.now(UTC) - timedelta(days=10)
    with SessionLocal() as db:
        user = db.get(User, uid)
        # known at the time
        events.apply_member_event(
            db,
            user,
            "rating",
            30,
            value=4.0,
            event_time=as_of - timedelta(days=2),
            now=as_of - timedelta(days=2),
        )
        # happened before as_of, but only arrived a day after it (late)
        late = events.apply_member_event(
            db,
            user,
            "rating",
            30,
            value=2.0,
            event_time=as_of - timedelta(days=1),
            now=as_of + timedelta(days=1),
        )
        # happened after as_of
        events.apply_member_event(
            db,
            user,
            "rating",
            30,
            value=5.0,
            event_time=as_of + timedelta(days=2),
            now=as_of + timedelta(days=2),
        )
        db.commit()  # fmt: skip

        def rating_at(as_of_, knowledge=None) -> float | None:
            frame = events.app_frames(db, as_of_, datetime.now(UTC), knowledge)[0]
            mine = frame[(frame.user_id == uid) & (frame.movie_id == 30)]
            return float(mine.rating.iloc[0]) if len(mine) else None

        assert rating_at(as_of) == 4.0  # the replay: the late event is invisible
        assert rating_at(as_of, as_of + timedelta(days=1)) == 2.0  # knowing what arrived later
        assert rating_at(None) == 5.0  # live
        assert rating_at(as_of - timedelta(days=3)) is None
        wm = events.watermark(db, "movie", as_of, datetime.now(UTC))
        assert wm["max_event_id"] < late.event.id
        assert wm["knowledge_time"] == wm["as_of"]
        # determinism: the same cut folds to the same frames however often it is read
        a = events.app_frames(db, as_of, datetime.now(UTC))
        b = events.app_frames(db, as_of, datetime.now(UTC))
        assert a[0].equals(b[0]) and a[1].equals(b[1]) and a[3] == b[3]


# --- the watermark ----------------------------------------------------------------------------------
def test_movie_inputs_carry_the_events_watermark(client, member):
    """The movie gather returns the watermark the run records (a generic run is persisted and checked in
    test_events_ingest.py; no movie run is created here, so later suites start without one)."""
    from jev_api.db import SessionLocal

    headers, uid = member
    svc = client.app.state.intel
    inter, movies = synthetic_frames()
    original = svc.inputs_factory
    svc.inputs_factory = lambda as_of: PipelineInputs(
        interactions=inter.copy(), movies=movies.copy(), dataset_meta={"dataset_version": "x"}, as_of=as_of
    )
    try:
        client.post("/movies/31/rate", headers=headers, json={"rating": 3.5})
        newest = _events()[-1].id
        now = datetime.now(UTC)
        with SessionLocal() as db:
            inputs, wm = svc.gather_movie_inputs(db, None, now)
        assert wm["domain"] == "movie" and wm["max_event_id"] == newest and wm["n_events"] >= 1
        assert wm["knowledge_time"] == wm["as_of"]  # live: both cuts are now
        mine = inputs.app_ratings[inputs.app_ratings.user_id == uid]
        assert mine[["movie_id", "rating"]].values.tolist() == [[31, 3.5]]
    finally:
        svc.inputs_factory = original


# --- the log is append-only ---------------------------------------------------------------------------------
def test_events_table_refuses_update_and_delete(client, member):
    from jev_api.db import SessionLocal

    headers, uid = member
    client.post("/movies/32/watch", headers=headers)
    for stmt in ("UPDATE events SET value = 1 WHERE user_id = :u", "DELETE FROM events WHERE user_id = :u"):
        with SessionLocal() as db:
            with pytest.raises(DatabaseError, match="append-only"):
                db.execute(text(stmt), {"u": uid})
            db.rollback()
    assert len(_events(user_id=uid)) == 1


def _insert(conn, table: str, t: datetime | None = None, **values) -> None:
    """Insert with every other NOT NULL column (without a server default) filled with a neutral value."""
    from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, MetaData, Table, insert

    tbl = Table(table, MetaData(), autoload_with=conn)
    for col in tbl.columns:
        if col.name in values or col.nullable or col.server_default is not None or col.primary_key:
            continue
        kind = col.type
        values[col.name] = (
            t if isinstance(kind, DateTime)
            else False if isinstance(kind, Boolean)
            else 0 if isinstance(kind, Integer | Float)
            else [] if isinstance(kind, JSON)
            else "x"
        )  # fmt: skip
    conn.execute(insert(tbl).values(**values))


def test_migration_0007_backfills_projections_and_round_trips(tmp_path):
    from alembic import command
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.orm import Session
    from test_migrations import _cfg

    from jev_api.services import events

    url = f"sqlite:///{tmp_path / 'm0007.db'}"
    cfg = _cfg(url)
    command.upgrade(cfg, "0006")
    eng = create_engine(url)
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    with eng.begin() as c:
        _insert(c, "users", id=1, email="a@x", password_hash="x", display_name="a", t=t0)
        _insert(c, "movies", id=1, title="One", t=t0)
        _insert(
            c, "ratings", user_id=1, movie_id=1, rating=4.0, created_at=t0, updated_at=t0 + timedelta(days=1)
        )
        _insert(c, "favorites", user_id=1, movie_id=1, source="onboarding", created_at=t0)
        _insert(c, "watch_history", user_id=1, movie_id=1, watched_at=t0)
        _insert(c, "watch_history", user_id=1, movie_id=1, watched_at=t0)
        _insert(c, "recommendation_feedback", user_id=1, movie_id=1, feedback="like", created_at=t0)
    command.upgrade(cfg, "0007")
    insp = inspect(eng)
    assert {"events", "idempotency_keys", "event_daily_counts"} <= set(insp.get_table_names())
    assert "event_watermark" in {c["name"] for c in insp.get_columns("intel_runs")}
    uniques = {u["name"] for u in insp.get_unique_constraints("events")}
    assert {"uq_events_source_key", "uq_events_event_id"} <= uniques
    assert {"ix_events_domain_time", "ix_events_domain_ingested"} <= {
        i["name"] for i in insp.get_indexes("events")
    }
    with Session(eng) as db:
        n = db.scalar(text("SELECT COUNT(*) FROM events WHERE source = 'backfill'"))
        assert n == 5
        assert events.verify_projections(db)["consistent"]
    command.downgrade(cfg, "0006")
    assert "events" not in inspect(eng).get_table_names()
    command.upgrade(cfg, "head")
    eng.dispose()


@pytest.mark.skipif(not os.environ.get("JEV_TEST_POSTGRES_URL"), reason="JEV_TEST_POSTGRES_URL not set")
def test_migration_0007_append_only_postgres():
    from alembic import command
    from sqlalchemy import create_engine
    from test_migrations import _cfg

    url = os.environ["JEV_TEST_POSTGRES_URL"]
    cfg = _cfg(url)
    command.downgrade(cfg, "base")  # a reused PostgreSQL database starts clean
    command.upgrade(cfg, "0006")
    eng = create_engine(url)
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    try:
        with eng.begin() as c:
            _insert(c, "users", id=1, email="a@x", password_hash="x", display_name="a", t=t0)
            _insert(c, "movies", id=1, title="One", t=t0)
            _insert(c, "ratings", user_id=1, movie_id=1, rating=4.0, created_at=t0, updated_at=t0)
        command.upgrade(cfg, "head")
        with eng.connect() as c:
            assert c.scalar(text("SELECT COUNT(*) FROM events WHERE source = 'backfill'")) == 1
        for stmt in ("UPDATE events SET value = 1", "DELETE FROM events", "TRUNCATE events"):
            with eng.connect() as c, pytest.raises(DatabaseError, match="append-only"):
                c.execute(text(stmt))
        command.downgrade(cfg, "0006")
        with eng.connect() as c:
            assert c.scalar(text("SELECT to_regclass('events')")) is None
            fn = "SELECT COUNT(*) FROM pg_proc WHERE proname = 'jev_events_append_only'"
            assert c.scalar(text(fn)) == 0
        command.upgrade(cfg, "head")
    finally:
        eng.dispose()
