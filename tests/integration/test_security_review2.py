"""Security regression tests (SECURITY.md, "Second review (v1.1.0)").

Client address resolution (no X-Forwarded-For parsing), server-generated request ids, API security
headers, and one verdict per member per recommendation or movie (upsert, deduplicated aggregates,
deduplicated live-feedback input, migration 0004).

Runs after test_security.py in the same session database.
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from conftest import admin_headers, register
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, insert, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_intel_api import synthetic_frames
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from jev_ml.intel import PipelineInputs


@pytest.fixture(scope="module")
def admin(client):
    return admin_headers(client)


@pytest.fixture()
def auth_limit(client):
    """Auth rate limit of 3/min with trust_proxy on (the deployed setting), restored afterwards."""
    settings = client.app.state.settings
    original = settings.auth_rate_limit_per_minute, settings.trust_proxy
    settings.auth_rate_limit_per_minute, settings.trust_proxy = 3, True
    client.app.state.cache.delete_prefix("rl:")
    yield
    settings.auth_rate_limit_per_minute, settings.trust_proxy = original
    client.app.state.cache.delete_prefix("rl:")


def _failed_login(c: TestClient, email: str, **headers: str):
    return c.post("/auth/login", json={"email": email, "password": "wrong-password"}, headers=headers)


def _last_audit(client, admin, action: str, actor: str) -> dict:
    items = client.get(f"/admin/audit?action={action}&actor={actor}&limit=1", headers=admin).json()["items"]
    assert items, (action, actor)
    return items[0]


# --- finding 1: client IP spoofing ---------------------------------------------------------------


def test_spoofed_forwarded_for_does_not_reset_the_auth_bucket(client, auth_limit):
    codes = [
        _failed_login(
            client, "spoof-bucket@example.com", **{"X-Forwarded-For": f"198.51.100.{i}"}
        ).status_code
        for i in range(5)
    ]
    assert codes[:3] == [401, 401, 401]
    assert codes[3:] == [429, 429], codes  # a fresh X-Forwarded-For per request is still one client


def test_spoofed_forwarded_for_is_not_audited(client, admin):
    email = "spoof-audit@example.com"
    r = _failed_login(client, email, **{"X-Forwarded-For": "203.0.113.66, 10.0.0.1"})
    assert r.status_code == 401
    entry = _last_audit(client, admin, "auth.login.failure", email)
    assert entry["detail"]["client"] == "testclient"  # the TCP peer, not the header


def test_trusted_proxy_resolves_the_rightmost_untrusted_hop(client, admin, auth_limit):
    """Behind uvicorn's --proxy-headers with the proxy listed in --forwarded-allow-ips, the app sees
    the right-most address the proxy appended; entries the client prepended are ignored."""
    proxied = TestClient(ProxyHeadersMiddleware(client.app, trusted_hosts="testclient"))
    email = "spoof-proxy@example.com"
    r = _failed_login(proxied, email, **{"X-Forwarded-For": "203.0.113.66, 198.51.100.200"})
    assert r.status_code == 401
    assert _last_audit(client, admin, "auth.login.failure", email)["detail"]["client"] == "198.51.100.200"
    # the same real client with a different spoofed prefix shares one bucket
    codes = [
        _failed_login(proxied, email, **{"X-Forwarded-For": f"192.0.2.{i}, 198.51.100.200"}).status_code
        for i in range(3)
    ]
    assert codes == [401, 401, 429], codes
    # an untrusted peer's header is ignored entirely
    untrusted = TestClient(ProxyHeadersMiddleware(client.app, trusted_hosts="10.9.9.9"))
    email = "spoof-untrusted@example.com"
    _failed_login(untrusted, email, **{"X-Forwarded-For": "198.51.100.201"})
    assert _last_audit(client, admin, "auth.login.failure", email)["detail"]["client"] == "testclient"


# --- finding 4: client-controlled request ids --------------------------------------------------------


def test_forged_request_id_is_not_the_audit_request_id(client, admin, caplog):
    email = "forged-rid@example.com"
    with caplog.at_level(logging.INFO, logger="jev_api"):
        r = _failed_login(client, email, **{"X-Request-ID": "forged-audit-id"})
    rid = r.headers["x-request-id"]
    assert rid != "forged-audit-id" and uuid.UUID(rid)
    assert r.json()["request_id"] == rid
    assert _last_audit(client, admin, "auth.login.failure", email)["request_id"] == rid
    line = next(rec for rec in caplog.records if getattr(rec, "extra_fields", {}).get("request_id") == rid)
    assert line.extra_fields["upstream_request_id"] == "forged-audit-id"  # kept for correlation only


@pytest.mark.parametrize("value", ["has space", "x" * 65, "<script>", "idé"])
def test_malformed_upstream_request_ids_are_dropped(client, caplog, value):
    with caplog.at_level(logging.INFO, logger="jev_api"):
        r = client.get("/health", headers={"X-Request-ID": value.encode("utf-8")})
    rid = r.headers["x-request-id"]
    assert uuid.UUID(rid)
    line = next(rec for rec in caplog.records if getattr(rec, "extra_fields", {}).get("request_id") == rid)
    assert line.extra_fields["upstream_request_id"] is None


# --- finding 5: security headers ------------------------------------------------------------------


def test_api_security_headers(client):
    r = client.get("/health")
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    assert r.headers["cross-origin-opener-policy"] == "same-origin"
    assert "strict-transport-security" not in r.headers  # plain http: no HSTS
    https = TestClient(client.app, base_url="https://testserver")
    assert https.get("/health").headers["strict-transport-security"].startswith("max-age=")


# --- finding 3: duplicate recommendation feedback -------------------------------------------------


def _totals(client, admin) -> dict:
    return client.get("/intel/recommendations?recent=0", headers=admin).json()["feedback_totals"]


def _feedback(client, headers, item, kind):
    body = {"movie_id": item["movie_id"], "feedback": kind, "recommendation_id": item["recommendation_id"]}
    return client.post("/recommendations/feedback", headers=headers, json=body)


def test_repeated_dislikes_count_once(client, admin):
    user = register(client, "dup-dislike@example.com")
    item = client.get("/recommendations?limit=1", headers=user).json()["items"][0]
    before = _totals(client, admin)
    responses = [_feedback(client, user, item, "dislike") for _ in range(25)]
    assert {r.status_code for r in responses} == {201}
    assert {r.json()["id"] for r in responses} == {responses[0].json()["id"]}  # one row, updated in place
    assert set(responses[0].json()) == {"id", "movie_id", "feedback", "recommendation_id", "created_at"}
    after = _totals(client, admin)
    assert after["dislike"] - before["dislike"] == 1


def test_changing_the_verdict_replaces_it(client, admin):
    user = register(client, "dup-change@example.com")
    item = client.get("/recommendations?limit=1", headers=user).json()["items"][0]
    before = _totals(client, admin)
    first = _feedback(client, user, item, "dislike").json()
    # a click is an interaction, not a verdict: its own row, the dislike stays
    click = _feedback(client, user, item, "clicked").json()
    assert click["id"] != first["id"]
    second = _feedback(client, user, item, "like").json()
    assert second["id"] == first["id"] and second["feedback"] == "like"
    after = _totals(client, admin)
    assert after["like"] - before["like"] == 1 and after["dislike"] == before["dislike"]
    assert after["clicked"] - before["clicked"] == 1
    hist = client.get("/recommendations/history?limit=50", headers=user).json()
    assert next(h for h in hist if h["id"] == item["recommendation_id"])["feedback"] == "like"
    # "not interested" hides the movie; changing it back to "like" brings it back into the pool
    _feedback(client, user, item, "not_interested")
    ids = [i["movie_id"] for i in client.get("/recommendations?limit=50", headers=user).json()["items"]]
    assert item["movie_id"] not in ids


def test_many_recommendation_ids_for_one_movie_count_once(client, admin):
    """Each served list creates new recommendation ids; aggregates still count one verdict per member
    per movie (defence in depth behind the per-recommendation unique index)."""
    from jev_api.db import SessionLocal
    from jev_api.models import Recommendation, User

    user = register(client, "dup-many-recs@example.com")
    item = client.get("/recommendations?limit=1", headers=user).json()["items"][0]
    with SessionLocal() as db:
        uid = db.scalar(select(User.id).where(User.email == "dup-many-recs@example.com"))
        src = db.get(Recommendation, item["recommendation_id"])
        clones = []
        for i in range(10):
            rec = Recommendation(
                user_id=uid,
                movie_id=src.movie_id,
                request_id=str(uuid.uuid4()),
                model_version=src.model_version,
                context=src.context,
                rank=i + 1,
                score=src.score,
                reason=src.reason,
                reason_code=src.reason_code,
                signals={},
            )
            db.add(rec)
            clones.append(rec)
        db.commit()
        clone_ids = [r.id for r in clones]
    before = _totals(client, admin)
    for rid in clone_ids:
        r = _feedback(client, user, {"movie_id": item["movie_id"], "recommendation_id": rid}, "dislike")
        assert r.status_code == 201
    after = _totals(client, admin)
    assert after["dislike"] - before["dislike"] == 1


def test_live_feedback_input_is_deduplicated(client):
    from jev_api.db import SessionLocal
    from jev_api.models import RecommendationFeedback, User

    inter, movies = synthetic_frames()
    svc = client.app.state.intel
    original = svc.inputs_factory
    svc.inputs_factory = lambda as_of: PipelineInputs(
        interactions=inter.copy(), movies=movies.copy(), dataset_meta={"dataset_version": "x"}, as_of=as_of
    )
    try:
        user = register(client, "dup-live@example.com")
        items = client.get("/recommendations?limit=3", headers=user).json()["items"]
        with SessionLocal() as db:
            n_before = len(svc.gather_inputs(db, None, datetime.now(UTC)).app_feedback)
        for item in items:
            for _ in range(10):
                _feedback(client, user, item, "not_interested")
                _feedback(client, user, item, "clicked")
        with SessionLocal() as db:
            inputs = svc.gather_inputs(db, None, datetime.now(UTC))
            uid = db.scalar(select(User.id).where(User.email == "dup-live@example.com"))
            stored = db.scalar(
                select(func.count())
                .select_from(RecommendationFeedback)
                .where(RecommendationFeedback.user_id == uid)
            )
        assert stored == 2 * len(items)  # one verdict + one click per recommendation
        assert len(inputs.app_feedback) - n_before == 2 * len(items)
    finally:
        svc.inputs_factory = original


def test_duplicate_rows_are_rejected_by_the_database(client):
    from jev_api.db import SessionLocal
    from jev_api.models import RecommendationFeedback, User

    user = register(client, "dup-db@example.com")
    item = client.get("/recommendations?limit=1", headers=user).json()["items"][0]
    assert _feedback(client, user, item, "like").status_code == 201
    with SessionLocal() as db:
        uid = db.scalar(select(User.id).where(User.email == "dup-db@example.com"))
        db.add(
            RecommendationFeedback(
                user_id=uid,
                movie_id=item["movie_id"],
                recommendation_id=item["recommendation_id"],
                feedback="dislike",
            )
        )
        with pytest.raises(IntegrityError):
            db.commit()
        db.rollback()
        # without a recommendation, the key is (user, movie)
        for _ in range(2):
            db.add(RecommendationFeedback(user_id=uid, movie_id=item["movie_id"], feedback="like"))
        with pytest.raises(IntegrityError):
            db.commit()


def _migration_roundtrip(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    from jev_api.models import Movie, Recommendation, RecommendationFeedback, User
    from jev_ml.paths import ROOT

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "jev_api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.downgrade(cfg, "base")  # a reused PostgreSQL database starts clean
    command.upgrade(cfg, "0003")
    eng = create_engine(url)
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    try:
        with Session(eng) as db:
            users = [User(email=f"m{i}@example.com", password_hash="x", display_name="m") for i in range(2)]
            movie = Movie(id=1, title="One")
            db.add_all([*users, movie])
            db.flush()
            # core inserts with explicit columns: the ORM model has columns added after 0003 (0005)
            rec_ids = [
                db.execute(
                    insert(Recommendation.__table__)
                    .values(
                        user_id=users[0].id,
                        movie_id=1,
                        request_id=str(uuid.uuid4()),
                        model_version="v",
                        context="feed",
                        rank=r,
                        score=1.0,
                        reason="r",
                        reason_code="c",
                        signals={},
                        created_at=t0,
                    )
                    .returning(Recommendation.__table__.c.id)
                ).scalar_one()
                for r in (1, 2)
            ]
            u0, u1, r1, r2 = users[0].id, users[1].id, rec_ids[0], rec_ids[1]
            rows = [
                # (user, rec, feedback, minutes after t0)
                (u0, r1, "dislike", 0),
                (u0, r1, "dislike", 1),
                (u0, r1, "like", 2),  # newest verdict on r1: kept
                (u0, r1, "clicked", 0),
                (u0, r1, "clicked", 3),  # newest click on r1: kept
                (u0, r2, "not_interested", 0),  # another recommendation: kept
                (u0, None, "dislike", 0),
                (u0, None, "dislike", 5),  # newest movie-level verdict: kept
                (u1, None, "like", 0),  # another member: kept
            ]
            db.execute(
                insert(RecommendationFeedback.__table__),
                [
                    {
                        "user_id": u,
                        "movie_id": 1,
                        "recommendation_id": r,
                        "feedback": f,
                        "created_at": t0 + timedelta(minutes=m),
                    }
                    for u, r, f, m in rows
                ],
            )
            db.commit()
        command.upgrade(cfg, "0004")
        fb = RecommendationFeedback.__table__
        with Session(eng) as db:
            left = db.execute(select(fb.c.user_id, fb.c.recommendation_id, fb.c.feedback)).all()
        assert sorted(left, key=repr) == sorted(
            [
                (u0, r1, "like"),
                (u0, r1, "clicked"),
                (u0, r2, "not_interested"),
                (u0, None, "dislike"),
                (u1, None, "like"),
            ],
            key=repr,
        )
        names = {i["name"] for i in inspect(eng).get_indexes("recommendation_feedback")}
        assert {"uq_feedback_verdict_rec", "uq_feedback_verdict_movie"} <= names
        with Session(eng) as db, pytest.raises(IntegrityError):
            dup = {
                "user_id": u0,
                "movie_id": 1,
                "recommendation_id": r1,
                "feedback": "dislike",
                "created_at": t0,
            }
            db.execute(insert(fb), dup)
            db.commit()
        command.downgrade(cfg, "0003")
        after = {i["name"] for i in inspect(eng).get_indexes("recommendation_feedback")}
        assert not {n for n in after if n.startswith("uq_feedback_")}
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")
    finally:
        eng.dispose()


def test_migration_0004_dedups_existing_rows_sqlite(tmp_path):
    _migration_roundtrip(f"sqlite:///{tmp_path / 'dedup.db'}")


@pytest.mark.skipif(not os.environ.get("JEV_TEST_POSTGRES_URL"), reason="JEV_TEST_POSTGRES_URL not set")
def test_migration_0004_dedups_existing_rows_postgres():
    _migration_roundtrip(os.environ["JEV_TEST_POSTGRES_URL"])
