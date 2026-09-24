"""Phase 2 security regression tests (docs/SECURITY_AUDIT_PHASE2.md).

F1 ensure_admin never promotes an account it cannot prove the operator owns; F2 session revocation
(logout jti denylist, revoke-all, password change, legacy tokens); F3 scoped, hashed, expiring and
revocable service tokens for observation ingestion; F4 the per-account login throttle and the signed
client address from the web proxy.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from conftest import admin_headers, register
from sqlalchemy import select
from test_events_ingest import DOMAIN, URL, _obs, generic  # noqa: F401 - the generic fixture

from jev_api.config import get_settings
from jev_api.db import SessionLocal
from jev_api.models import AuditLog, ServiceToken, User
from jev_api.security import CLIENT_HEADER, sign_client_address
from jev_api.services.sync import ensure_admin

PROXY_SECRET = "proxy-secret-for-tests-0123456789"


@pytest.fixture(scope="module")
def admin(client):
    return admin_headers(client)


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _login(client, email: str, password: str, **headers: str):
    r = client.post("/auth/login", json={"email": email, "password": password}, headers=headers)
    client.cookies.clear()
    return r


def _audits(action: str, target_id: str | None = None) -> list[AuditLog]:
    with SessionLocal() as db:
        q = select(AuditLog).where(AuditLog.action == action)
        if target_id is not None:
            q = q.where(AuditLog.target_id == target_id)
        return list(db.scalars(q))


# --- F1: ensure_admin ---------------------------------------------------------------------------------
def test_ensure_admin_does_not_promote_a_squatted_account(client, monkeypatch):
    """An attacker registers the configured admin address first; a restart must not make them admin."""
    squat = register(client, "squatter-admin@example.com", password="attacker-pass-1")
    s = get_settings()
    monkeypatch.setattr(s, "admin_email", "squatter-admin@example.com")
    with SessionLocal() as db:
        ensure_admin(db)
    with SessionLocal() as db:
        assert db.scalar(select(User.is_admin).where(User.email == "squatter-admin@example.com")) is False
    assert client.get("/admin/stats", headers=squat).status_code == 403


def test_ensure_admin_promotes_only_with_the_configured_password(client, monkeypatch):
    member = register(client, "owner-admin@example.com", password="operator-pass-9")
    s = get_settings()
    monkeypatch.setattr(s, "admin_email", "owner-admin@example.com")
    monkeypatch.setattr(s, "admin_password", type(s.admin_password)("operator-pass-9"))
    with SessionLocal() as db:
        ensure_admin(db)
        uid = db.scalar(select(User.id).where(User.email == "owner-admin@example.com"))
    with SessionLocal() as db:
        assert db.get(User, uid).is_admin is True
    assert _audits("user.role_change", str(uid)), "promotion is audited"
    assert client.get("/users/me", headers=member).status_code == 401, "pre-promotion sessions end"
    fresh = _login(client, "owner-admin@example.com", "operator-pass-9")
    assert client.get("/admin/stats", headers=_bearer(fresh.json()["access_token"])).status_code == 200


# --- F2: revocation -----------------------------------------------------------------------------------
def test_logout_revokes_the_token(client):
    h = register(client, "logout-me@example.com")
    assert client.get("/users/me", headers=h).status_code == 200
    assert client.post("/auth/logout", headers=h).status_code == 204
    assert client.get("/users/me", headers=h).status_code == 401
    assert any(a.actor == "logout-me@example.com" for a in _audits("auth.logout"))


def test_logout_revokes_only_that_session(client):
    register(client, "two-sessions@example.com")
    a = _bearer(_login(client, "two-sessions@example.com", "password-123").json()["access_token"])
    b = _bearer(_login(client, "two-sessions@example.com", "password-123").json()["access_token"])
    assert client.post("/auth/logout", headers=a).status_code == 204
    assert client.get("/users/me", headers=a).status_code == 401
    assert client.get("/users/me", headers=b).status_code == 200


def test_cookie_logout_needs_csrf_header(client):
    register(client, "cookie-logout@example.com")
    r = client.post("/auth/login", json={"email": "cookie-logout@example.com", "password": "password-123"})
    token = r.json()["access_token"]
    try:
        assert client.post("/auth/logout").status_code == 403  # cookie without X-JEV-CSRF
        assert client.post("/auth/logout", headers={"X-JEV-CSRF": "1"}).status_code == 204
    finally:
        client.cookies.clear()
    assert client.get("/users/me", headers=_bearer(token)).status_code == 401


def test_logout_all_revokes_every_session(client):
    register(client, "everywhere@example.com")
    a = _bearer(_login(client, "everywhere@example.com", "password-123").json()["access_token"])
    b = _bearer(_login(client, "everywhere@example.com", "password-123").json()["access_token"])
    assert client.post("/auth/logout-all", headers=a).status_code == 204
    assert client.get("/users/me", headers=a).status_code == 401
    assert client.get("/users/me", headers=b).status_code == 401
    assert any(a.actor == "everywhere@example.com" for a in _audits("auth.revoke_all"))
    assert _login(client, "everywhere@example.com", "password-123").status_code == 200


def test_password_change_revokes_other_sessions(client):
    old = register(client, "rotate@example.com")
    bad = {"current_password": "not-it-at-all", "new_password": "new-password-456"}
    assert client.post("/auth/password", headers=old, json=bad).status_code == 403
    good = {"current_password": "password-123", "new_password": "new-password-456"}
    r = client.post("/auth/password", headers=old, json=good)
    client.cookies.clear()
    assert r.status_code == 200, r.text
    assert client.get("/users/me", headers=old).status_code == 401
    assert client.get("/users/me", headers=_bearer(r.json()["access_token"])).status_code == 200
    assert _login(client, "rotate@example.com", "password-123").status_code == 401
    assert _login(client, "rotate@example.com", "new-password-456").status_code == 200
    assert any(a.actor == "rotate@example.com" for a in _audits("auth.password_change"))


def test_admin_can_end_a_members_sessions(client, admin):
    h = register(client, "incident@example.com")
    uid = client.get("/users/me", headers=h).json()["id"]
    assert client.post(f"/admin/users/{uid}/revoke-sessions", headers=admin).status_code == 204
    assert client.get("/users/me", headers=h).status_code == 401


def test_tokens_without_jti_or_version_are_rejected(client):
    """A token minted before revocation existed (no jti/ver) cannot bypass the checks."""
    register(client, "legacy@example.com")
    with SessionLocal() as db:
        uid = db.scalar(select(User.id).where(User.email == "legacy@example.com"))
    now = datetime.now(UTC)
    s = get_settings()
    base = {"sub": str(uid), "adm": False, "iat": now, "exp": now + timedelta(hours=1), "typ": "access"}
    for claims in (base, {**base, "jti": "x"}, {**base, "ver": 0}, {**base, "jti": "x", "ver": "0"}):
        token = jwt.encode(claims, s.secret_key(), algorithm=s.jwt_algorithm)
        assert client.get("/users/me", headers=_bearer(token)).status_code == 401, claims


# --- F3: service tokens -------------------------------------------------------------------------------
def _create_token(client, admin, **body):
    payload = {"name": "ingest-bot", "scopes": [f"events:ingest:{DOMAIN}"], **body}
    return client.post("/admin/service-tokens", headers=admin, json=payload)


def test_service_token_is_shown_once_and_stored_hashed(client, admin):
    r = _create_token(client, admin)
    assert r.status_code == 201, r.text
    token = r.json()["token"]
    assert token.startswith("jevst_") and len(token) > 40
    listed = client.get("/admin/service-tokens", headers=admin).json()
    mine = next(t for t in listed if t["id"] == r.json()["id"])
    assert "token" not in mine and "token_hash" not in mine
    assert token not in str(listed)
    with SessionLocal() as db:
        row = db.get(ServiceToken, mine["id"])
        assert row.token_hash != token and token not in row.token_hash
    audit = _audits("token.create", str(mine["id"]))
    assert audit and token not in str(audit[0].detail)


def test_service_token_posts_observations_but_reads_nothing_else(client, admin, generic):  # noqa: F811
    token = _create_token(client, admin).json()["token"]
    h = _bearer(token)
    r = client.post(URL, headers=h, json={"observations": [_obs("A", "2021-01-01T00:00:00Z", 5.0)]})
    assert r.status_code == 200 and r.json()["accepted"] == 1, r.text
    entry = _audits("events.ingest")[-1]
    assert entry.actor.startswith("service-token:") and entry.actor_user_id is None
    for path in ("/admin/stats", "/admin/audit", "/intel/runs", "/admin/service-tokens", "/users/me"):
        assert client.get(path, headers=h).status_code == 401, path
    assert client.post("/events", headers=h, json={"events": []}).status_code in (401, 422)
    assert client.post("/events/replay", headers=h, json={}).status_code == 401


def test_service_token_scope_is_enforced(client, admin, generic):  # noqa: F811
    token = _create_token(client, admin, scopes=["events:ingest:generic:other"]).json()["token"]
    body = {"observations": [_obs("A", "2021-02-01T00:00:00Z", 5.0)]}
    assert client.post(URL, headers=_bearer(token), json=body).status_code == 403
    wildcard = _create_token(client, admin, scopes=["events:ingest:*"]).json()["token"]
    assert client.post(URL, headers=_bearer(wildcard), json=body).status_code == 200


def test_service_token_validation(client, admin):
    for scopes in ([], ["admin"], ["events:ingest:movie"], ["events:ingest:generic:../x"], ["*"]):
        assert _create_token(client, admin, scopes=scopes).status_code == 422, scopes
    assert _create_token(client, admin, expires_in_days=0).status_code == 422
    assert _create_token(client, admin, expires_in_days=10_000).status_code == 422
    member = register(client, "token-member@example.com")
    assert _create_token(client, member).status_code == 403


def test_revoked_and_expired_service_tokens_are_rejected(client, admin, generic):  # noqa: F811
    body = {"observations": [_obs("A", "2021-03-01T00:00:00Z", 5.0)]}
    created = _create_token(client, admin).json()
    assert client.delete(f"/admin/service-tokens/{created['id']}", headers=admin).json()["active"] is False
    assert client.post(URL, headers=_bearer(created["token"]), json=body).status_code == 401
    assert _audits("token.revoke", str(created["id"]))
    expiring = _create_token(client, admin).json()
    with SessionLocal() as db:
        db.get(ServiceToken, expiring["id"]).expires_at = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()
    assert client.post(URL, headers=_bearer(expiring["token"]), json=body).status_code == 401
    assert client.post(URL, headers=_bearer("jevst_" + "x" * 43), json=body).status_code == 401


def test_idempotency_scopes_of_tokens_and_users_cannot_collide(client, admin, generic):  # noqa: F811
    """A token must never be answered with a stored response of the admin who has the same id."""
    token = _create_token(client, admin).json()["token"]
    key = {"Idempotency-Key": "collide-1"}
    body = {"observations": [_obs("A", "2021-04-01T00:00:00Z", 5.0)]}
    first = client.post(URL, headers={**admin, **key}, json=body).json()
    second = client.post(URL, headers={**_bearer(token), **key}, json=body).json()
    assert first["batch_id"] != second["batch_id"]


# --- F4: login throttling -----------------------------------------------------------------------------
@pytest.fixture()
def throttle(client, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "security_login_max_failures", 3)
    client.app.state.cache.delete_prefix("rl:")
    yield
    client.app.state.cache.delete_prefix("rl:")


def test_per_account_throttle_blocks_even_the_right_password(client, throttle):
    register(client, "brute@example.com")
    codes = [_login(client, "brute@example.com", f"guess-{i}-xx").status_code for i in range(3)]
    assert codes == [401, 401, 401]
    r = _login(client, "brute@example.com", "password-123")
    assert r.status_code == 429 and int(r.headers["Retry-After"]) > 0
    # other accounts, from the same (shared) client address, are unaffected
    register(client, "bystander@example.com")
    assert _login(client, "bystander@example.com", "password-123").status_code == 200
    # unknown emails are throttled the same way (no account enumeration)
    for i in range(3):
        _login(client, "nobody-here@example.com", f"guess-{i}-xx")
    assert _login(client, "nobody-here@example.com", "whatever-1").status_code == 429
    # audited once per window, not once per attempt
    _login(client, "brute@example.com", "password-123")
    hits = [a for a in _audits("auth.login.throttled") if a.actor == "brute@example.com"]
    assert len(hits) == 1


def test_per_account_throttle_ignores_forged_client_addresses(client, throttle):
    register(client, "spoof-brute@example.com")
    for i in range(3):
        _login(client, "spoof-brute@example.com", f"g-{i}-xxxx", **{"X-Forwarded-For": f"10.0.0.{i}"})
    r = _login(client, "spoof-brute@example.com", "password-123", **{"X-Forwarded-For": "10.9.9.9"})
    assert r.status_code == 429


@pytest.fixture()
def proxy(client, monkeypatch):
    s = client.app.state.settings
    monkeypatch.setattr(s, "security_proxy_secret", type(s.jwt_secret)(PROXY_SECRET))
    monkeypatch.setattr(s, "auth_rate_limit_per_minute", 3)
    client.app.state.cache.delete_prefix("rl:")
    yield
    client.app.state.cache.delete_prefix("rl:")


def _signed(ip: str, secret: str = PROXY_SECRET, ts: int | None = None) -> dict:
    return {CLIENT_HEADER: sign_client_address(ip, secret, ts)}


def test_signed_client_addresses_get_their_own_auth_bucket(client, proxy):
    flood = [_login(client, f"f{i}@example.com", "x-wrong-1", **_signed("203.0.113.7")) for i in range(5)]
    assert [r.status_code for r in flood] == [401, 401, 401, 429, 429]
    # a different signed client behind the same proxy still signs in
    assert _login(client, "admin@example.com", "admin-pass-123", **_signed("198.51.100.2")).status_code == 200


def test_forged_or_stale_client_signatures_fall_back_to_the_peer(client, proxy):
    old = int(datetime.now(UTC).timestamp()) - 3600
    forged = [
        _signed("192.0.2.1", secret="not-the-secret"),
        _signed("192.0.2.2", ts=old),
        {CLIENT_HEADER: "v1:192.0.2.3:1:" + "0" * 64},
        {CLIENT_HEADER: "garbage"},
        {"X-Forwarded-For": "192.0.2.9"},
    ]
    codes = [
        _login(client, f"forge{i}@example.com", "x-wrong-1", **h).status_code for i, h in enumerate(forged)
    ]
    # all five share the peer's bucket (limit 3): a forged header never opens a fresh bucket
    assert codes == [401, 401, 401, 429, 429]


def test_signed_addresses_are_audited(client, admin, proxy):
    _login(client, "audited-client@example.com", "x-wrong-1", **_signed("2001:db8::5"))
    entry = [a for a in _audits("auth.login.failure") if a.actor == "audited-client@example.com"][-1]
    assert entry.detail["client"] == "2001:db8::5"


def test_public_health_ml_hides_the_model_load_error(client, monkeypatch):
    """F6: the public /health/ml never echoes the load exception (it can carry file paths)."""
    holder = client.app.state.engines
    monkeypatch.setattr(holder, "_engine", None)
    monkeypatch.setattr(holder, "poll_seconds", 0.0)
    monkeypatch.setattr(holder, "last_error", "failed to load model x: [Errno 2] /secret/path/models/x")
    r = client.get("/health/ml")
    assert r.status_code == 503
    assert r.json() == {"status": "unavailable", "error": "no recommendation model is loaded"}
