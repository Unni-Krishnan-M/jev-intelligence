from conftest import admin_headers, register


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"
    ml = client.get("/health/ml").json()
    assert ml["status"] == "ok" and ml["self_check"]["recommendations"] == 5


def test_auth_flow_and_validation(client):
    h = register(client, "alice@example.com", "Alice")
    me = client.get("/users/me", headers=h).json()
    assert me["email"] == "alice@example.com" and not me["is_admin"]
    assert (
        client.post(
            "/auth/register",
            json={"email": "alice@example.com", "password": "password-123", "display_name": "A"},
        ).status_code
        == 409
    )
    assert (
        client.post("/auth/login", json={"email": "alice@example.com", "password": "wrong-pass"}).status_code
        == 401
    )
    assert (
        client.post("/auth/login", json={"email": "nobody@example.com", "password": "x"}).status_code == 401
    )
    assert (
        client.post(
            "/auth/register", json={"email": "bad", "password": "password-123", "display_name": "A"}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/auth/register", json={"email": "b@example.com", "password": "short", "display_name": "B"}
        ).status_code
        == 422
    )
    assert client.get("/users/me").status_code == 401
    assert client.get("/users/me", headers={"Authorization": "Bearer not-a-jwt"}).status_code == 401


def test_password_is_hashed(client):
    from sqlalchemy import select

    from jev_api.db import SessionLocal
    from jev_api.models import User

    register(client, "hash@example.com")
    with SessionLocal() as db:
        u = db.scalar(select(User).where(User.email == "hash@example.com"))
        assert u.password_hash.startswith("$argon2id$") and "password-123" not in u.password_hash


def test_catalog_endpoints(client):
    page = client.get("/movies?page=1&page_size=10&genre=Horror").json()
    assert page["total"] > 0 and all("Horror" in m["genres"] for m in page["items"])
    assert client.get("/movies/1").json()["title"]
    assert client.get("/movies/424242").status_code == 404
    found = client.get("/movies/search?q=sci-fi story").json()
    assert found and all("sci-fi story" in m["title"].lower() for m in found)
    assert client.get("/movies/search?q=%25").json() == []  # LIKE wildcards are escaped
    assert client.get("/movies/search?q=' OR 1=1 --").status_code == 200  # bound parameter, no injection
    assert len(client.get("/genres").json()) >= 4


def test_interactions_and_recommendations(client):
    h = register(client, "bob@example.com", "Bob")
    assert client.post(
        "/users/me/onboarding", headers=h, json={"genres": ["Sci-Fi"], "movie_ids": [1]}
    ).json()["onboarding_completed"]
    for mid, val in ((2, 5.0), (3, 4.5), (40, 1.0)):
        assert client.post(f"/movies/{mid}/rate", headers=h, json={"rating": val}).status_code == 200
    assert (
        client.post("/movies/2/rate", headers=h, json={"rating": 4.2}).status_code == 422
    )  # half steps only
    assert client.post("/movies/4/watch", headers=h).json()["watched"]
    assert client.post("/movies/5/favorite", headers=h, json={"favorite": True}).json()["is_favorite"]
    detail = client.get("/movies/2", headers=h).json()
    assert detail["user_rating"] == 5.0

    r = client.get("/recommendations?limit=8", headers=h)
    assert r.status_code == 200
    body = r.json()
    ids = [i["movie_id"] for i in body["items"]]
    assert len(ids) == 8 and not set(ids) & {1, 2, 3, 4, 5, 40}
    assert body["model_version"] and body["generated_at"]
    assert all(i["reason"] and set(i["signals"]) >= {"content", "latent"} for i in body["items"])
    assert abs(sum(body["effective_weights"].values()) - 1) < 1e-3
    # second identical request is served from cache
    assert client.get("/recommendations?limit=8", headers=h).json()["cached"] is True
    # pagination
    p2 = client.get("/recommendations?limit=4&offset=4", headers=h).json()
    assert [i["movie_id"] for i in p2["items"]] == ids[4:8]

    # feedback persisted, and "not interested" removes the movie from future lists
    target = body["items"][0]
    fb = client.post(
        "/recommendations/feedback",
        headers=h,
        json={
            "movie_id": target["movie_id"],
            "feedback": "not_interested",
            "recommendation_id": target["recommendation_id"],
        },
    )
    assert fb.status_code == 201
    again = [i["movie_id"] for i in client.get("/recommendations?limit=8", headers=h).json()["items"]]
    assert target["movie_id"] not in again
    hist = client.get("/recommendations/history?limit=50", headers=h).json()
    assert any(x["id"] == target["recommendation_id"] and x["feedback"] == "not_interested" for x in hist)

    profile = client.get("/users/me/profile", headers=h).json()
    assert profile["counts"] == {"ratings": 3, "favorites": 2, "watches": 1, "negative_feedback": 1}
    assert profile["explicit_genres"] == ["Sci-Fi"]
    assert client.get("/users/me/history", headers=h).json()["total"] == 1


def test_other_users_recommendation_id_rejected(client):
    h1 = register(client, "carol@example.com")
    h2 = register(client, "dave@example.com")
    rec = client.get("/recommendations?limit=1", headers=h1).json()["items"][0]
    r = client.post(
        "/recommendations/feedback",
        headers=h2,
        json={"movie_id": rec["movie_id"], "feedback": "like", "recommendation_id": rec["recommendation_id"]},
    )
    assert r.status_code == 404


def test_similar_trending_sections(client):
    assert len(client.get("/recommendations/similar/1?limit=5").json()["items"]) == 5
    assert client.get("/recommendations/similar/999999").status_code == 404
    assert client.get("/recommendations/trending?limit=5").json()["items"]
    assert client.get("/movies/popular?limit=5").status_code == 200
    h = register(client, "erin@example.com")
    assert client.get("/recommendations/because-you-watched", headers=h).json()["items"] == []
    client.post("/movies/1/watch", headers=h)
    byw = client.get("/recommendations/because-you-watched", headers=h).json()
    assert byw["anchor"]["id"] == 1 and byw["anchor_kind"] == "watched" and byw["items"]


def test_admin_endpoints(client):
    user = register(client, "frank@example.com")
    assert client.get("/models", headers=user).status_code == 403
    assert client.get("/experiments", headers=user).status_code == 403
    adm = admin_headers(client)
    models = client.get("/models", headers=adm).json()
    assert len(models) >= 2 and sum(m["is_active"] for m in models) == 1
    detail = client.get(f"/models/{models[0]['id']}", headers=adm).json()
    assert "components" in detail["manifest"]
    exps = client.get("/experiments", headers=adm).json()
    assert exps and "ndcg@10" in exps[0]["headline"]
    exp = client.get(f"/experiments/{exps[0]['id']}", headers=adm).json()
    names = {(m["model_name"], m["metric"], m["k"]) for m in exp["metrics"]}
    assert ("hybrid", "f1", 10) in names and ("hybrid", "ndcg", 10) in names
    stats = client.get("/admin/stats", headers=adm).json()
    assert stats["users"] >= 1
    summary = client.get("/models/active/summary").json()
    assert summary["model_version"] and summary["n_items"] > 0 and isinstance(summary["comparison"], dict)

    # switch the active model and back
    inactive = next(m for m in models if not m["is_active"])
    active = next(m for m in models if m["is_active"])
    assert client.post(f"/models/{inactive['id']}/activate", headers=adm).json()["is_active"]
    assert client.get("/health/ml").json()["model_version"] == inactive["version"]
    assert client.post(f"/models/{active['id']}/activate", headers=adm).status_code == 200


def test_cookie_auth_requires_csrf_header(client):
    r = client.post(
        "/auth/register", json={"email": "gina@example.com", "password": "password-123", "display_name": "G"}
    )
    assert r.status_code == 201 and "jev_session" in r.cookies
    try:
        assert client.get("/users/me").status_code == 200  # cookie works for reads
        assert client.post("/movies/1/watch").status_code == 403
        assert client.post("/movies/1/watch", headers={"X-JEV-CSRF": "1"}).status_code == 200
    finally:
        client.cookies.clear()


def test_security_headers_and_errors(client):
    r = client.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-request-id"]
    bad = client.get("/movies/not-a-number")
    assert bad.status_code == 422 and "request_id" in bad.json()


def test_rate_limit():
    from fastapi.testclient import TestClient

    from jev_api.config import Settings
    from jev_api.main import create_app

    app = create_app(Settings(auth_rate_limit_per_minute=3, auto_migrate=False))
    with TestClient(app) as c:
        codes = [
            c.post("/auth/login", json={"email": "x@example.com", "password": "y"}).status_code
            for _ in range(5)
        ]
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]


def test_secret_redaction():
    from jev_api.logging_setup import redact

    out = redact(
        {
            "password": "hunter2",
            "nested": {"api_key": "k"},
            "msg": "Bearer abc.def.ghi",
            "url": "postgresql://jev:s3cret@db:5432/jev",
        }
    )
    assert out["password"] == "[REDACTED]" and out["nested"]["api_key"] == "[REDACTED]"
    assert "abc.def" not in out["msg"] and "s3cret" not in out["url"]
