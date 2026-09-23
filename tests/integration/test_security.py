"""Security regression tests (SECURITY.md, "Intelligence layer review").

Authorization over every registered route, CSRF on cookie-authenticated intel writes, input
bounds (oversized ids, non-ASCII digits, NaN/Infinity, huge strings, out-of-range dates), LIKE
wildcard escaping, sanitised run errors and the generic 500 envelope.

Runs after test_intel_api.py / test_intel_v11.py in the same session database.
"""

from __future__ import annotations

import logging

import pytest
from conftest import admin_headers, register
from fastapi.routing import APIRoute
from test_intel_api import synthetic_frames

from jev_api.deps import require_admin
from jev_ml.intel import PipelineInputs

HUGE_INT = "9" * 30  # beyond SQLite/PostgreSQL BIGINT


@pytest.fixture(scope="module")
def intel(client):
    inter, movies = synthetic_frames()
    svc = client.app.state.intel
    original = svc.inputs_factory

    def factory(as_of):
        return PipelineInputs(
            interactions=inter.copy(),
            movies=movies.copy(),
            dataset_meta={"dataset_version": "synthetic-intel-v1"},
            as_of=as_of,
        )

    svc.inputs_factory = factory
    yield svc
    svc.inputs_factory = original


@pytest.fixture(scope="module")
def admin(client):
    return admin_headers(client)


@pytest.fixture(scope="module")
def member(client):
    return register(client, "security-member@example.com")


@pytest.fixture(scope="module")
def latest_run(client, intel, admin):
    r = client.post("/intel/runs", headers=admin, json={})
    assert r.status_code == 200 and r.json()["status"] == "succeeded", r.text
    return r.json()


def _api_routes(app) -> list:
    """Every API route with its full path. FastAPI >= 0.140 nests included routers
    (fastapi.routing._IncludedRouter), so app.routes alone no longer lists them."""
    try:
        from fastapi.routing import iter_route_contexts
    except ImportError:  # older FastAPI: included routes are flattened into app.routes
        return [r for r in app.routes if isinstance(r, APIRoute)]
    return [rc for rc in iter_route_contexts(app.routes) if isinstance(rc.original_route, APIRoute)]


def _dependency_calls(route) -> set:
    calls, stack = set(), [route.dependant]
    while stack:
        d = stack.pop()
        if d.call is not None:
            calls.add(d.call)
        stack.extend(d.dependencies)
    return calls


def _admin_routes(app) -> list:
    return [r for r in _api_routes(app) if r.path.startswith(("/intel", "/admin"))]


def _concrete(path: str) -> str:
    """A request path for a route template (placeholders filled with harmless values)."""
    return (
        path.replace("{series_id:path}", "volume:all")
        .replace("{entity}", "signals")
        .replace("{warning_id}", "1")
        .replace("{decision_ref}", "1")
        .replace("{run_ref}", "1")
    )


def test_every_intel_and_admin_route_requires_admin(client):
    """Fails when a future /intel/* or /admin/* route forgets the AdminUser dependency."""
    routes = _admin_routes(client.app)
    assert len(routes) >= 25  # the enumeration really sees the intel router
    missing = [f"{sorted(r.methods)} {r.path}" for r in routes if require_admin not in _dependency_calls(r)]
    assert not missing, f"routes without the admin guard: {missing}"


def test_admin_routes_reject_anonymous_and_members(client, member):
    for r in _admin_routes(client.app):
        path = _concrete(r.path) + ("?key=x" if "/history/" in r.path else "")
        for method in r.methods:
            anon = client.request(method, path, json={})
            assert anon.status_code == 401, (method, path, anon.status_code)
            m = client.request(method, path, headers=member, json={})
            assert m.status_code == 403, (method, path, m.status_code)


def test_other_admin_routes_are_guarded(client):
    """Admin-only routes outside /intel and /admin (models, experiments, POST /movies)."""
    guarded = {
        (m, r.path)
        for r in _api_routes(client.app)
        if require_admin in _dependency_calls(r)
        for m in r.methods
    }
    for expected in (
        ("GET", "/models"),
        ("GET", "/models/{model_id}"),
        ("POST", "/models/{model_id}/activate"),
        ("GET", "/experiments"),
        ("GET", "/experiments/{experiment_id}"),
        ("POST", "/movies"),
    ):
        assert expected in guarded, expected


def test_cookie_writes_under_intel_need_csrf_header(client, intel, latest_run):
    r = client.post("/auth/login", json={"email": "admin@example.com", "password": "admin-pass-123"})
    assert r.status_code == 200 and client.cookies.get("jev_session")
    try:
        writes = (
            ("POST", "/intel/runs", {}),
            ("PATCH", "/intel/warnings/1", {"status": "acknowledged"}),
            ("POST", "/intel/feedback", {"target_type": "warning", "target_id": "1", "verdict": "useful"}),
            ("POST", "/intel/scenarios", {"series_id": "volume:all"}),
        )
        for method, path, body in writes:
            resp = client.request(method, path, json=body)
            assert resp.status_code == 403 and resp.json()["detail"] == "missing CSRF header", path
        # reads with the cookie need no header; a write with it gets past the CSRF check
        assert client.get("/intel/status").status_code == 200
        ok = client.post("/intel/scenarios", json={"series_id": "volume:all"}, headers={"X-JEV-CSRF": "1"})
        assert ok.status_code == 200, ok.text
    finally:
        client.cookies.clear()


@pytest.mark.parametrize(
    "path",
    [
        f"/intel/runs/{HUGE_INT}",
        "/intel/runs/%C2%B2",  # "²": str.isdigit() is true, int() is not
        f"/intel/decisions/{HUGE_INT}",
        "/intel/decisions/%D9%A3",  # Arabic-Indic digit three
        f"/intel/warnings/{HUGE_INT}",
        f"/movies/{HUGE_INT}",
        f"/recommendations/similar/{HUGE_INT}",
    ],
)
def test_oversized_or_unicode_ids_are_client_errors(client, intel, admin, latest_run, path):
    r = client.get(path, headers=admin)
    assert r.status_code in (404, 422), (path, r.status_code, r.text)
    assert set(r.json()) >= {"detail", "request_id"}


def test_feedback_with_oversized_warning_id_is_404(client, intel, admin, latest_run):
    for tid in (HUGE_INT[:64], "²"):
        r = client.post(
            "/intel/feedback",
            headers=admin,
            json={"target_type": "warning", "target_id": tid, "verdict": "useful"},
        )
        assert r.status_code == 404, (tid, r.text)


def test_unbounded_strings_are_rejected(client, intel, admin, latest_run):
    long = "x" * 5000
    assert client.get(f"/intel/series/{long}", headers=admin).status_code == 422
    assert client.get(f"/intel/signals?run_id={long}", headers=admin).status_code == 422
    assert client.get(f"/intel/predictions?series_id={long}", headers=admin).status_code == 422
    assert client.get(f"/intel/history/signals?key={long}", headers=admin).status_code == 422
    assert client.get("/intel/history/../../etc/passwd?key=x", headers=admin).status_code == 404
    assert client.get("/intel/history/passwd?key=x", headers=admin).status_code == 422
    r = client.patch("/intel/warnings/1", headers=admin, json={"status": "acknowledged", "note": long})
    assert r.status_code == 422
    r = client.post(
        "/intel/feedback",
        headers=admin,
        json={"target_type": "warning", "target_id": "1", "verdict": "useful", "note": long},
    )
    assert r.status_code == 422
    r = client.post("/intel/runs", headers=admin, json={"as_of": long})
    assert r.status_code == 422
    r = client.post("/intel/scenarios", headers=admin, json={"series_id": "volume:all", "title": long})
    assert r.status_code == 422


def test_scenario_rejects_non_finite_and_out_of_range_numbers(client, intel, admin, latest_run):
    base = '{"series_id": "volume:all", "scenarios": [{"kind": "custom", "trend_multiplier": %s}]}'
    for value in ("NaN", "Infinity", "-Infinity", "1e308"):
        r = client.post(
            "/intel/scenarios",
            headers={**admin, "Content-Type": "application/json"},
            content=base % value,
        )
        assert r.status_code == 422, (value, r.text)
    for body in (
        {"series_id": "volume:all", "horizon_months": 10**6},
        {"series_id": "volume:all", "horizon_months": 0},
        {"series_id": "volume:all", "scenarios": [{"kind": "continue"}] * 50},
        {"series_id": "volume:all", "scenarios": [{"kind": "custom", "shock_month": 10**9}]},
    ):
        assert client.post("/intel/scenarios", headers=admin, json=body).status_code == 422, body
    # unknown keys are dropped, not stored with a saved scenario
    r = client.post(
        "/intel/scenarios",
        headers=admin,
        json={
            "series_id": "volume:all",
            "save": True,
            "scenarios": [{"kind": "continue", "junk": "y" * 10_000}],
        },
    )
    assert r.status_code == 200, r.text
    saved = client.get("/intel/scenarios?limit=1", headers=admin).json()["items"][0]
    assert saved["id"] == r.json()["id"] and "junk" not in saved["input"]["scenarios"][0]


@pytest.mark.parametrize(
    "as_of", ["0001-01-01T00:00:00+05:00", "9999-12-31T23:00:00-05:00", "2016-13-40", "２０１６-01-01"]
)
def test_as_of_edge_cases_are_422(client, intel, admin, as_of):
    r = client.post("/intel/runs", headers=admin, json={"as_of": as_of})
    assert r.status_code == 422, (as_of, r.text)


def test_evidence_search_escapes_like_wildcards(client, intel, admin, latest_run):
    everything = client.get("/intel/evidence?limit=1", headers=admin).json()["total"]
    assert everything > 0
    for q in ("%", "_", "%%", "\\", "' OR 1=1 --"):
        r = client.get("/intel/evidence", headers=admin, params={"q": q})
        assert r.status_code == 200 and r.json()["total"] < everything, q


def test_failed_run_error_hides_paths_and_sql(client, intel, admin):
    good = intel.inputs_factory

    def broken(as_of):
        raise FileNotFoundError(2, "No such file or directory", "/home/someone/secret/data/processed/x.csv")

    intel.inputs_factory = broken
    try:
        r = client.post("/intel/runs", headers=admin, json={})
    finally:
        intel.inputs_factory = good
    body = r.json()
    assert r.status_code == 200 and body["status"] == "failed"
    assert "/home/someone" not in body["error"] and "x.csv" in body["error"]

    def sql_failure(*args, **kwargs):
        from sqlalchemy.exc import IntegrityError

        raise IntegrityError(
            "INSERT INTO intel_signals (run_id) VALUES (?)", {"run_id": 1}, Exception("boom")
        )

    original = intel._persist_success
    intel._persist_success = sql_failure
    try:
        r = client.post("/intel/runs", headers=admin, json={})
    finally:
        intel._persist_success = original
    body = r.json()
    assert body["status"] == "failed" and "INSERT INTO" not in body["error"]
    assert "[SQL" not in body["error"] and "IntegrityError" in body["error"]


def test_unhandled_errors_return_generic_envelope(client, intel, admin, monkeypatch, caplog):
    def explode(*args, **kwargs):
        raise RuntimeError("secret internals at /srv/jev/app.py line 1")

    monkeypatch.setattr(intel, "open_warning_counts", explode)
    with caplog.at_level(logging.ERROR, logger="jev_api"):
        r = client.get("/intel/status", headers=admin)
    assert r.status_code == 500
    assert r.json()["detail"] == "internal server error" and r.json()["request_id"]
    assert "secret internals" not in r.text
    assert any("unhandled error" in rec.getMessage() for rec in caplog.records)


def test_no_endpoint_serves_html(client, intel, admin, latest_run):
    note = "<script>alert(1)</script>"
    r = client.post(
        "/intel/scenarios",
        headers=admin,
        json={"series_id": "volume:all", "save": True, "title": note},
    )
    assert r.status_code == 200
    listing = client.get("/intel/scenarios?limit=1", headers=admin)
    assert listing.headers["content-type"].startswith("application/json")
    assert listing.json()["items"][0]["title"] == note  # stored and returned verbatim, as data
    assert listing.headers["x-content-type-options"] == "nosniff"
    routes = _api_routes(client.app)
    assert len(routes) > 50
    for r in routes:
        cls = getattr(r.original_route, "response_class", None)
        assert "HTML" not in getattr(cls, "__name__", repr(cls)), r.path


def test_manual_runs_are_rate_limited(client, intel, admin):
    settings = client.app.state.settings
    original = settings.intel_run_rate_limit_per_minute
    settings.intel_run_rate_limit_per_minute = 1
    client.app.state.cache.delete_prefix("rl:")
    try:
        first = client.post("/intel/runs", headers=admin, json={})
        second = client.post("/intel/runs", headers=admin, json={})
    finally:
        settings.intel_run_rate_limit_per_minute = original
        client.app.state.cache.delete_prefix("rl:")
    assert first.status_code == 200
    assert second.status_code == 429 and second.headers["Retry-After"]
    assert client.get("/intel/status", headers=admin).status_code == 200  # reads are not affected


def test_oversized_ids_in_bodies_and_pages_are_422(client, intel, admin):
    user = register(client, "security-bodies@example.com")
    huge = 10**30
    for path, body in (
        ("/recommendations/feedback", {"movie_id": huge, "feedback": "like"}),
        ("/recommendations/feedback", {"movie_id": 1, "feedback": "like", "recommendation_id": huge}),
        ("/users/me/onboarding", {"genres": ["Drama"], "movie_ids": [huge]}),
    ):
        assert client.post(path, headers=user, json=body).status_code == 422, (path, body)
    body = {"id": 2**40, "title": "Too big"}
    assert client.post("/movies", headers=admin, json=body).status_code == 422
    for path in (
        f"/movies?page={huge}",
        f"/movies?min_ratings={huge}",
        f"/users/me/ratings?page={huge}",
        f"/intel/runs?offset={huge}",
        f"/admin/audit?offset={huge}",
        f"/recommendations?min_ratings={huge}",
    ):
        r = client.get(path, headers=admin if path.startswith(("/intel", "/admin")) else user)
        assert r.status_code == 422, (path, r.status_code)


def test_evaluation_report_with_nan_is_served_as_null(client, intel, admin):
    from conftest import TMP_ROOT

    d = TMP_ROOT / "experiments" / "intel-eval-29990101T000000Z"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text('{"pipeline_version": "x", "latency": {"pipeline_ms_mean": NaN}}')
    try:
        r = client.get("/intel/evaluation", headers=admin)
        assert r.status_code == 200 and r.json()["report"]["latency"]["pipeline_ms_mean"] is None
        assert r.json()["run_dir"] == d.name  # a directory name, never a filesystem path
    finally:
        (d / "report.json").unlink()
        d.rmdir()
