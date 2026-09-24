"""Intelligence API (/intel/*, /admin/metrics) against a synthetic dataset with a known spike.

The service's inputs loader is pointed at a 6-genre, 6-year synthetic rating stream whose last
complete month (2017-12) carries a 6x Horror spike, so the full pipeline runs in well under a
second and reliably raises the warning `anomaly:series_spike:share:genre:Horror`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from conftest import TMP_ROOT, admin_headers, register

from jev_ml.intel import PipelineInputs

GENRES = ["Drama", "Comedy", "Horror", "Sci-Fi", "Action", "Romance"]
SPIKE_KEY = "anomaly:series_spike:share:genre:Horror"


def synthetic_frames(seed: int = 11, months: int = 72) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    n_movies = 120  # covers the ids of the API test catalogue, so app ratings validate
    movies = pd.DataFrame(
        {
            "movie_id": np.arange(1, n_movies + 1),
            "title": [f"Film {i}" for i in range(1, n_movies + 1)],
            "genres": [GENRES[(i - 1) % len(GENRES)] for i in range(1, n_movies + 1)],
        }
    )
    start = pd.Timestamp("2012-01-01", tz="UTC")
    rows = []
    for m in range(months + 1):
        m0 = (start + pd.DateOffset(months=m)).timestamp()
        m1 = (start + pd.DateOffset(months=m + 1)).timestamp()
        if m == months:
            m1 = m0 + 14 * 86400  # the final month is partial
        for g in GENRES:
            n = int(rng.poisson(90)) * (6 if g == "Horror" and m == months - 1 else 1)
            mids = movies.movie_id[movies.genres == g].to_numpy()
            for _ in range(n):
                rows.append(
                    (
                        int(rng.integers(1, 301)),
                        int(rng.choice(mids)),
                        float(rng.choice([2.5, 3.0, 3.5, 4.0, 4.5])),
                        int(rng.uniform(m0, m1 - 1)),
                    )
                )
    df = pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])
    df = df.drop_duplicates(["user_id", "movie_id", "timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True), movies


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


def run(client, admin, **body):
    r = client.post("/intel/runs", headers=admin, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "succeeded", r.json()["error"]
    return r.json()


def spike_warnings(client, admin):
    return client.get(f"/intel/warnings?key={SPIKE_KEY}&limit=200", headers=admin).json()["items"]


def test_auth_required(client, intel):
    assert client.get("/intel/status").status_code == 401
    assert client.get("/admin/metrics").status_code == 401
    user = register(client, "intel-member@example.com")
    for path in ("/intel/status", "/intel/runs", "/intel/warnings", "/admin/metrics"):
        assert client.get(path, headers=user).status_code == 403, path
    assert client.post("/intel/runs", headers=user, json={}).status_code == 403


def test_no_run_yet(client, intel, admin):
    for path in ("/intel/signals", "/intel/trends", "/intel/anomalies", "/intel/risks", "/intel/actions"):
        r = client.get(path, headers=admin)
        assert r.status_code == 404 and r.json()["detail"] == "no intelligence run yet", path
    assert client.get("/intel/predictions", headers=admin).status_code == 404
    st = client.get("/intel/status", headers=admin).json()
    assert st["latest_run"] is None and st["health"]["pipeline"] == "never_run"
    assert client.get("/intel/runs", headers=admin).json() == {"items": [], "total": 0}


def test_run_is_persisted(client, intel, admin):
    body = run(client, admin)
    assert body["trigger"] == "manual" and body["as_of"].startswith("2018-01-14")
    assert body["duration_ms"] > 0 and body["finished_at"] and body["error"] is None
    assert body["summary"]["counts"]["warnings"] >= 1
    assert {"load_inputs", "forecast", "persist"} <= set(body["stage_ms"])
    assert body["data_version"] == "synthetic-intel-v1"
    runs = client.get("/intel/runs", headers=admin).json()
    assert runs["total"] == 1 and runs["items"][0]["run_id"] == body["run_id"]
    assert client.get(f"/intel/runs/{body['run_id']}", headers=admin).json()["id"] == body["id"]
    assert client.get(f"/intel/runs/{body['id']}", headers=admin).json()["run_id"] == body["run_id"]
    assert client.get("/intel/runs/nope", headers=admin).status_code == 404
    (w,) = spike_warnings(client, admin)
    assert w["status"] == "new" and w["occurrences"] == 1 and w["severity"] == "critical"
    assert w["first_seen_run_id"] == body["run_id"] == w["last_seen_run_id"]


def test_list_endpoints_filters_and_pagination(client, intel, admin):
    sig = client.get("/intel/signals?limit=3", headers=admin).json()
    assert set(sig) == {"items", "total", "limit", "offset", "run_id", "as_of", "domain", "mode"}
    assert sig["mode"] == "live"
    assert len(sig["items"]) == 3 and sig["total"] > 3 and sig["limit"] == 3 and sig["run_id"]
    page2 = client.get("/intel/signals?limit=3&offset=3", headers=admin).json()
    assert {s["id"] for s in page2["items"]}.isdisjoint({s["id"] for s in sig["items"]})
    kinds = client.get("/intel/signals?kind=anomaly&limit=200", headers=admin).json()
    assert kinds["items"] and all(s["kind"] == "anomaly" for s in kinds["items"])
    et = client.get("/intel/signals?entity_type=genre&limit=200", headers=admin).json()
    assert all(s["entity_type"] == "genre" for s in et["items"])
    down = client.get("/intel/trends?direction=down&limit=200", headers=admin).json()
    assert all(t["direction"] == "down" for t in down["items"])
    assert client.get("/intel/trends?direction=sideways", headers=admin).status_code == 422
    assert client.get("/intel/signals?limit=0", headers=admin).status_code == 422
    assert client.get("/intel/signals?limit=201", headers=admin).status_code == 422
    crit = client.get("/intel/anomalies?severity=critical", headers=admin).json()
    assert crit["total"] >= 1 and all(a["severity"] == "critical" for a in crit["items"])
    assert client.get("/intel/risks", headers=admin).status_code == 200
    acts = client.get("/intel/actions", headers=admin).json()
    assert acts["total"] >= 1 and acts["items"][0]["id"].startswith("act-")
    assert client.get("/intel/signals?run_id=unknown", headers=admin).status_code == 404

    pred = client.get("/intel/predictions", headers=admin).json()
    assert pred["forecasts"] and pred["lapse"]["status"] in ("ok", "insufficient_data")
    series = client.get("/intel/series/share:genre:Horror", headers=admin).json()
    assert series["series"]["id"] == "share:genre:Horror" and series["series"]["points"]
    assert any(a["kind"] == "series_spike" for a in series["anomalies"])
    encoded = client.get("/intel/series/volume%3Aall", headers=admin)
    assert encoded.status_code == 200 and encoded.json()["forecast"]["series_id"] == "volume:all"
    assert client.get("/intel/series/volume:genre:Nope", headers=admin).status_code == 404


def test_warning_dedup_and_lifecycle(client, intel, admin):
    run(client, admin)
    (w,) = spike_warnings(client, admin)  # no duplicate for the same key
    assert w["occurrences"] == 2 and w["first_seen_run_id"] != w["last_seen_run_id"]
    wid = w["id"]

    r = client.patch(
        f"/intel/warnings/{wid}", headers=admin, json={"status": "acknowledged", "note": "on it"}
    )
    assert r.status_code == 200 and r.json()["status"] == "acknowledged"
    assert client.patch(f"/intel/warnings/{wid}", headers=admin, json={"status": "new"}).status_code == 409
    assert client.patch(f"/intel/warnings/{wid}", headers=admin, json={"status": "bogus"}).status_code == 422
    assert (
        client.patch("/intel/warnings/999999", headers=admin, json={"status": "resolved"}).status_code == 404
    )
    r = client.patch(f"/intel/warnings/{wid}", headers=admin, json={"status": "resolved"})
    assert r.json()["status"] == "resolved"
    assert (
        client.patch(f"/intel/warnings/{wid}", headers=admin, json={"status": "dismissed"}).status_code == 409
    )

    hist = client.get(f"/intel/warnings/{wid}", headers=admin).json()["history"]
    assert [(h["from_status"], h["to_status"]) for h in hist] == [
        (None, "new"),
        ("new", "acknowledged"),
        ("acknowledged", "resolved"),
    ]
    assert hist[0]["actor"] == "system" and hist[1]["actor"] == "admin@example.com"
    assert hist[1]["note"] == "on it"

    # resolved key fires again -> a new warning pointing at the old one
    run(client, admin)
    ws = spike_warnings(client, admin)
    assert len(ws) == 2
    reopened = next(x for x in ws if x["id"] != wid)
    assert reopened["status"] == "new" and reopened["reopened_from"] == wid and reopened["occurrences"] == 1
    assert (
        "reopened"
        in client.get(f"/intel/warnings/{reopened['id']}", headers=admin).json()["history"][0]["note"]
    )

    # dismissed (false positive) key stays suppressed on the next run
    r = client.patch(f"/intel/warnings/{reopened['id']}", headers=admin, json={"status": "dismissed"})
    assert r.json()["status"] == "dismissed" and r.json()["suppressed_until"]
    run(client, admin)
    ws = spike_warnings(client, admin)
    assert len(ws) == 2 and all(x["status"] in ("resolved", "dismissed") for x in ws)
    anoms = client.get("/intel/anomalies?kind=series_spike&suppressed=true&limit=200", headers=admin).json()
    assert any(a["dedup_key"] == SPIKE_KEY for a in anoms["items"])

    listing = client.get("/intel/warnings?status=dismissed", headers=admin).json()
    assert set(listing) == {"items", "total", "limit", "offset"}
    assert all(x["status"] == "dismissed" for x in listing["items"]) and listing["total"] >= 1


def test_escalation_after_dismissal_reopens(client, intel, admin):
    from sqlalchemy import select

    from jev_api.db import SessionLocal
    from jev_api.models import IntelWarning

    with SessionLocal() as db:
        w = db.scalar(
            select(IntelWarning).where(IntelWarning.key == SPIKE_KEY, IntelWarning.status == "dismissed")
        )
        w.dismissed_severity = "low"  # as if it had been dismissed while still low
        db.commit()
        dismissed_id = w.id
    run(client, admin)
    ws = spike_warnings(client, admin)
    fresh = [x for x in ws if x["status"] == "new"]
    assert len(fresh) == 1 and fresh[0]["reopened_from"] == dismissed_id
    assert (
        "escalated"
        in client.get(f"/intel/warnings/{fresh[0]['id']}", headers=admin).json()["history"][0]["note"]
    )

    # an expired suppression window also lets the key fire again
    client.patch(f"/intel/warnings/{fresh[0]['id']}", headers=admin, json={"status": "dismissed"})
    with SessionLocal() as db:
        w = db.get(IntelWarning, fresh[0]["id"])
        w.suppressed_until = datetime.now(UTC) - timedelta(days=1)
        db.commit()
    run(client, admin)
    assert sum(x["status"] == "new" for x in spike_warnings(client, admin)) == 1


def test_open_warning_unique_per_key_in_db(client, intel):
    from sqlalchemy import select
    from sqlalchemy.exc import IntegrityError

    from jev_api.db import SessionLocal
    from jev_api.models import IntelWarning

    with SessionLocal() as db:
        existing = db.scalar(
            select(IntelWarning).where(IntelWarning.key == SPIKE_KEY, IntelWarning.status == "new")
        )
        db.add(IntelWarning(key=existing.key, title="dup", severity="low", status="acknowledged"))
        with pytest.raises(IntegrityError):
            db.commit()


def test_decisions_and_feedback(client, intel, admin):
    decs = client.get("/intel/decisions?limit=5", headers=admin).json()
    assert set(decs) == {"items", "total", "limit", "offset"} and decs["total"] > 5
    d = decs["items"][0]
    assert d["id"].startswith("dec-") and d["db_id"] and d["run_id"] and d["as_of"] and d["created_at"]
    assert d["feedback"] == {"correct": 0, "incorrect": 0} and "state" in d and "option_scores" in d
    retrain = client.get("/intel/decisions?key=retrain_model", headers=admin).json()
    assert retrain["items"] and all(x["key"] == "retrain_model" for x in retrain["items"])

    fb = {"target_type": "decision", "target_id": d["id"], "verdict": "correct", "note": "agree"}
    r = client.post("/intel/feedback", headers=admin, json=fb)
    assert r.status_code == 201 and r.json()["actor"] == "admin@example.com"
    assert (
        client.post("/intel/feedback", headers=admin, json={**fb, "verdict": "incorrect"}).status_code == 201
    )
    one = client.get(f"/intel/decisions/{d['db_id']}", headers=admin).json()
    assert one["feedback"] == {"correct": 1, "incorrect": 1}
    assert client.get(f"/intel/decisions/{d['id']}", headers=admin).json()["id"] == d["id"]
    assert client.get("/intel/decisions/999999", headers=admin).status_code == 404

    # target validation
    bad = [
        {
            "target_type": "decision",
            "target_id": str(d["db_id"]),
            "verdict": "correct",
        },  # db_id is not the id
        {"target_type": "warning", "target_id": "999999", "verdict": "useful"},
        {"target_type": "action", "target_id": "act-nope", "verdict": "useful"},
        {"target_type": "prediction", "target_id": "fc-nope", "verdict": "correct"},
    ]
    for body in bad:
        assert client.post("/intel/feedback", headers=admin, json=body).status_code == 404, body
    assert client.post("/intel/feedback", headers=admin, json={**fb, "verdict": "great"}).status_code == 422
    assert client.post("/intel/feedback", headers=admin, json={**fb, "verdict": "useful"}).status_code == 422
    assert client.post("/intel/feedback", headers=admin, json={**fb, "note": "x" * 1001}).status_code == 422

    w = spike_warnings(client, admin)[0]
    act = client.get("/intel/actions", headers=admin).json()["items"][0]
    fc = client.get("/intel/predictions", headers=admin).json()["forecasts"][0]
    for body in (
        {"target_type": "warning", "target_id": str(w["id"]), "verdict": "false_positive"},
        {"target_type": "warning", "target_id": str(w["id"]), "verdict": "useful"},
        {"target_type": "action", "target_id": act["id"], "verdict": "useful"},
        {"target_type": "prediction", "target_id": fc["id"], "verdict": "correct", "outcome": "held up"},
    ):
        assert client.post("/intel/feedback", headers=admin, json=body).status_code == 201, body
    listing = client.get("/intel/feedback?limit=2", headers=admin).json()
    assert listing["total"] == 6 and len(listing["items"]) == 2
    s = listing["summary"]
    assert s["decision"] == {"correct": 1, "incorrect": 1, "accuracy": 0.5}
    assert s["warning"] == {"useful": 1, "not_useful": 0, "false_positive": 1, "precision": 0.5}
    assert s["action"] == {"useful": 1, "not_useful": 0} and s["prediction"] == {"correct": 1, "incorrect": 0}


def test_scenarios(client, intel, admin):
    body = {
        "series_id": "volume:genre:Drama",
        "horizon_months": 6,
        "as_of": None,
        "scenarios": [
            {"name": "Trend continues", "kind": "continue"},
            {"name": "Shock", "kind": "shock", "level_shift_pct": -20, "shock_month": 1},
        ],
        "save": True,
        "title": "Drama what-if",
    }
    r = client.post("/intel/scenarios", headers=admin, json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["id"] and out["series_id"] == "volume:genre:Drama" and len(out["scenarios"]) == 2
    assert out["comparison"][0]["rank"] == 1 and out["baseline"]["points"]
    unsaved = client.post("/intel/scenarios", headers=admin, json={**body, "save": False}).json()
    assert unsaved["id"] is None
    replay = client.post(
        "/intel/scenarios", headers=admin, json={**body, "save": False, "as_of": "2016-06-01"}
    )
    assert replay.status_code == 200 and replay.json()["as_of"].startswith("2016-06-01")
    saved = client.get("/intel/scenarios", headers=admin).json()
    assert saved["total"] == 1 and saved["items"][0]["title"] == "Drama what-if"
    assert saved["items"][0]["input"]["series_id"] == "volume:genre:Drama" and saved["items"][0]["output"]

    for bad in (
        {**body, "series_id": "volume:genre:Nope"},
        {**body, "horizon_months": 99},
        {**body, "scenarios": [{"kind": "explode"}]},
        {**body, "scenarios": [{"kind": "shock"}]},  # shock needs level_shift_pct
        {**body, "as_of": "1990-01-01"},
    ):
        r = client.post("/intel/scenarios", headers=admin, json=bad)
        assert r.status_code == 422 and r.json()["detail"], bad


def test_run_validation_busy_and_failure(client, intel, admin):
    for as_of in ("not-a-date", "1990-01-01", "2030-01-01"):
        r = client.post("/intel/runs", headers=admin, json={"as_of": as_of})
        assert r.status_code == 422 and r.json()["detail"], as_of
    replay = run(client, admin, as_of="2016-06-01")
    assert replay["as_of"] == "2016-06-01T00:00:00Z"
    assert (
        client.get(f"/intel/signals?run_id={replay['run_id']}", headers=admin).json()["as_of"]
        == replay["as_of"]
    )

    assert intel._lock.acquire(blocking=False)
    try:
        assert client.post("/intel/runs", headers=admin, json={}).status_code == 409
    finally:
        intel._lock.release()

    good = intel.inputs_factory

    def broken(as_of):
        raise OSError("processed data missing")

    intel.inputs_factory = broken
    try:
        r = client.post("/intel/runs", headers=admin, json={})
    finally:
        intel.inputs_factory = good
    assert (
        r.status_code == 200
        and r.json()["status"] == "failed"
        and "processed data missing" in r.json()["error"]
    )
    st = client.get("/intel/status", headers=admin).json()
    assert st["latest_run"]["status"] == "failed" and st["health"]["pipeline"] == "failed"
    assert st["summary"] is not None  # served from the latest successful run
    assert client.get("/intel/signals", headers=admin).status_code == 200
    run(client, admin)


def test_status_shape(client, intel, admin):
    st = client.get("/intel/status", headers=admin).json()
    assert set(st) == {
        "latest_run",
        "summary",
        "data",
        "model",
        "warnings_open",
        "recent_decisions",
        "top_signals",
        "top_risks",
        "confidence_histogram",
        "health",
        "domain",  # v1.2
        "domain_info",
        "mode",  # P2.4: which runs "latest" meant
    }
    assert st["latest_run"]["status"] == "succeeded" and st["summary"]["status"] in (
        "nominal",
        "watch",
        "alert",
    )
    assert st["data"]["sources"] and "score" in st["data"]["quality"]
    assert st["model"]["version"] and st["model"]["dataset_version"] == "synthetic-test-v1"
    assert set(st["warnings_open"]["by_severity"]) == {"critical", "high", "medium", "low"}
    assert st["warnings_open"]["total"] == sum(st["warnings_open"]["by_severity"].values()) >= 1
    assert len(st["recent_decisions"]) == 5 and "db_id" in st["recent_decisions"][0]
    assert len(st["top_signals"]) <= 5 and len(st["top_risks"]) <= 5
    assert [b["bin"] for b in st["confidence_histogram"]] == [
        "0.0-0.2",
        "0.2-0.4",
        "0.4-0.6",
        "0.6-0.8",
        "0.8-1.0",
    ]
    assert sum(b["n"] for b in st["confidence_histogram"]) > 0
    assert st["health"] == {"database": "ok", "cache": "ok", "model": "ok", "pipeline": "ok"}


def test_evaluation_endpoint(client, intel, admin):
    assert client.get("/intel/evaluation", headers=admin).json() == {
        "available": False,
        "run_dir": None,
        "report": None,
        "domain": "movie",  # v1.2
        "platform": None,
    }
    d = TMP_ROOT / "experiments" / "intel-eval-20260101T000000Z"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text('{"pipeline_version": "intel-1.0.0", "notes": []}')
    try:
        ev = client.get("/intel/evaluation", headers=admin).json()
        assert (
            ev["available"] and ev["run_dir"] == d.name and ev["report"]["pipeline_version"] == "intel-1.0.0"
        )
    finally:
        (d / "report.json").unlink()


def test_admin_metrics(client, intel, admin):
    before = client.get("/admin/metrics", headers=admin).json()
    client.get("/intel/signals", headers=admin)
    client.get("/intel/signals", headers=admin)
    m = client.get("/admin/metrics", headers=admin).json()
    route = "GET /intel/signals"
    assert m["http_requests_by_route"][route] == before["http_requests_by_route"].get(route, 0) + 2
    assert m["http_requests_by_status"]["2xx"] > before["http_requests_by_status"]["2xx"]
    assert "GET /intel/series/{series_id:path}" in m["http_requests_by_route"]  # templates, not raw paths
    assert m["http_latency_ms"]["all"]["count"] > 0
    assert m["intel_pipeline"]["succeeded"] >= 1 and m["intel_pipeline"]["failed"] >= 1
    assert m["intel_pipeline"]["last_duration_ms"] > 0 and m["intel_pipeline"]["in_progress"] is False
    assert m["intel_stage_ms"]["forecast"]["count"] >= 1 and "lapse" in m["intel_stage_ms"]
    assert m["intel_warnings"]["created"] >= 1 and m["intel_warnings"]["suppressed_keys"] >= 0
    assert m["intel_warnings_open"]["total"] >= 1
    assert "retrain_model" in m["intel_decisions_latest_run"]
    assert "movielens" in m["data_freshness"] and m["model"]["loaded"] is True


def test_migration_roundtrip(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from jev_ml.paths import ROOT

    url = f"sqlite:///{tmp_path / 'migrate.db'}"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "jev_api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    intel_tables = {
        "intel_runs",
        "intel_warnings",
        "intel_warning_events",
        "intel_decisions",
        "intel_scenarios",
        "intel_feedback",
    }

    def tables() -> set[str]:
        eng = create_engine(url)
        try:
            return set(inspect(eng).get_table_names())
        finally:
            eng.dispose()

    command.upgrade(cfg, "head")
    assert intel_tables <= tables()
    command.downgrade(cfg, "0001")
    assert not intel_tables & tables() and "users" in tables()
    command.upgrade(cfg, "head")
    assert intel_tables <= tables()
    command.downgrade(cfg, "base")
    assert tables() <= {"alembic_version"}
