"""v1.2 platform API (docs/platform.md, sections 8 and 10): domains, per-domain runs and isolation,
migration 0005, /me/intelligence*, and the recommendation-strategy block of GET /recommendations.

Generic domains come from a temporary directory holding configs/domains/*.yaml and a small synthetic
CSV (no network, no downloaded data): `synth-rates` and `synth-rates-b` share one dataset (so the same
warning key fires in both), `synth-missing` points at a file that does not exist (unavailable).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml
from conftest import TMP_ROOT, admin_headers, register
from sqlalchemy import column, create_engine, insert, inspect, select, table, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from test_intel_api import synthetic_frames

from jev_ml.intel import PipelineInputs

SYNTH = "generic:synth-rates"
SYNTH_B = "generic:synth-rates-b"
MISSING = "generic:synth-missing"
A_KEY = "warning:series:rate:A"
MOVIE_KEYS = {"recommendation", "user_intelligence", "lapse", "raters", "model_governance", "scenarios"}


def rates_config(key: str, file: str) -> dict:
    return {
        "key": key,
        "name": f"Synthetic rates ({key})",
        "description": "three areas, monthly level; area A deteriorates",
        "source": {
            "name": "synth",
            "file": file,
            "license": "synthetic (test data)",
            "expected_update": "monthly",
            "max_lag_days": 70,
            "availability_lag_days": {"default": 20},
        },
        "columns": {
            "timestamp": "date",
            "entity": "area",
            "value": "rate",
            "groups": ["region"],
            "entity_type_default": "area",
        },
        "value_range": [0, 100],
        "series": [
            {
                "id": "rate:{group}",
                "metric": "level",
                "group_by": "area",
                "entity_type": "area",
                "unit": "%",
                "adverse_direction": "up",
                "forecast": True,
                "anomaly_basis": "change",
                "resolution": 0.1,
            },
            {
                "id": "rate:region:{group}",
                "metric": "mean",
                "group_by": "region",
                "entity_type": "region",
                "unit": "%",
                "adverse_direction": "up",
            },
        ],
    }


def rates_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01-01", "2020-06-01", freq="MS")
    rows = []
    for area, region in (("A", "N"), ("B", "N"), ("C", "S")):
        v = 5 + np.round(rng.normal(0, 0.1, len(months)), 1)
        if area == "A":
            v[-24:] += np.linspace(0, 1.5, 24)
            v[-1] += 2.0
        for m, x in zip(months, v, strict=True):
            rows.append((m.strftime("%Y-%m-%d"), area, round(float(x), 2), region))
    return pd.DataFrame(rows, columns=["date", "area", "rate", "region"])


@pytest.fixture(scope="module")
def domains_root():
    root = TMP_ROOT / "domains-root"
    cfg = root / "configs" / "domains"
    cfg.mkdir(parents=True, exist_ok=True)
    rates_frame().to_csv(root / "rates.csv", index=False)
    for key, file in (
        ("synth-rates", "rates.csv"),
        ("synth-rates-b", "rates.csv"),
        ("synth-missing", "nope.csv"),
    ):
        (cfg / f"{key}.yaml").write_text(yaml.safe_dump(rates_config(key, file)))
    return root


@pytest.fixture(scope="module")
def platform(client, domains_root):
    """Movie inputs from the synthetic spike dataset, generic domains from `domains_root`."""
    inter, movies = synthetic_frames()
    svc = client.app.state.intel
    original = svc.inputs_factory, svc.domains_root
    svc.inputs_factory = lambda as_of: PipelineInputs(
        interactions=inter.copy(),
        movies=movies.copy(),
        dataset_meta={"dataset_version": "synthetic-intel-v1"},
        as_of=as_of,
    )
    svc.domains_root = domains_root
    svc.invalidate_domains()
    yield svc
    svc.inputs_factory, svc.domains_root = original
    svc.invalidate_domains()


@pytest.fixture(scope="module")
def admin(client):
    return admin_headers(client)


def run(client, admin, **body):
    r = client.post("/intel/runs", headers=admin, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "succeeded", r.json()["error"]
    return r.json()


SYNTH_NOW = datetime(2020, 7, 25, tzinfo=UTC)


def live_run(client, admin, domain: str) -> dict:
    """A *live* run of a synthetic generic domain on 2020-07-25 (the service clock is set to that day).
    Since P2.4 an explicit as_of makes a replay, which never touches live warnings; these tests exercise
    the live warning lifecycle, so they move the clock instead of passing as_of."""
    svc = client.app.state.intel
    original = svc.clock
    svc.clock = lambda: SYNTH_NOW
    try:
        out = run(client, admin, domain=domain)
    finally:
        svc.clock = original
    assert out["mode"] == "live"
    return out


@pytest.fixture(scope="module")
def runs(client, platform, admin):
    """One live run per domain: movie (synthetic), synth-rates and synth-rates-b on 2020-07-25."""
    return {
        "movie": run(client, admin),
        SYNTH: live_run(client, admin, SYNTH),
        SYNTH_B: live_run(client, admin, SYNTH_B),
    }


def warnings(client, admin, domain: str, **params) -> list[dict]:
    q = {"domain": domain, "limit": 200, **params}
    r = client.get("/intel/warnings", headers=admin, params=q)
    assert r.status_code == 200, r.text
    return r.json()["items"]


# --- domains ---------------------------------------------------------------------------------------
def test_domains_listing(client, platform, admin):
    user = register(client, "platform-member@example.com")
    assert client.get("/intel/domains").status_code == 401
    assert client.get("/intel/domains", headers=user).status_code == 403
    body = client.get("/intel/domains", headers=admin).json()
    by = {d["key"]: d for d in body["items"]}
    assert {"movie", SYNTH, SYNTH_B, MISSING} <= set(by)
    for d in by.values():
        assert set(d) == {
            "key",
            "name",
            "description",
            "entity_types",
            "frequency",
            "sources",
            "capabilities",
            "available",
            "reason",
            "latest_run",
            "warnings_open",
        }
    movie = by["movie"]
    assert movie["name"] == "Movies" and movie["available"] is True and movie["reason"] is None
    assert set(movie["capabilities"]) == MOVIE_KEYS and all(movie["capabilities"].values())
    assert {"genre", "user", "model", "platform"} <= set(movie["entity_types"])
    assert movie["frequency"] == "month" and any(s["source"] == "movielens" for s in movie["sources"])
    synth = by[SYNTH]
    assert synth["available"] and synth["reason"] is None and synth["frequency"] == "month"
    assert synth["capabilities"]["scenarios"] is True
    assert not any(synth["capabilities"][c] for c in MOVIE_KEYS - {"scenarios"})
    assert synth["sources"][0]["license"] == "synthetic (test data)"
    missing = by[MISSING]
    assert missing["available"] is False and "nope.csv" in missing["reason"]
    assert missing["latest_run"] is None and missing["warnings_open"] == 0


def test_unknown_and_unavailable_domains(client, platform, admin):
    for path in ("/intel/signals", "/intel/warnings", "/intel/status", "/intel/runs", "/intel/evaluation"):
        r = client.get(path, headers=admin, params={"domain": "generic:nope"})
        assert r.status_code == 404 and "unknown domain" in r.json()["detail"], path
        r = client.get(path, headers=admin, params={"domain": MISSING})
        assert r.status_code == 409 and "nope.csv" in r.json()["detail"], path
    for bad in ("../etc", "Movie", "x" * 65, "a b"):
        assert client.get("/intel/signals", headers=admin, params={"domain": bad}).status_code == 422, bad
    r = client.post("/intel/runs", headers=admin, json={"domain": "generic:nope"})
    assert r.status_code == 404
    r = client.post("/intel/runs", headers=admin, json={"domain": MISSING})
    assert r.status_code == 409 and "nope.csv" in r.json()["detail"]
    assert client.post("/intel/runs", headers=admin, json={"domain": "../x"}).status_code == 422
    r = client.post(f"/intel/runs?domain={SYNTH}", headers=admin, json={"domain": "movie"})
    assert r.status_code == 422
    # as_of is validated against the domain's own data range
    r = client.post("/intel/runs", headers=admin, json={"domain": SYNTH, "as_of": "2001-01-01"})
    assert r.status_code == 422 and "2010-01-01" in r.json()["detail"]
    future = (datetime.now(UTC) + timedelta(days=5)).date().isoformat()
    r = client.post("/intel/runs", headers=admin, json={"domain": SYNTH, "as_of": future})
    assert r.status_code == 422 and "future" in r.json()["detail"]


# --- runs, filtering and isolation ----------------------------------------------------------------------
def test_run_per_domain(client, admin, runs):
    m, g = runs["movie"], runs[SYNTH]
    assert m["domain"] == "movie" and g["domain"] == SYNTH
    assert g["as_of"].startswith("2020-07-25") and g["summary"]["counts"]["warnings"] >= 1
    assert m["pipeline_version"].startswith("intel-") and g["pipeline_version"] != m["pipeline_version"]
    listed = client.get("/intel/runs", headers=admin, params={"domain": SYNTH}).json()
    assert listed["total"] == 1 and listed["items"][0]["run_id"] == g["run_id"]
    assert all(r["domain"] == "movie" for r in client.get("/intel/runs", headers=admin).json()["items"])
    assert (
        client.get(f"/intel/runs/{g['run_id']}", headers=admin, params={"domain": SYNTH}).status_code == 200
    )
    assert client.get(f"/intel/runs/{g['run_id']}", headers=admin).status_code == 404  # not a movie run
    st = client.get("/intel/status", headers=admin, params={"domain": SYNTH}).json()
    assert st["domain"] == SYNTH and st["latest_run"]["run_id"] == g["run_id"]
    assert st["model"] is None and st["health"]["model"] == "not_applicable"
    assert st["domain_info"]["capabilities"]["lapse"] is False
    assert client.get("/intel/status", headers=admin).json()["latest_run"]["domain"] == "movie"
    domains = {d["key"]: d for d in client.get("/intel/domains", headers=admin).json()["items"]}
    assert domains[SYNTH]["latest_run"]["run_id"] == g["run_id"]
    assert domains[SYNTH]["warnings_open"] == len(warnings(client, admin, SYNTH, status="new"))


def test_run_backed_reads_are_per_domain(client, admin, runs):
    g = runs[SYNTH]
    for path in ("/intel/signals", "/intel/trends", "/intel/anomalies", "/intel/risks", "/intel/actions"):
        page = client.get(path, headers=admin, params={"domain": SYNTH, "limit": 200}).json()
        assert page["run_id"] == g["run_id"] and page["domain"] == SYNTH, path
        assert all(i.get("domain") in (SYNTH, None) for i in page["items"]), path
    trends = client.get("/intel/trends", headers=admin, params={"domain": SYNTH, "limit": 200}).json()
    assert "rate:A" in {t["series_id"] for t in trends["items"]}
    movie_trends = client.get("/intel/trends?limit=200", headers=admin).json()["items"]
    assert "rate:A" not in {t["series_id"] for t in movie_trends}
    s = client.get("/intel/series/rate:A", headers=admin, params={"domain": SYNTH}).json()
    assert s["series"]["id"] == "rate:A" and s["domain"] == SYNTH
    assert client.get("/intel/series/rate:A", headers=admin).status_code == 404
    pred = client.get("/intel/predictions", headers=admin, params={"domain": SYNTH}).json()
    assert pred["domain"] == SYNTH and pred["forecasts"]
    ev = client.get("/intel/evidence", headers=admin, params={"domain": SYNTH, "limit": 5}).json()
    assert ev["run_id"] == g["run_id"] and ev["total"] > 0
    # a run id of another domain is not found under this one
    r = client.get(
        "/intel/signals", headers=admin, params={"run_id": runs["movie"]["run_id"], "domain": SYNTH}
    )
    assert r.status_code == 404
    hist = client.get(
        "/intel/history/trends", headers=admin, params={"key": "rate:A", "domain": SYNTH}
    ).json()
    assert hist["domain"] == SYNTH and [p["run_id"] for p in hist["items"]] == [g["run_id"]]
    assert client.get("/intel/history/trends?key=rate:A", headers=admin).json()["items"] == []
    batches = client.get("/intel/decisions/batches", headers=admin, params={"domain": SYNTH}).json()
    assert batches["domain"] == SYNTH and any(
        b.get("name") == "early_warning" or "early_warning_level" in b.get("keys", [])
        for b in batches["items"]
    )


def test_warnings_are_isolated_per_domain(client, admin, runs):
    a = warnings(client, admin, SYNTH)
    b = warnings(client, admin, SYNTH_B)
    movie = warnings(client, admin, "movie")
    assert a and all(w["domain"] == SYNTH for w in a)
    assert all(w["domain"] == "movie" for w in movie)
    # the same key is open in both generic domains at once (one open warning per (domain, key))
    wa = next(w for w in a if w["key"] == A_KEY)
    wb = next(w for w in b if w["key"] == A_KEY)
    assert wa["id"] != wb["id"] and wa["status"] == wb["status"] == "new"
    assert wa["decision_id"] and wa["early_warning_level"] in ("WARNING", "URGENT_ACTION")
    assert A_KEY not in {w["key"] for w in movie}
    assert {w["id"] for w in a}.isdisjoint({w["id"] for w in movie})
    # addressed by id: only under its own domain
    assert client.get(f"/intel/warnings/{wa['id']}", headers=admin).status_code == 404
    assert (
        client.get(f"/intel/warnings/{wa['id']}", headers=admin, params={"domain": SYNTH_B}).status_code
        == 404
    )
    one = client.get(f"/intel/warnings/{wa['id']}", headers=admin, params={"domain": SYNTH}).json()
    assert one["key"] == A_KEY and one["history"]
    dec = client.get(f"/intel/decisions/{wa['decision_id']}", headers=admin, params={"domain": SYNTH}).json()
    assert dec["key"] == "early_warning_level" and dec["domain"] == SYNTH
    assert client.get(f"/intel/decisions/{wa['decision_id']}", headers=admin).status_code == 404
    decs = client.get("/intel/decisions", headers=admin, params={"domain": SYNTH, "limit": 200}).json()
    assert decs["total"] and all(d["domain"] == SYNTH for d in decs["items"])
    # a transition needs the right domain; dismissing in A leaves B's warning open
    r = client.patch(f"/intel/warnings/{wa['id']}", headers=admin, json={"status": "dismissed"})
    assert r.status_code == 404
    r = client.patch(
        f"/intel/warnings/{wa['id']}", headers=admin, params={"domain": SYNTH}, json={"status": "dismissed"}
    )
    assert r.status_code == 200 and r.json()["status"] == "dismissed"
    # re-running A suppresses the dismissed key there only; B's warning is updated, not suppressed
    live_run(client, admin, SYNTH)
    live_run(client, admin, SYNTH_B)
    assert [w["status"] for w in warnings(client, admin, SYNTH, key=A_KEY)] == ["dismissed"]
    (wb2,) = warnings(client, admin, SYNTH_B, key=A_KEY)
    assert wb2["id"] == wb["id"] and wb2["occurrences"] == 2


def test_feedback_and_scenarios_per_domain(client, admin, runs):
    (w,) = [x for x in warnings(client, admin, SYNTH_B) if x["key"] == A_KEY]
    body = {"target_type": "warning", "target_id": str(w["id"]), "verdict": "useful"}
    assert client.post("/intel/feedback", headers=admin, json=body).status_code == 404  # not a movie warning
    r = client.post("/intel/feedback", headers=admin, params={"domain": SYNTH_B}, json=body)
    assert r.status_code == 201 and r.json()["domain"] == SYNTH_B
    listed = client.get("/intel/feedback", headers=admin, params={"domain": SYNTH_B}).json()
    assert listed["total"] == 1 and listed["summary"]["warning"]["useful"] == 1
    movie_fb = client.get("/intel/feedback?limit=200", headers=admin).json()["items"]
    assert r.json()["id"] not in {f["id"] for f in movie_fb}
    dec_body = {"target_type": "decision", "target_id": w["decision_id"], "verdict": "correct"}
    assert client.post("/intel/feedback", headers=admin, json=dec_body).status_code == 404
    assert (
        client.post("/intel/feedback", headers=admin, params={"domain": SYNTH_B}, json=dec_body).status_code
        == 201
    )
    spec = {"series_id": "rate:A", "horizon_months": 6, "save": True}
    r = client.post("/intel/scenarios", headers=admin, params={"domain": SYNTH_B}, json=spec)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["domain"] == SYNTH_B and out["id"] and out["scenarios"]
    saved = client.get("/intel/scenarios", headers=admin, params={"domain": SYNTH_B}).json()
    assert [s["id"] for s in saved["items"]] == [out["id"]] and saved["items"][0]["domain"] == SYNTH_B
    assert out["id"] not in {s["id"] for s in client.get("/intel/scenarios", headers=admin).json()["items"]}
    # an unknown series of that domain is a 422 from the scenario engine
    bad = {"series_id": "volume:all"}
    assert (
        client.post("/intel/scenarios", headers=admin, params={"domain": SYNTH_B}, json=bad).status_code
        == 422
    )
    # the recommender console is movie-only
    assert client.get("/intel/recommendations", headers=admin, params={"domain": SYNTH}).status_code == 404


def test_evaluation_per_domain(client, admin, runs):
    d = TMP_ROOT / "experiments" / "platform-eval-20260101T000000Z"
    d.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": "2026-01-01T00:00:00Z",
        "horizon": 6,
        "domains": [
            {"domain": "movie", "forecast": {"n": 1}, "warnings": {}},
            {"domain": SYNTH, "forecast": {"n": 2}, "warnings": {"precision": 0.5}},
        ],
    }
    (d / "report.json").write_text(json.dumps(report))
    drift = TMP_ROOT / "experiments" / "drift-eval-20260101T000000Z"
    drift.mkdir(parents=True, exist_ok=True)
    (drift / "report.json").write_text(
        json.dumps({"detector": {"x": 1}, "adaptation": {}, "p": float("nan")})
    )
    try:
        g = client.get("/intel/evaluation", headers=admin, params={"domain": SYNTH}).json()
        assert g["available"] is False and g["report"] is None and g["domain"] == SYNTH and g["reason"]
        assert g["platform"]["run_dir"] == d.name and g["platform"]["report"]["forecast"] == {"n": 2}
        m = client.get("/intel/evaluation", headers=admin).json()
        assert m["platform"]["report"]["domain"] == "movie"
        b = client.get("/intel/evaluation", headers=admin, params={"domain": SYNTH_B}).json()
        assert b["platform"] is None
        dr = client.get("/intel/evaluation/drift", headers=admin).json()
        assert dr["available"] and dr["run_dir"] == drift.name and dr["report"]["p"] is None
        assert (
            client.get("/intel/evaluation/drift", headers=admin, params={"domain": SYNTH}).status_code == 404
        )
        runs_g = client.get("/intel/evaluation/runs", headers=admin, params={"domain": SYNTH}).json()
        assert runs_g == {"items": [], "total": 0}
    finally:
        (d / "report.json").unlink()
        (drift / "report.json").unlink()


def test_run_audit_and_metrics(client, admin, runs):
    page = client.get("/admin/audit?action=intel.run&limit=200", headers=admin).json()
    by_run = {e["target_id"]: e for e in page["items"]}
    assert by_run[runs[SYNTH]["run_id"]]["detail"]["domain"] == SYNTH
    assert by_run[runs["movie"]["run_id"]]["detail"]["domain"] == "movie"
    m = client.get("/admin/metrics", headers=admin).json()
    assert m["intel_runs_by_domain"][f"{SYNTH}:succeeded"] >= 2
    assert m["intel_runs_by_domain"]["movie:succeeded"] >= 1
    assert m["intel_domains"][SYNTH]["latest_run_id"] and m["intel_domains"][MISSING]["available"] is False


# --- /me/intelligence -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def member(client):
    h = register(client, "me-intel@example.com", "Mia")
    for i, mid in enumerate(range(1, 16)):
        rating = 5.0 if mid <= 10 else 2.0
        assert client.post(f"/movies/{mid}/rate", headers=h, json={"rating": rating}).status_code == 200, i
    assert client.patch("/users/me/preferences", headers=h, json={"genres": ["Sci-Fi"]}).status_code == 200
    return h


def test_me_intelligence_shape_and_auth(client, member):
    assert client.get("/me/intelligence").status_code == 401
    assert client.post("/me/intelligence/scenarios", json={}).status_code == 401
    assert client.post("/me/intelligence/feedback", json={}).status_code == 401
    me = client.get("/users/me", headers=member).json()
    body = client.get("/me/intelligence", headers=member).json()
    assert set(body) >= {
        "user_id",
        "as_of",
        "profile",
        "preference_history",
        "drift",
        "strategy",
        "signals",
        "recommendations",
    }
    assert body["user_id"] == me["id"] and body["profile"]["n_events"] == 15
    assert set(body["profile"]) == {"n_events", "first_event", "last_event"}
    assert {"categories", "windows"} <= set(body["preference_history"])
    drift = body["drift"]
    assert drift["status"] in ("ok", "insufficient_data") and drift["confidence_kind"] == "evidence"
    assert {"historical_window", "recent_window", "aspects", "summary", "evidence"} <= set(drift)
    strat = body["strategy"]
    assert strat["spec_id"] == "recommendation_strategy" and strat["id"].startswith("dec-")
    assert set(strat["options"]) == {"standard", "adapt_to_recent", "explore"}
    assert strat["state"]["served_strategy"] == "standard"
    recs = body["recommendations"]
    assert recs and all(r["decision_id"] == strat["id"] for r in recs)
    assert {"item_id", "title", "rank", "score", "reason", "confidence", "decision_id", "evidence"} <= set(
        recs[0]
    )
    rated = set(range(1, 16))
    assert not rated & {r["item_id"] for r in recs}  # nothing the member already rated
    # only the caller's own data: there is no user parameter, and another member sees their own
    other = register(client, "me-intel-other@example.com")
    assert (
        client.get(f"/me/intelligence?user_id={me['id']}", headers=other).json()["profile"]["n_events"] == 0
    )
    assert client.get("/me/intelligence?k=0", headers=member).status_code == 422
    assert client.get("/me/intelligence?k=51", headers=member).status_code == 422


def test_me_intelligence_needs_a_model(client, member):
    engines = client.app.state.engines
    saved = engines._engine
    engines._engine = None
    try:
        r = client.get("/me/intelligence", headers=member)
        assert r.status_code == 503 and "model not loaded" in r.json()["detail"]
        assert client.post("/me/intelligence/scenarios", headers=member, json={}).status_code == 503
    finally:
        engines._engine = saved


def test_me_scenarios(client, member):
    r = client.post("/me/intelligence/scenarios", headers=member, json={"k": 5})
    assert r.status_code == 200, r.text
    out = r.json()
    assert {"as_of", "assumptions", "uncertainty_note", "baseline", "scenarios", "evidence"} <= set(out)
    assert [s["kind"] for s in out["scenarios"]] == ["continue", "accelerate", "reverse"]
    body = {"k": 3, "scenarios": [{"name": "faster", "kind": "accelerate", "factor": 3.0}]}
    one = client.post("/me/intelligence/scenarios", headers=member, json=body).json()
    assert [s["name"] for s in one["scenarios"]] == ["faster"] and len(
        one["baseline"]["recommendations"]
    ) <= 3
    bad = [
        {"k": 0},
        {"k": 51},
        {"k": "ten"},
        {"scenarios": [{"kind": "sideways"}]},
        {"scenarios": [{"kind": "accelerate", "factor": 0.5}]},  # accelerate needs 1 < factor <= 5
        {"scenarios": [{"kind": "continue", "factor": 1e9}]},
        {"scenarios": [{"kind": "reverse", "name": "x" * 81}]},
        {"scenarios": [{"kind": "continue"}] * 5},
        {"scenarios": "all"},
    ]
    for b in bad:
        assert client.post("/me/intelligence/scenarios", headers=member, json=b).status_code == 422, b


def test_recommendations_carry_the_strategy_decision(client, member):
    from jev_api.db import SessionLocal
    from jev_api.models import Recommendation

    r = client.get("/recommendations?limit=5", headers=member).json()
    block = r["intelligence"]
    assert set(block) >= {
        "decision_id",
        "strategy",
        "confidence",
        "confidence_kind",
        "drift_detected",
        "summary",
        "evidence",
    }
    assert block["decision_id"].startswith("dec-") and block["served_strategy"] == "standard"
    assert block["strategy"] in (None, "standard")  # the synthetic members have no drift to act on
    assert block["confidence_kind"] == "margin" and block["summary"]
    assert block["strategy"] is None or block["abstained"] is False
    assert all(i["decision_id"] == block["decision_id"] and i["strategy"] == "standard" for i in r["items"])
    # existing fields unchanged
    assert {"items", "model_version", "generated_at", "request_id", "effective_weights", "profile"} <= set(r)
    again = client.get("/recommendations?limit=5", headers=member).json()
    assert again["cached"] is True and again["intelligence"]["decision_id"] == block["decision_id"]
    me = client.get("/me/intelligence", headers=member).json()
    assert me["strategy"]["id"] == block["decision_id"]  # the same decision on both endpoints
    with SessionLocal() as db:
        row = db.scalar(select(Recommendation).where(Recommendation.request_id == r["request_id"]).limit(1))
        assert row is not None and row.decision_id == block["decision_id"] and row.strategy == "standard"


def test_me_feedback(client, admin, member):
    recs = client.get("/recommendations?limit=3", headers=member).json()
    decision = recs["intelligence"]["decision_id"]
    item = recs["items"][0]["movie_id"]
    body = {"target_type": "recommendation", "target_id": str(item), "verdict": "accepted"}
    first = client.post("/me/intelligence/feedback", headers=member, json=body)
    assert first.status_code == 201, first.text
    out = first.json()
    assert {"id", "target_type", "target_id", "verdict", "created_at"} <= set(out)
    assert out["target_id"] == str(item) and out["verdict"] == "accepted" and out["decision_id"] == decision
    # repeats are deduplicated: same row, still 201
    again = client.post("/me/intelligence/feedback", headers=member, json={**body, "target_id": f"00{item}"})
    assert again.status_code == 201 and again.json()["id"] == out["id"]
    changed = client.post("/me/intelligence/feedback", headers=member, json={**body, "verdict": "rejected"})
    assert changed.json()["id"] == out["id"] and changed.json()["verdict"] == "rejected"
    strat = {"target_type": "strategy", "target_id": decision, "verdict": "accepted", "note": "fits me"}
    s = client.post("/me/intelligence/feedback", headers=member, json=strat)
    assert s.status_code == 201 and s.json()["target_id"] == decision
    listed = client.get("/me/intelligence/feedback", headers=member).json()
    assert {f["target_type"] for f in listed["items"]} == {"strategy", "recommendation"} and listed[
        "total"
    ] == 2
    # validation and ownership
    other = register(client, "me-intel-third@example.com")
    assert client.get("/me/intelligence/feedback", headers=other).json()["total"] == 0
    assert client.post("/me/intelligence/feedback", headers=other, json=strat).status_code == 404
    bad = [
        {**body, "verdict": "like"},
        {**body, "target_type": "decision"},
        {**body, "target_id": "abc"},
        {**body, "target_id": ""},
        {**body, "target_id": "9" * 65},
        {**body, "note": "x" * 1001},
        {**strat, "target_id": "not-a-decision"},
        {"target_type": "strategy", "verdict": "accepted"},
    ]
    for b in bad:
        assert client.post("/me/intelligence/feedback", headers=member, json=b).status_code == 422, b
    missing = {**body, "target_id": "424242"}
    assert client.post("/me/intelligence/feedback", headers=member, json=missing).status_code == 404
    unknown = {**strat, "target_id": "dec-0000000000"}
    assert client.post("/me/intelligence/feedback", headers=member, json=unknown).status_code == 404
    # audited once per change (a repeat is not an action), counted in metrics
    page = client.get("/admin/audit?action=me.feedback&actor=me-intel@example.com&limit=50", headers=admin)
    entries = page.json()["items"]
    assert len(entries) == 3 and {e["target_type"] for e in entries} == {"strategy", "recommendation"}
    assert all("note" not in e["detail"] for e in entries)  # free text is not copied into the audit log
    m = client.get("/admin/metrics", headers=admin).json()
    assert m["me_feedback"]["recommendation:accepted"] >= 1 and m["me_feedback"]["unchanged"] >= 1
    assert sum(v for k, v in m["strategy_decisions"].items() if k != "errors") >= 1
    assert m["strategy_overhead_ms"]["all"]["count"] >= 1


def test_me_feedback_cookie_needs_csrf(client, member):
    r = client.post("/auth/login", json={"email": "me-intel@example.com", "password": "password-123"})
    assert r.status_code == 200
    try:
        body = {"target_type": "recommendation", "target_id": "20", "verdict": "accepted"}
        assert client.post("/me/intelligence/feedback", json=body).status_code == 403
        assert client.post("/me/intelligence/scenarios", json={}).status_code == 403
        ok = client.post("/me/intelligence/feedback", json=body, headers={"X-JEV-CSRF": "1"})
        assert ok.status_code == 201
    finally:
        client.cookies.clear()


# --- migration 0005 -------------------------------------------------------------------------------------
def test_migration_0005_roundtrip_sqlite(tmp_path):
    _roundtrip_0005(f"sqlite:///{tmp_path / 'domains.db'}")


@pytest.mark.skipif(not os.environ.get("JEV_TEST_POSTGRES_URL"), reason="JEV_TEST_POSTGRES_URL not set")
def test_migration_0005_roundtrip_postgres():
    _roundtrip_0005(os.environ["JEV_TEST_POSTGRES_URL"])


def _roundtrip_0005(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    from jev_ml.paths import ROOT

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "jev_api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.downgrade(cfg, "base")  # a reused PostgreSQL database starts clean
    command.upgrade(cfg, "0004")
    eng = create_engine(url)
    now = "2026-09-01 00:00:00.000000"  # raw SQL: bound as text (sqlite3's datetime adapter is deprecated)

    def warning_row(key: str, status: str = "new", **extra) -> dict:
        return {
            "key": key,
            "title": key,
            "description": "",
            "severity": "high",
            "status": status,
            "trigger": "{}",
            "evidence": "[]",
            "recommended_action": "",
            "source": "{}",
            "detected_at": now,
            "last_seen_at": now,
            "occurrences": 1,
            "created_at": now,
            "updated_at": now,
            **extra,
        }

    def insert_warning(db: Session, **row) -> None:
        cols = ", ".join(row)
        db.execute(
            text(f"INSERT INTO intel_warnings ({cols}) VALUES ({', '.join(':' + c for c in row)})"), row
        )

    try:
        with Session(eng) as db:
            run_id = str(uuid.uuid4())
            db.execute(
                text(
                    'INSERT INTO intel_runs (run_id, "trigger", status, started_at, pipeline_version, '
                    "stage_ms) VALUES (:r, 'manual', 'succeeded', :t, 'intel-1.1.0', '{}')"
                ),
                {"r": run_id, "t": now},
            )
            insert_warning(db, **warning_row("k1"))
            db.commit()
        command.upgrade(cfg, "head")
        insp = inspect(eng)
        for name in ("intel_runs", "intel_warnings", "intel_decisions", "intel_scenarios", "intel_feedback"):
            cols = {c["name"]: c for c in insp.get_columns(name)}
            assert "domain" in cols and not cols["domain"]["nullable"], name
        assert "user_intel_feedback" in insp.get_table_names()
        assert {"decision_id", "strategy"} <= {c["name"] for c in insp.get_columns("recommendations")}
        idx = {i["name"]: i for i in insp.get_indexes("intel_warnings")}
        assert idx["uq_intel_warnings_open_key"]["column_names"] == ["domain", "key"]
        with Session(eng) as db:
            assert db.execute(text("SELECT domain FROM intel_runs")).scalar_one() == "movie"  # backfilled
            assert db.execute(text("SELECT domain FROM intel_warnings")).scalar_one() == "movie"
            insert_warning(db, **warning_row("k1", domain=SYNTH))  # same key, other domain: allowed
            db.commit()
        with Session(eng) as db, pytest.raises(IntegrityError):
            insert_warning(db, **warning_row("k1", domain=SYNTH))  # second open k1 in one domain
            db.commit()
        with Session(eng) as db:  # the widened CHECK accepts me.feedback
            audit = table("audit_logs", column("at"), column("actor"), column("action"), column("detail"))
            db.execute(insert(audit), {"at": now, "actor": "system", "action": "me.feedback", "detail": "{}"})
            db.commit()
        command.downgrade(cfg, "0004")
        insp = inspect(eng)
        assert "domain" not in {c["name"] for c in insp.get_columns("intel_warnings")}
        assert "user_intel_feedback" not in insp.get_table_names()
        idx = {i["name"]: i for i in insp.get_indexes("intel_warnings")}
        assert idx["uq_intel_warnings_open_key"]["column_names"] == ["key"]
        with Session(eng) as db:
            # the other domain's warning is gone, the movie rows stay
            assert db.execute(text("SELECT count(*) FROM intel_warnings")).scalar_one() == 1
            assert db.execute(text("SELECT count(*) FROM intel_runs")).scalar_one() == 1
            assert (
                db.execute(text("SELECT count(*) FROM audit_logs WHERE action = 'me.feedback'")).scalar_one()
                == 0
            )
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")
    finally:
        eng.dispose()
