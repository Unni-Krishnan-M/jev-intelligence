"""v1.1 API additions (docs/intelligence.md, section 9.3): normalised run objects and evidence,
history across runs, recommender monitoring, audit log, evaluation runs, score decisions and
decision batches, recommendation confidence.

Runs after test_intel_api.py in the same session database, against the same synthetic stream.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import TMP_ROOT, admin_headers, register
from sqlalchemy import func, select
from test_intel_api import SPIKE_KEY, synthetic_frames

from jev_ml.intel import PipelineInputs

NEW_TABLES = {
    "intel_signals",
    "intel_trends",
    "intel_anomalies",
    "intel_forecasts",
    "intel_risks",
    "intel_evidence",
    "audit_logs",
    "intel_evaluation_runs",
}
SECRETS = ("admin-pass-123", "password-123", "wrong-pass-456", "test-secret-not-for-production")


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


def run(client, admin) -> dict[str, Any]:
    r = client.post("/intel/runs", headers=admin, json={})
    assert r.status_code == 200 and r.json()["status"] == "succeeded", r.text
    return r.json()


def test_auth_required(client, intel):
    paths = (
        "/intel/evidence",
        "/intel/history/signals?key=x",
        "/intel/recommendations",
        "/intel/evaluation/runs",
        "/intel/decisions/batches",
        "/admin/audit",
    )
    member = register(client, "v11-member@example.com")
    for path in paths:
        assert client.get(path).status_code == 401, path
        assert client.get(path, headers=member).status_code == 403, path


def test_migration_0003_roundtrip(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import create_engine, inspect

    from jev_ml.paths import ROOT

    url = f"sqlite:///{tmp_path / 'migrate.db'}"
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "jev_api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)

    def schema() -> tuple[set[str], set[str], set[str], str]:
        eng = create_engine(url)
        try:
            insp = inspect(eng)
            tables = set(insp.get_table_names())
            dec = (
                {c["name"] for c in insp.get_columns("intel_decisions")}
                if "intel_decisions" in tables
                else set()
            )
            rec = {c["name"] for c in insp.get_columns("recommendations")}
            checks = (
                " ".join(c["sqltext"] for c in insp.get_check_constraints("intel_decisions")) if dec else ""
            )
            return tables, dec, rec, checks
        finally:
            eng.dispose()

    v11_dec = {"batch_id", "answer_value", "answer_interval", "scale"}
    command.upgrade(cfg, "head")
    tables, dec, rec, checks = schema()
    assert tables >= NEW_TABLES and v11_dec <= dec and {"confidence", "confidence_kind"} <= rec
    assert "'interval'" in checks
    command.downgrade(cfg, "0002")
    tables, dec, rec, checks = schema()
    assert not NEW_TABLES & tables and not v11_dec & dec and "confidence" not in rec
    assert "intel_runs" in tables and "'interval'" not in checks
    command.upgrade(cfg, "head")
    tables, dec, rec, _ = schema()
    assert tables >= NEW_TABLES and v11_dec <= dec and "confidence" in rec
    command.downgrade(cfg, "base")


def test_normalised_rows_match_run_json(client, intel, admin):
    from jev_api.db import SessionLocal
    from jev_api.models import (
        IntelAnomalyRow,
        IntelEvidenceRow,
        IntelForecastRow,
        IntelRiskRow,
        IntelRun,
        IntelSignalRow,
        IntelTrendRow,
    )
    from jev_api.services.intel import evidence_owners

    body = run(client, admin)
    with SessionLocal() as db:
        pk = db.scalar(select(IntelRun.id).where(IntelRun.run_id == body["run_id"]))
        res = intel.result_for(db, db.get(IntelRun, pk))

        def n(model) -> int:
            return db.scalar(select(func.count()).select_from(model).where(model.run_id == pk))

        assert n(IntelSignalRow) == len(res["signals"]) == body["summary"]["counts"]["signals"]
        assert n(IntelTrendRow) == len(res["trends"])
        assert n(IntelAnomalyRow) == len(res["anomalies"])
        assert n(IntelForecastRow) == len(res["predictions"]["forecasts"])
        assert n(IntelRiskRow) == len(res["risks"])
        expected_ev = sum(len(item.get("evidence") or []) for _, item in evidence_owners(res))
        assert n(IntelEvidenceRow) == expected_ev > 0
        owners = set(db.scalars(select(IntelEvidenceRow.owner_type).where(IntelEvidenceRow.run_id == pk)))
        assert {"signal", "anomaly", "decision", "warning", "action"} <= owners
        sig = db.scalar(select(IntelSignalRow).where(IntelSignalRow.run_id == pk).limit(1))
        assert sig.payload == next(s for s in res["signals"] if s["id"] == sig.signal_id)
        spike = db.scalar(
            select(IntelAnomalyRow).where(
                IntelAnomalyRow.run_id == pk, IntelAnomalyRow.dedup_key == SPIKE_KEY
            )
        )
        assert spike.severity == "critical" and spike.detected_at is not None
    assert body["stage_ms"]["persist"] > 0
    m = client.get("/admin/metrics", headers=admin).json()
    assert m["intel_rows_persisted"]["intel_evidence"] >= expected_ev
    assert m["intel_pipeline"]["last_evidence_rows"] == expected_ev
    assert m["intel_evidence_rows_per_run"]["all"]["count"] >= 1


def test_evidence_endpoint(client, intel, admin):
    page = client.get("/intel/evidence?limit=5", headers=admin).json()
    assert set(page) == {"items", "total", "limit", "offset", "run_id", "as_of"}
    assert len(page["items"]) == 5 and page["total"] > 5 and page["run_id"]
    e = page["items"][0]
    assert set(e) == {
        "id",
        "kind",
        "label",
        "value",
        "detail",
        "ref",
        "owner_type",
        "owner_id",
        "owner_title",
        "position",
        "run_id",
    }
    assert e["run_id"] == page["run_id"] and e["owner_title"]
    nxt = client.get("/intel/evidence?limit=5&offset=5", headers=admin).json()
    assert {x["id"] for x in nxt["items"]}.isdisjoint({x["id"] for x in page["items"]})

    warn = client.get("/intel/evidence?owner_type=warning&limit=200", headers=admin).json()
    assert warn["items"] and all(x["owner_type"] == "warning" for x in warn["items"])
    spike = client.get(
        f"/intel/evidence?owner_type=warning&owner_id={SPIKE_KEY}&limit=200", headers=admin
    ).json()
    assert spike["total"] >= 1 and all(x["owner_id"] == SPIKE_KEY for x in spike["items"])
    assert [x["position"] for x in spike["items"]] == list(range(spike["total"]))
    series = client.get("/intel/evidence?kind=series&limit=200", headers=admin).json()
    assert series["items"] and all(x["kind"] == "series" for x in series["items"])
    q = client.get("/intel/evidence?q=HORROR&limit=200", headers=admin).json()
    assert q["total"] >= 1
    assert all("horror" in f"{x['label']} {x['detail'] or ''} {x['owner_title']}".lower() for x in q["items"])
    assert (
        client.get("/intel/evidence?q=%25_%25&limit=5", headers=admin).json()["total"] == 0
    )  # wildcards escaped
    assert client.get("/intel/evidence?owner_type=bogus", headers=admin).status_code == 422
    assert client.get("/intel/evidence?run_id=unknown", headers=admin).status_code == 404


def test_history_across_runs(client, intel, admin):
    first = run(client, admin)
    second = run(client, admin)
    sig = client.get("/intel/signals?limit=1", headers=admin).json()["items"][0]
    h = client.get(f"/intel/history/signals?key={sig['dedup_key']}", headers=admin).json()
    assert h["entity"] == "signals" and h["key"] == sig["dedup_key"]
    assert len(h["items"]) >= 2
    assert set(h["items"][0]) == {
        "run_id",
        "as_of",
        "created_at",
        "id",
        "observed_at",
        "value",
        "score",
        "level",
        "direction",
    }
    stamps = [x["created_at"] for x in h["items"]]
    assert stamps == sorted(stamps)  # oldest -> newest
    assert [x["run_id"] for x in h["items"][-2:]] == [first["run_id"], second["run_id"]]
    assert h["items"][-1]["score"] == sig["strength"] and h["items"][-1]["direction"] == sig["direction"]
    by_id = client.get(f"/intel/history/signals?key={sig['id']}", headers=admin).json()
    assert by_id["key"] == sig["dedup_key"] and by_id["items"] == h["items"]

    anom = client.get(f"/intel/history/anomalies?key={SPIKE_KEY}", headers=admin).json()
    assert (
        anom["items"] and anom["items"][-1]["level"] == "critical" and anom["items"][-1]["direction"] == "up"
    )
    trend = client.get("/intel/trends?limit=1", headers=admin).json()["items"][0]
    th = client.get(f"/intel/history/trends?key={trend['series_id']}", headers=admin).json()
    assert th["items"][-1]["run_id"] == second["run_id"] and th["items"][-1]["value"] == trend["slope"]
    risks = client.get("/intel/risks", headers=admin).json()["items"]
    if risks:
        rk = f"risk:{risks[0]['kind']}:{risks[0]['entity']}"
        rh = client.get(f"/intel/history/risks?key={rk}", headers=admin).json()
        assert rh["items"][-1]["level"] == risks[0]["level"] and rh["items"][-1]["score"] == risks[0]["score"]
    assert client.get("/intel/history/signals?key=nope", headers=admin).json()["items"] == []
    assert client.get("/intel/history/widgets?key=x", headers=admin).status_code == 422
    assert client.get("/intel/history/signals", headers=admin).status_code == 422


def test_score_decisions_and_batches(client, intel, admin, monkeypatch):
    """A result with a score decision and a decision batch (section 9.1) round-trips through the DB."""
    import jev_api.services.intel as svc_mod

    real = svc_mod.run_pipeline

    class WithBatch:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def to_dict(self):
            data = self._inner.to_dict()
            base = data["decisions"][0]
            score = {
                **base,
                "id": "dec-score000001",
                "key": "editorial_slot_share",
                "spec_id": "editorial_slot_share",
                "kind": "score",
                "options": [],
                "option_scores": {},
                "answer": 0.23,
                "answer_interval": [0.18, 0.29],
                "scale": {"min": 0, "max": 1, "unit": "share of home-rail slots"},
                "confidence": 0.8,
                "confidence_kind": "interval",
                "batch_id": "batch-test0001",
            }
            data["decisions"] = [{**base, "batch_id": "batch-test0001"}, score, *data["decisions"][1:]]
            data["decision_batches"] = [
                {
                    "id": "batch-test0001",
                    "name": "editorial",
                    "question": "How should the home rail be programmed?",
                    "keys": [base["key"], "editorial_slot_share"],
                    "decision_ids": [base["id"], "dec-score000001"],
                    "state_hash": "0" * 40,
                    "policy_versions": {"editorial_slot_share": "slots-1.0.0"},
                }
            ]
            return data

    monkeypatch.setattr(svc_mod, "run_pipeline", lambda inputs, config=None: WithBatch(real(inputs, config)))
    body = run(client, admin)
    monkeypatch.setattr(svc_mod, "run_pipeline", real)

    decs = client.get(
        f"/intel/decisions?batch_id=batch-test0001&run_id={body['run_id']}", headers=admin
    ).json()
    assert decs["total"] == 2 and all(d["batch_id"] == "batch-test0001" for d in decs["items"])
    score = next(d for d in decs["items"] if d["kind"] == "score")
    assert score["answer"] == 0.23 and score["answer_value"] == 0.23
    assert score["answer_interval"] == [0.18, 0.29] and score["scale"]["unit"] == "share of home-rail slots"
    assert score["confidence_kind"] == "interval" and score["options"] == []
    other = next(d for d in decs["items"] if d["kind"] != "score")
    assert isinstance(other["answer"], str | type(None)) and other["answer_value"] is None
    assert client.get("/intel/decisions?batch_id=batch-none", headers=admin).json()["total"] == 0
    batches = client.get(f"/intel/decisions/batches?run_id={body['run_id']}", headers=admin).json()
    assert batches["run_id"] == body["run_id"] and len(batches["items"]) == 1
    assert batches["items"][0]["decision_ids"][1] == "dec-score000001"
    ev = client.get(f"/intel/evidence?run_id={body['run_id']}&owner_id=dec-score000001", headers=admin).json()
    assert ev["total"] == len(next(d for d in decs["items"] if d["kind"] == "score")["evidence"])

    # a normal run again (latest); without decision_batches the groups come from stored batch ids
    run(client, admin)
    latest = client.get("/intel/decisions/batches", headers=admin).json()
    assert isinstance(latest["items"], list)
    assert client.get("/intel/decisions/batches?run_id=unknown", headers=admin).status_code == 404


def test_recommendation_confidence_and_monitoring(client, intel, admin, tiny_model_dir):
    user = register(client, "v11-viewer@example.com")
    recs = client.get("/recommendations?limit=5", headers=user).json()
    assert recs["items"]
    for item in recs["items"]:
        assert "confidence" in item and "confidence_kind" in item
        assert item["confidence"] is None or 0 <= item["confidence"] <= 1
    sim = client.get("/recommendations/similar/1?limit=3").json()["items"]
    assert sim and all("confidence" in i and "confidence_kind" in i for i in sim)
    trending = client.get("/recommendations/trending?limit=3").json()["items"]
    assert all("confidence" in i for i in trending)
    hist = client.get("/recommendations/history", headers=user).json()
    assert hist and all("confidence" in h and "confidence_kind" in h for h in hist)
    first = recs["items"][0]
    fb = {"movie_id": first["movie_id"], "feedback": "like", "recommendation_id": first["recommendation_id"]}
    assert client.post("/recommendations/feedback", headers=user, json=fb).status_code == 201

    uncalibrated = not (tiny_model_dir / "calibration.json").exists()
    mon = client.get("/intel/recommendations?recent=3", headers=admin).json()
    assert set(mon) == {
        "model_version",
        "calibration",
        "served",
        "feedback_totals",
        "reason_codes",
        "confidence_histogram",
        "recent",
    }
    assert mon["model_version"] and (mon["calibration"] is None) == uncalibrated
    assert mon["served"]["total"] >= 5 and mon["served"]["per_day"][-1]["count"] >= 5
    assert set(mon["feedback_totals"]) == {"like", "dislike", "not_interested", "clicked"}
    assert mon["feedback_totals"]["like"] >= 1
    code = next(c for c in mon["reason_codes"] if c["code"] == first["reason_code"])
    assert set(code) == {"code", "served", "like", "dislike", "not_interested", "clicked", "positive_rate"}
    assert code["like"] >= 1 and 0 < code["positive_rate"] <= 1
    assert [b["bin"] for b in mon["confidence_histogram"]][:2] == ["0.0-0.1", "0.1-0.2"]
    assert len(mon["confidence_histogram"]) == 10
    assert sum(b["n"] for b in mon["confidence_histogram"]) == mon["served"]["with_confidence"]
    assert len(mon["recent"]) == 3
    assert {
        "id",
        "user_id",
        "movie_id",
        "title",
        "rank",
        "score",
        "confidence",
        "confidence_kind",
        "reason",
        "reason_code",
        "created_at",
    } <= set(mon["recent"][0])

    summary = client.get("/models/active/summary").json()
    assert "calibration" in summary and (summary["calibration"] is None) == uncalibrated
    ml = client.get("/health/ml").json()
    assert "calibration" in ml and (ml["calibration"] is None) == uncalibrated


def test_recommendation_confidence_persisted(client, intel, admin):
    """A calibrated engine's confidence is stored on the served row and returned in history."""
    from jev_api.db import SessionLocal
    from jev_api.models import Recommendation

    engine = client.app.state.engines.engine
    real = engine.recommend

    def calibrated(*args, **kwargs):
        out = real(*args, **kwargs)
        for i, r in enumerate(out):
            r.confidence, r.confidence_kind = round(0.9 - 0.1 * i, 4), "probability"
        return out

    engine.recommend = calibrated
    try:
        user = register(client, "v11-calibrated@example.com")
        items = client.get("/recommendations?limit=3", headers=user).json()["items"]
    finally:
        del engine.recommend
    assert [i["confidence"] for i in items] == [0.9, 0.8, 0.7]
    assert all(i["confidence_kind"] == "probability" for i in items)
    with SessionLocal() as db:
        row = db.get(Recommendation, items[0]["recommendation_id"])
        assert row.confidence == 0.9 and row.confidence_kind == "probability"
    hist = client.get("/recommendations/history", headers=user).json()
    assert {h["confidence"] for h in hist} == {0.9, 0.8, 0.7}
    mon = client.get("/intel/recommendations", headers=admin).json()
    assert mon["served"]["with_confidence"] >= 3
    bins = {b["bin"]: b["n"] for b in mon["confidence_histogram"]}
    assert bins["0.9-1.0"] >= 1 and bins["0.8-0.9"] >= 1 and bins["0.7-0.8"] >= 1


def test_audit_log(client, intel, admin):
    # every wired action, once
    assert (
        client.post(
            "/auth/login", json={"email": "admin@example.com", "password": "wrong-pass-456"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/auth/login", json={"email": "ghost@example.com", "password": "wrong-pass-456"}
        ).status_code
        == 401
    )
    register(client, "v11-audited@example.com")
    body = run(client, admin)
    w = client.get("/intel/warnings?status=new&limit=1", headers=admin).json()["items"]
    assert w, "the synthetic stream always leaves an open warning"
    r = client.patch(
        f"/intel/warnings/{w[0]['id']}", headers=admin, json={"status": "acknowledged", "note": "seen"}
    )
    assert r.status_code == 200
    dec = client.get("/intel/decisions?limit=1", headers=admin).json()["items"][0]
    client.post(
        "/intel/feedback",
        headers=admin,
        json={"target_type": "decision", "target_id": dec["id"], "verdict": "correct"},
    )
    r = client.post(
        "/intel/scenarios",
        headers=admin,
        json={"series_id": "volume:all", "horizon_months": 3, "save": True, "title": "audited"},
    )
    assert r.status_code == 200, r.text
    active = next(m for m in client.get("/models", headers=admin).json() if m["is_active"])
    assert client.post(f"/models/{active['id']}/activate", headers=admin).status_code == 200

    def entries(**params) -> list[dict[str, Any]]:
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        page = client.get(f"/admin/audit?limit=200&{qs}", headers=admin).json()
        assert set(page) == {"items", "total", "limit", "offset"}
        return page["items"]

    for action in (
        "auth.login.success",
        "auth.login.failure",
        "auth.register",
        "intel.run",
        "warning.transition",
        "feedback.create",
        "scenario.save",
        "model.activate",
    ):
        got = entries(action=action)
        assert got and all(e["action"] == action for e in got), action
        assert set(got[0]) == {
            "id",
            "at",
            "actor_user_id",
            "actor",
            "action",
            "target_type",
            "target_id",
            "detail",
            "request_id",
        }
    failures = entries(action="auth.login.failure")
    assert {f["detail"]["reason"] for f in failures[:2]} == {"wrong_password", "unknown_email"}
    assert failures[0]["actor"] == "ghost@example.com" and failures[0]["actor_user_id"] is None
    reg = entries(action="auth.register", actor="V11-AUDITED@example.com")
    assert len(reg) == 1 and reg[0]["actor_user_id"] and reg[0]["target_id"] == str(reg[0]["actor_user_id"])
    runs = entries(action="intel.run")
    assert runs[0]["target_id"] == body["run_id"] and runs[0]["actor"] == "admin@example.com"
    assert runs[0]["detail"]["status"] == "succeeded" and runs[0]["detail"]["persisted"]["intel_evidence"] > 0
    assert runs[0]["request_id"]  # API-triggered: carries the request id
    fb = entries(action="feedback.create")[0]
    assert (
        fb["target_type"] == "decision"
        and fb["target_id"] == dec["id"]
        and fb["detail"]["verdict"] == "correct"
    )
    assert entries(action="model.activate")[0]["detail"]["version"] == active["version"]
    tr = entries(action="warning.transition")[0]
    assert tr["target_id"] == str(w[0]["id"]) and tr["detail"]["from"] == "new"
    assert tr["detail"]["to"] == "acknowledged" and tr["detail"]["note"] == "seen"
    # newest first, pagination
    page = client.get("/admin/audit?limit=2", headers=admin).json()
    assert len(page["items"]) == 2 and page["items"][0]["id"] > page["items"][1]["id"]
    assert client.get("/admin/audit?action=bogus", headers=admin).status_code == 422
    # never a secret, anywhere in the log
    from jev_api.db import SessionLocal
    from jev_api.models import AuditLog

    with SessionLocal() as db:
        dump = json.dumps(
            [[e.actor, e.target_id, e.detail] for e in db.scalars(select(AuditLog))], default=str
        )
    assert not [s for s in SECRETS if s in dump]
    assert "eyJ" not in dump and "Bearer" not in dump
    m = client.get("/admin/metrics", headers=admin).json()
    assert (
        m["audit_entries_by_action"]["auth.login.failure"] >= 2
        and m["audit_entries_by_action"]["intel.run"] >= 1
    )


def test_audit_never_breaks_the_request(client, intel, admin, monkeypatch):
    """An audit write that fails is logged and skipped; the login still succeeds."""
    from jev_api.services import audit

    def boom(*args, **kwargs):
        raise RuntimeError("audit store down")

    monkeypatch.setattr(audit, "_clean", boom)
    r = client.post("/auth/login", json={"email": "admin@example.com", "password": "admin-pass-123"})
    client.cookies.clear()
    assert r.status_code == 200
    assert (
        client.post("/auth/login", json={"email": "admin@example.com", "password": "nope-nope"}).status_code
        == 401
    )


def test_evaluation_runs_sync(client, intel, admin):
    base = TMP_ROOT / "experiments"
    report = {
        "created_at": "2026-01-01T00:00:00Z",
        "pipeline_version": "intel-1.0.0",
        "data_version": "synthetic-intel-v1",
        "as_of": "2018-01-14T00:00:00Z",
        "forecast": {"summary": {"median_mase": 0.93, "share_beating_naive": 1.0}},
        "lapse": {"metrics": {"auc": 0.886, "ece": 0.061}},
        "anomaly": {
            "injection": {"attack_types": [{"type": "random", "auc": 0.9}, {"type": "average", "auc": 1.0}]}
        },
        "latency": {"pipeline_ms_mean": 815.0},
    }
    dirs = [base / "intel-eval-20260101T000000Z", base / "intel-eval-20260201T000000Z"]
    try:
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            (d / "report.json").write_text(json.dumps(report))
        (dirs[1] / "report.json").write_text(
            json.dumps({"pipeline_version": "intel-1.1.0"})
        )  # partial report
        out = client.get("/intel/evaluation/runs", headers=admin).json()
        assert set(out) == {"items", "total"} and out["total"] >= 2
        newest, older = out["items"][0], next(i for i in out["items"] if i["run_dir"] == dirs[0].name)
        assert newest["run_dir"] == dirs[1].name and newest["pipeline_version"] == "intel-1.1.0"
        assert all(v is None for v in newest["headline"].values())
        assert older["headline"] == {
            "median_mase": 0.93,
            "share_beating_naive": 1.0,
            "lapse_auc": 0.886,
            "lapse_ece": 0.061,
            "shilling_auc_mean": 0.95,
            "pipeline_ms_mean": 815.0,
        }
        assert older["created_at"] == "2026-01-01T00:00:00Z" and older["data_version"] == "synthetic-intel-v1"
        # a changed file is re-synced; an unchanged one keeps its row
        (dirs[0] / "report.json").write_text(json.dumps({**report, "latency": {"pipeline_ms_mean": 700.0}}))
        again = client.get("/intel/evaluation/runs", headers=admin).json()
        row = next(i for i in again["items"] if i["run_dir"] == dirs[0].name)
        assert row["headline"]["pipeline_ms_mean"] == 700.0 and row["id"] == older["id"]
        assert again["total"] == out["total"]
    finally:
        for d in dirs:
            (d / "report.json").unlink(missing_ok=True)
