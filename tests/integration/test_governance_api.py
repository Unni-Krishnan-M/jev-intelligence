"""Retraining and model governance through the API: snapshot with app feedback, candidate registration,
gate pass and reject, promotion (engine swap), rollback, lineage, audit rows, the retrain lock and the
decision-triggered scheduler.

Uses the synthetic fixture model (tests/conftest.py) as the incumbent and the synthetic processed data
as the snapshot base. Every test that changes the serving model restores it (other test modules share
the session-scoped client).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from conftest import MODEL_PARAMS, admin_headers, register

from jev_ml.models.hybrid import HybridConfig
from jev_ml.paths import CONFIGS_DIR
from jev_ml.registry import active_version, read_registry

WAIT = 300.0


@pytest.fixture(scope="module")
def gov(client, processed_dir: Path, tmp_path_factory):
    """Point the governance settings at private dirs; restore settings and the serving model after."""
    from jev_api.services.governance import get_service

    app = client.app
    settings = app.state.settings
    root = tmp_path_factory.mktemp("gov-api")
    base_cfg = yaml.safe_load((CONFIGS_DIR / "experiment.yaml").read_text())
    equivalent = {**base_cfg, "models": {**MODEL_PARAMS, "hybrid": asdict(HybridConfig())}}
    broken = json.loads(json.dumps(base_cfg))
    broken["models"]["hybrid"]["weights"] = {
        "content": 0, "collaborative": 0, "latent": 0, "popularity": 1, "preference": 0, "recency": 0,
    }  # fmt: skip
    cfgs = {"equivalent": root / "equivalent.yaml", "broken": root / "broken.yaml"}
    cfgs["equivalent"].write_text(yaml.safe_dump(equivalent))
    cfgs["broken"].write_text(yaml.safe_dump(broken))
    overrides = {
        "processed_dir": processed_dir,
        "governance_snapshots_dir": root / "snapshots",
        "governance_calibrate": False,
        "governance_auto_promote": False,
        "governance_gate_bootstrap_b": 200,
        "governance_gate_latency_p95_ms": 2000.0,
        "governance_config_path": cfgs["equivalent"],
    }
    saved = {k: getattr(settings, k) for k in overrides}
    for k, v in overrides.items():
        setattr(settings, k, v)
    original = active_version(settings.models_dir)
    svc = get_service(app)
    yield {
        "svc": svc,
        "settings": settings,
        "cfgs": cfgs,
        "admin": admin_headers(client),
        "original": original,
    }
    svc.wait_idle(WAIT)
    if active_version(settings.models_dir) != original:
        from jev_api.db import SessionLocal
        from jev_api.services.sync import sync_model_versions

        app.state.engines.activate(original, action="activate", actor="test-teardown")
        with SessionLocal() as db:
            sync_model_versions(db)
    for k, v in saved.items():
        setattr(settings, k, v)


def _run_job(client, gov: dict, config: str | None = None, **body: Any) -> dict[str, Any]:
    """POST /governance/retrain with an optional config path override, wait, return the job."""
    settings = gov["settings"]
    if config is not None:
        settings.governance_config_path = gov["cfgs"][config]
    try:
        r = client.post("/governance/retrain", headers=gov["admin"], json={"quick": True, **body})
        assert r.status_code == 202, r.text
        assert gov["svc"].wait_idle(WAIT)
    finally:
        settings.governance_config_path = gov["cfgs"]["equivalent"]
    job = client.get(f"/governance/jobs/{r.json()['job_id']}", headers=gov["admin"]).json()
    assert job["status"] == "succeeded", job
    return job


def _models(client, gov: dict) -> dict[str, dict[str, Any]]:
    body = client.get("/governance/models", headers=gov["admin"]).json()
    return {m["version"]: m for m in body["items"]}


def _audit(client, gov: dict, action: str) -> list[dict[str, Any]]:
    return client.get(f"/admin/audit?action={action}&limit=200", headers=gov["admin"]).json()["items"]


def test_governance_requires_admin(client, gov):
    member = register(client, f"gov-member-{uuid.uuid4().hex[:6]}@example.com")
    assert client.get("/governance/models", headers=member).status_code == 403
    assert client.post("/governance/retrain", headers=member, json={}).status_code == 403
    assert client.get("/governance/jobs").status_code == 401
    # a version is a directory name: paths are refused before they reach the filesystem
    assert client.get("/governance/models/..%2Fetc/lineage", headers=gov["admin"]).status_code in (404, 422)
    assert (
        client.post("/governance/retrain", headers=gov["admin"], json={"config": "../x.yaml"}).status_code
        == 422
    )


def test_snapshot_includes_app_feedback_deterministically(client, gov):
    user = register(client, f"gov-fb-{uuid.uuid4().hex[:6]}@example.com")
    assert client.post("/movies/1/rate", headers=user, json={"rating": 4.5}).status_code == 200
    assert client.post("/movies/2/favorite", headers=user, json={"favorite": True}).status_code == 200
    assert client.post("/movies/3/watch", headers=user).status_code == 200
    r = client.post("/recommendations/feedback", headers=user, json={"movie_id": 4, "feedback": "dislike"})
    assert r.status_code == 201, r.text
    r = client.post(
        "/recommendations/feedback", headers=user, json={"movie_id": 5, "feedback": "not_interested"}
    )
    assert r.status_code == 201, r.text
    me = client.get("/users/me", headers=user).json()

    first = client.post("/governance/snapshots", headers=gov["admin"])
    assert first.status_code == 200, first.text
    second = client.post("/governance/snapshots", headers=gov["admin"]).json()
    snap = first.json()
    assert second["snapshot_id"] == snap["snapshot_id"] and second["content_hash"] == snap["content_hash"]
    assert snap["row_counts"]["app"] >= 4 and snap["row_counts"]["exclusions"] >= 1
    assert snap["watermark"]["ratings"]
    d = gov["settings"].governance_snapshots_dir / snap["snapshot_id"]
    app_rows = pd.read_csv(d / "app_interactions.csv")
    mine = app_rows[app_rows["app_user_id"] == me["id"]].set_index("movie_id")["signal"].to_dict()
    assert mine == {1: "rating", 2: "favorite", 3: "watch", 4: "dislike"}
    assert (app_rows["user_id"] == app_rows["app_user_id"] + gov["settings"].governance_app_user_offset).all()
    excl = pd.read_csv(d / "exclusions.csv")
    assert ((excl["movie_id"] == 5) & (excl["user_id"] == me["id"] + 10_000_000)).any()
    listed = client.get("/governance/snapshots", headers=gov["admin"]).json()
    assert [s["snapshot_id"] for s in listed].count(snap["snapshot_id"]) == 1
    assert _audit(client, gov, "dataset.snapshot")[0]["target_id"] == snap["snapshot_id"]


def test_candidate_passes_gate_promotes_and_rolls_back(client, gov):
    before = client.get("/health/ml").json()["model_version"]
    assert before == gov["original"]
    job = _run_job(client, gov)
    version = job["model_version"]
    assert job["trigger"] == "manual" and job["snapshot_id"] and job["gate_passed"] is True
    assert [s["step"] for s in job["steps"]] == ["snapshot", "train", "gate"]
    # never auto-activated
    assert client.get("/health/ml").json()["model_version"] == before
    models = _models(client, gov)
    m = models[version]
    assert m["state"] == "candidate" and not m["is_active"]
    assert m["source"] == "job" and m["job_id"] == job["job_id"]
    # the bootstrap model came from files in models/, not from a job of this database
    assert models[before]["source"] == "artifact"
    # datetimes carry an explicit UTC offset (SQLite returns naive values)
    for ts in (job["created_at"], job["finished_at"], m["gated_at"]):
        assert ts.endswith("+00:00") or ts.endswith("Z"), ts
    assert m["gates"]["ndcg@10"]["status"] == "pass"
    assert m["gates"]["ndcg@10"]["numbers"]["diff"] == pytest.approx(0.0, abs=1e-12)

    lin = client.get(f"/governance/models/{version}/lineage", headers=gov["admin"]).json()
    for key in ("snapshot_id", "config_hash", "seed", "dataset_version", "experiment_run", "gate", "job"):
        assert lin[key] is not None, key
    assert "git_commit" in lin
    assert lin["snapshot"]["manifest"]["content_hash"] == lin["snapshot"]["content_hash"]
    assert lin["dataset_version"] == lin["snapshot_id"] == job["snapshot_id"]
    assert lin["lineage"]["job_id"] == job["job_id"] and lin["job"]["job_id"] == job["job_id"]
    assert {e["target_id"] for e in _audit(client, gov, "model.register")} >= {version}
    assert {e["target_id"] for e in _audit(client, gov, "model.retrain")} >= {job["job_id"]}

    # promotion swaps the engine
    r = client.post(f"/governance/models/{version}/promote", headers=gov["admin"], json={})
    assert r.status_code == 200, r.text
    assert r.json() == {"version": version, "previous": before, "action": "promote", "forced": False,
                        "gate_passed": True}  # fmt: skip
    assert client.get("/health/ml").json()["model_version"] == version
    models = _models(client, gov)
    assert models[version]["state"] == "active" and models[before]["state"] == "retired"
    promo = _audit(client, gov, "model.promote")[0]
    assert (
        promo["target_id"] == version
        and promo["detail"]["promoted"] is True
        and promo["detail"]["gate_ok"] is True
    )
    reg = read_registry(gov["settings"].models_dir)
    assert reg["active"] == version and reg["previous"] == before

    # rollback restores the previous one
    r = client.post("/governance/models/rollback", headers=gov["admin"], json={"reason": "test"})
    assert r.status_code == 200, r.text
    assert r.json()["version"] == before and r.json()["previous"] == version
    assert client.get("/health/ml").json()["model_version"] == before
    rb = _audit(client, gov, "model.rollback")[0]
    assert rb["detail"] == {"from": version, "to": before, "reason": "test"}
    assert _models(client, gov)[version]["state"] == "retired"


def test_gate_rejects_a_worse_candidate_and_force_is_audited(client, gov):
    before = client.get("/health/ml").json()["model_version"]
    job = _run_job(client, gov, config="broken")
    version = job["model_version"]
    assert job["gate_passed"] is False
    m = _models(client, gov)[version]
    assert m["state"] == "rejected" and not m["promotable"]
    assert m["gates"]["ndcg@10"]["status"] == "fail" and m["gate_reasons"]
    reject = _audit(client, gov, "model.reject")[0]
    assert reject["target_id"] == version and reject["detail"]["reasons"]

    # the gate blocks promotion; the refusal is audited with passed=false
    r = client.post(f"/governance/models/{version}/promote", headers=gov["admin"], json={})
    assert r.status_code == 409 and r.json()["blockers"] and isinstance(r.json()["detail"], str)
    refused = _audit(client, gov, "model.promote")[0]
    assert refused["target_id"] == version and refused["detail"]["promoted"] is False
    # so does the legacy endpoint
    mv_id = m["model_version_id"]
    assert client.post(f"/models/{mv_id}/activate", headers=gov["admin"]).status_code == 409
    assert client.get("/health/ml").json()["model_version"] == before
    # force needs a reason
    r = client.post(f"/governance/models/{version}/promote", headers=gov["admin"], json={"force": True})
    assert r.status_code == 422
    r = client.post(
        f"/governance/models/{version}/promote", headers=gov["admin"], json={"force": True, "reason": "drill"}
    )
    assert r.status_code == 200 and r.json()["forced"] is True
    assert client.get("/health/ml").json()["model_version"] == version
    forced = _audit(client, gov, "model.promote")[0]
    assert forced["detail"]["forced"] is True and forced["detail"]["reason"] == "drill"
    lin = client.get(f"/governance/models/{version}/lineage", headers=gov["admin"]).json()
    assert lin["forced"] is True and lin["force_reason"] == "drill"
    # a rollback needs a reason in the API
    assert client.post("/governance/models/rollback", headers=gov["admin"], json={}).status_code == 422
    assert (
        client.post("/governance/models/rollback", headers=gov["admin"], json={"reason": "  "}).status_code
        == 422
    )
    assert (
        client.post(
            "/governance/models/rollback", headers=gov["admin"], json={"reason": "undo drill"}
        ).status_code
        == 200
    )
    assert client.get("/health/ml").json()["model_version"] == before


def test_retrain_lock_prevents_concurrent_jobs(client, gov):
    from jev_api.db import engine
    from jev_api.services.governance import RetrainLock

    a = RetrainLock(engine, ttl_minutes=5)
    b = RetrainLock(engine, ttl_minutes=5)
    assert a.acquire()
    try:
        assert not b.acquire()
        r = client.post("/governance/retrain", headers=gov["admin"], json={"quick": True})
        assert r.status_code == 409 and "lock" in r.text
    finally:
        a.release()
    assert b.acquire()
    b.release()
    # an expired lease (a crashed holder) does not block forever
    stale = RetrainLock(engine, ttl_minutes=-1)
    assert stale.acquire()
    assert b.acquire()
    b.release()


def test_scheduler_acts_on_the_live_retrain_decision_once(client, gov):
    from jev_api.db import SessionLocal
    from jev_api.models import IntelDecision, IntelRun

    settings = gov["settings"]
    svc = gov["svc"]
    run_id = str(uuid.uuid4())
    decision_id = f"dec-gov-{uuid.uuid4().hex[:8]}"
    now = datetime.now(UTC)
    with SessionLocal() as db:
        db.add(IntelRun(run_id=run_id, domain="movie", trigger="schedule", status="succeeded", mode="live",
                        started_at=now, finished_at=now, pipeline_version="test"))  # fmt: skip
        db.flush()
        db.add(IntelDecision(
            run_id=run_id, decision_id=decision_id, key="retrain_model", spec_id="retrain_model",
            policy_version="retrain-1.0.0", question="retrain?", kind="boolean", options=["yes", "no"],
            answer="yes", confidence=1.0, confidence_kind="rule", as_of=now, domain="movie",
        ))  # fmt: skip
        db.commit()
    saved = settings.governance_schedule_interval_minutes
    settings.governance_schedule_interval_minutes = 1e-6
    try:
        with SessionLocal() as db:
            res = svc.scheduler_tick(db)
        assert res["action"] == "started" and res["trigger"] == "decision", res
        assert res["decision_id"] == decision_id
        with SessionLocal() as db:
            again = svc.scheduler_tick(db)
        assert again["action"] == "skipped" and "already" in again["reason"]
        job = client.get(f"/governance/jobs/{res['job_id']}", headers=gov["admin"]).json()
        assert job["status"] == "succeeded" and job["decision_id"] == decision_id
        lin = client.get(f"/governance/models/{job['model_version']}/lineage", headers=gov["admin"]).json()
        assert lin["decision_id"] == decision_id and lin["lineage"]["trigger"] == "decision"
        # a live "no" does not trigger
        with SessionLocal() as db:
            dec = db.query(IntelDecision).filter_by(decision_id=decision_id).one()
            dec.answer = "no"
            db.commit()
            assert svc.scheduler_tick(db)["action"] == "skipped"
        # with the default spacing, a recent automatic job blocks the next one
        settings.governance_schedule_interval_minutes = saved
        with SessionLocal() as db:
            assert "interval" in svc.scheduler_tick(db)["reason"]
    finally:
        settings.governance_schedule_interval_minutes = saved
        with SessionLocal() as db:  # other modules count intelligence runs: leave none behind
            db.query(IntelDecision).filter_by(run_id=run_id).delete()
            db.query(IntelRun).filter_by(run_id=run_id).delete()
            db.commit()
    assert active_version(settings.models_dir) == gov["original"]  # never auto-promoted


def test_engine_holder_follows_promotions_from_another_process(client, gov, tmp_path):
    from jev_api.services.ml import EngineHolder
    from jev_ml.registry import set_active

    settings = gov["settings"]
    holder = EngineHolder(settings.models_dir, poll_seconds=0)
    assert holder.load().version == gov["original"]
    other = next(v for v, s in read_registry(settings.models_dir)["states"].items() if s == "retired")
    set_active(other, settings.models_dir, action="promote", actor="another-process")
    try:
        holder.follow_registry(wait=True)
        assert holder.engine is not None and holder.engine.version == other
    finally:
        set_active(gov["original"], settings.models_dir, action="rollback", actor="test")
    holder.follow_registry(wait=True)
    assert holder.engine.version == gov["original"]
