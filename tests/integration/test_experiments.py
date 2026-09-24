"""Online A/B experiments end to end (docs/EXPERIMENTATION.md): lifecycle, assignment, the serving path,
the exposure log, outcome attribution, results, member decisions and migration 0009."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from conftest import admin_headers, register
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from jev_api.config import get_settings
from jev_api.db import SessionLocal
from jev_api.models import Recommendation, User
from jev_api.models.experiments import AbAssignment, AbExperiment, AbExposure, AbOutcome, MemberDecision
from jev_api.services.experiments import bucket_of, enrolled, variant_index

BASE = "/experiments/online"


@pytest.fixture(scope="module")
def admin(client):
    return admin_headers(client)


@pytest.fixture
def cleanup(client, admin):
    """Stop whatever an experiment test left running, so other modules see the default serving."""
    yield
    with SessionLocal() as db:
        keys = db.scalars(select(AbExperiment.key).where(AbExperiment.status.in_(("running", "paused"))))
        for key in list(keys):
            assert client.post(f"{BASE}/{key}/stop", headers=admin).status_code == 200


def _key(prefix: str = "exp") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def body(key: str, **over) -> dict:
    out = {
        "key": key,
        "name": f"Test {key}",
        "hypothesis": "more diversity lifts engagement",
        "primary_metric": "interaction_rate",
        "variants": [
            {"name": "control", "is_control": True},
            {"name": "diverse", "config": {"hybrid_overrides": {"diversity_lambda": 0.6}}},
        ],
        "analysis": {"min_users_per_variant": 2},
    }
    out.update(over)
    return out


def create(client, admin, key: str | None = None, **over) -> dict:
    r = client.post(BASE, headers=admin, json=body(key or _key(), **over))
    assert r.status_code == 201, r.text
    return r.json()


def member(client, prefix: str = "ab", n_ratings: int = 6) -> tuple[dict, int]:
    email = f"{prefix}-{uuid.uuid4().hex[:8]}@example.com"
    h = register(client, email)
    for mid in range(1, n_ratings + 1):
        assert client.post(f"/movies/{mid}/rate", headers=h, json={"rating": 5.0}).status_code == 200
    return h, client.get("/users/me", headers=h).json()["id"]


def salt_of(key: str) -> str:
    with SessionLocal() as db:
        return db.scalar(select(AbExperiment.salt).where(AbExperiment.key == key))


def exposures(user_id: int) -> list[AbExposure]:
    with SessionLocal() as db:
        return list(
            db.scalars(select(AbExposure).where(AbExposure.user_id == user_id).order_by(AbExposure.id)).all()
        )


# --- admin API and lifecycle ----------------------------------------------------------------------------
def test_admin_only_and_validation(client, admin):
    h, _ = member(client, "ab-auth", 0)
    assert client.post(BASE, json=body(_key())).status_code == 401
    assert client.post(BASE, headers=h, json=body(_key())).status_code == 403
    assert client.get(f"{BASE}/list", headers=h).status_code == 403
    bad = [
        body(_key(), variants=[{"name": "a", "is_control": True}]),  # one variant
        body(_key(), variants=[{"name": "a"}, {"name": "b"}]),  # no control
        body(_key(), variants=[{"name": "a", "is_control": True}, {"name": "b", "is_control": True}]),
        body(_key(), variants=[{"name": "a", "is_control": True}, {"name": "a"}]),  # duplicate
        body(
            _key(),
            variants=[
                {"name": "c", "is_control": True},
                {"name": "t", "config": {"hybrid_overrides": {"nope": 1}}},
            ],
        ),
        body(
            _key(),
            variants=[
                {"name": "c", "is_control": True},
                {"name": "t", "config": {"hybrid_overrides": {"diversity_lambda": 3}}},
            ],
        ),
        body(
            _key(),
            variants=[
                {"name": "c", "is_control": True},
                {"name": "t", "config": {"model_version": "not-registered"}},
            ],
        ),
        body(
            _key(), variants=[{"name": "c", "is_control": True}, {"name": "t", "config": {"surprise": True}}]
        ),
        body(_key(), primary_metric="revenue"),
        body(_key(), traffic_percent=101),
        body("Bad Key"),
        body("list"),
    ]
    for b in bad:
        assert client.post(BASE, headers=admin, json=b).status_code == 422, b
    exp = create(client, admin)
    assert exp["status"] == "draft" and exp["data_source"] == "live"
    assert [v["name"] for v in exp["variants"]] == ["control", "diverse"]
    assert set(exp["allowed_actions"]) == {"start", "ramp", "update", "delete"}
    assert client.post(BASE, headers=admin, json=body(exp["key"])).status_code == 409  # key taken
    listed = client.get(f"{BASE}/list", headers=admin).json()
    assert exp["key"] in {e["key"] for e in listed["items"]}
    assert client.get(f"{BASE}/{exp['key']}", headers=admin).json()["id"] == exp["id"]
    assert client.get(f"{BASE}/missing-exp", headers=admin).status_code == 404
    # the offline experiments listing is untouched
    assert client.get("/experiments", headers=admin).status_code == 200
    page = client.get("/admin/audit?action=experiment.create&limit=200", headers=admin).json()["items"]
    assert any(e["target_id"] == exp["key"] for e in page)


def test_lifecycle_state_machine(client, admin, cleanup):
    exp = create(client, admin)
    k = exp["key"]
    for action in ("pause", "stop", "conclude"):  # not from draft
        assert client.post(f"{BASE}/{k}/{action}", headers=admin).status_code == 409, action
    patched = client.patch(f"{BASE}/{k}", headers=admin, json={"hypothesis": "edited", "traffic_percent": 50})
    assert patched.status_code == 200 and patched.json()["hypothesis"] == "edited"
    assert client.post(f"{BASE}/{k}/start", headers=admin).json()["status"] == "running"
    assert client.post(f"{BASE}/{k}/start", headers=admin).status_code == 409  # already running
    assert (
        client.patch(f"{BASE}/{k}", headers=admin, json={"hypothesis": "x"}).status_code == 409
    )  # immutable
    assert client.delete(f"{BASE}/{k}", headers=admin).status_code == 409
    assert client.post(f"{BASE}/{k}/conclude", headers=admin).status_code == 409  # must stop first
    assert client.post(f"{BASE}/{k}/pause", headers=admin).json()["status"] == "paused"
    assert client.post(f"{BASE}/{k}/pause", headers=admin).status_code == 409
    ramped = client.post(f"{BASE}/{k}/ramp", headers=admin, json={"traffic_percent": 80})
    assert ramped.status_code == 200 and ramped.json()["traffic_percent"] == 80
    assert client.post(f"{BASE}/{k}/start", headers=admin).json()["status"] == "running"  # resume
    stopped = client.post(f"{BASE}/{k}/stop", headers=admin).json()
    assert stopped["status"] == "stopped" and stopped["stopped_at"]
    for action in ("start", "pause", "stop"):
        assert client.post(f"{BASE}/{k}/{action}", headers=admin).status_code == 409, action
    assert client.post(f"{BASE}/{k}/ramp", headers=admin, json={"traffic_percent": 10}).status_code == 409
    done = client.post(f"{BASE}/{k}/conclude", headers=admin).json()
    assert done["status"] == "concluded" and done["conclusion"]["decision"] == "inconclusive"
    assert done["conclusion"]["winner"] is None and done["allowed_actions"] == []
    for action in ("start", "pause", "stop", "conclude"):
        assert client.post(f"{BASE}/{k}/{action}", headers=admin).status_code == 409, action
    frozen = client.get(f"{BASE}/{k}/results", headers=admin).json()
    assert frozen["conclusion"] == done["conclusion"]
    actions = {
        e["action"]
        for e in client.get("/admin/audit?limit=200", headers=admin).json()["items"]
        if e["target_id"] == k
    }
    assert actions == {
        "experiment.create",
        "experiment.start",
        "experiment.pause",
        "experiment.ramp",
        "experiment.stop",
        "experiment.conclude",
    }
    draft = create(client, admin)
    assert client.delete(f"{BASE}/{draft['key']}", headers=admin).status_code == 204
    assert client.get(f"{BASE}/{draft['key']}", headers=admin).status_code == 404


def test_one_active_experiment_per_surface(client, admin, cleanup):
    a, b = create(client, admin), create(client, admin)
    assert client.post(f"{BASE}/{a['key']}/start", headers=admin).status_code == 200
    r = client.post(f"{BASE}/{b['key']}/start", headers=admin)
    assert r.status_code == 409 and a["key"] in r.json()["detail"]
    assert client.post(f"{BASE}/{a['key']}/pause", headers=admin).status_code == 200
    assert client.post(f"{BASE}/{b['key']}/start", headers=admin).status_code == 409  # paused still holds it
    # the database enforces it too (partial unique index), whatever the code path
    with SessionLocal() as db, pytest.raises(IntegrityError):
        db.execute(text("UPDATE ab_experiments SET status = 'running' WHERE key = :k"), {"k": b["key"]})
        db.commit()
    assert client.post(f"{BASE}/{a['key']}/stop", headers=admin).status_code == 200
    assert client.post(f"{BASE}/{b['key']}/start", headers=admin).status_code == 200


# --- serving ------------------------------------------------------------------------------------------
def test_default_serving_unchanged_without_a_running_experiment(client, admin, cleanup):
    h, uid = member(client, "ab-default")
    before = client.get("/recommendations?limit=8", headers=h).json()
    assert before["cached"] is False and before["source_request_id"] is None
    ex = exposures(uid)
    assert len(ex) == 1 and ex[0].experiment_id is None and ex[0].variant is None
    assert ex[0].request_id == before["request_id"] and not ex[0].cached
    assert [i["movie_id"] for i in ex[0].items] == [i["movie_id"] for i in before["items"]]
    with SessionLocal() as db:
        rows = db.scalars(
            select(Recommendation).where(Recommendation.request_id == before["request_id"])
        ).all()
        assert rows and all(r.experiment_id is None and r.variant is None for r in rows)
    # a running experiment with 0 % traffic enrols nobody: the list is byte-for-byte the same
    exp = create(client, admin, traffic_percent=0)
    assert client.post(f"{BASE}/{exp['key']}/start", headers=admin).status_code == 200
    during = client.get("/recommendations?limit=8", headers=h).json()
    strip = lambda r: [(i["movie_id"], i["score"], i["rank"], i["reason"]) for i in r["items"]]  # noqa: E731
    assert strip(during) == strip(before) and during["model_version"] == before["model_version"]
    assert exposures(uid)[-1].experiment_id is None


def test_exposure_logged_on_cache_hit_with_a_fresh_request_id(client, admin, cleanup):
    exp = create(client, admin)
    assert client.post(f"{BASE}/{exp['key']}/start", headers=admin).status_code == 200
    h, uid = member(client, "ab-cache")
    first = client.get("/recommendations?limit=6", headers=h).json()
    second = client.get("/recommendations?limit=6", headers=h).json()
    assert first["cached"] is False and second["cached"] is True
    assert second["request_id"] != first["request_id"]
    assert second["source_request_id"] == first["request_id"]
    assert [i["recommendation_id"] for i in second["items"]] == [
        i["recommendation_id"] for i in first["items"]
    ]
    ex = exposures(uid)
    assert len(ex) == 2
    assert [e.cached for e in ex] == [False, True]
    assert ex[1].request_id == second["request_id"] and ex[1].source_request_id == first["request_id"]
    assert ex[0].experiment_id == ex[1].experiment_id == exp["id"] and ex[0].variant == ex[1].variant
    assert ex[0].items == ex[1].items and ex[1].latency_ms is not None
    with SessionLocal() as db:  # no new recommendation rows on the hit
        n = db.scalar(
            select(text("count(*)")).select_from(Recommendation).where(Recommendation.user_id == uid)
        )
        assert n == len(first["items"])
        rows = db.scalars(select(Recommendation).where(Recommendation.user_id == uid)).all()
        assert {(r.experiment_id, r.variant) for r in rows} == {(exp["id"], ex[0].variant)}


def test_assignment_is_sticky_deterministic_and_stable_under_ramp(client, admin, cleanup):
    exp = create(client, admin, traffic_percent=50)
    k = exp["key"]
    assert client.post(f"{BASE}/{k}/start", headers=admin).status_code == 200
    salt = salt_of(k)
    users = [member(client, "ab-ramp", 3) for _ in range(12)]
    for h, _ in users:
        assert client.get("/recommendations?limit=3", headers=h).status_code == 200

    def assignments() -> dict[int, str]:
        with SessionLocal() as db:
            rows = (
                db.execute(
                    select(AbAssignment.user_id, AbAssignment.variant).where(
                        AbAssignment.experiment_id == exp["id"]
                    )
                )
                .tuples()
                .all()
            )
            return dict(rows)

    first = assignments()
    # enrolment and variant follow the documented hashes exactly
    assert set(first) == {uid for _, uid in users if enrolled(salt, uid, 50)}
    names = ["control", "diverse"]
    assert all(first[u] == names[variant_index(salt, u, [1.0, 1.0])] for u in first)
    with SessionLocal() as db:
        buckets = dict(db.execute(select(AbAssignment.user_id, AbAssignment.bucket)).tuples().all())
    assert all(buckets[u] == bucket_of(salt, u) for u in first)
    # ramp up: everyone enrolled, nobody who was in moves
    assert client.post(f"{BASE}/{k}/ramp", headers=admin, json={"traffic_percent": 100}).status_code == 200
    for h, _ in users:
        client.get("/recommendations?limit=3", headers=h)
    second = assignments()
    assert len(second) == len(users) and all(second[u] == v for u, v in first.items())
    # ramp down to 0: sticky, existing members stay in their variant; new members are not enrolled
    assert client.post(f"{BASE}/{k}/ramp", headers=admin, json={"traffic_percent": 0}).status_code == 200
    for h, uid in users:
        client.get("/recommendations?limit=3", headers=h)
        assert exposures(uid)[-1].variant == second[uid]
    late, late_id = member(client, "ab-late", 3)
    client.get("/recommendations?limit=3", headers=late)
    assert late_id not in assignments() and exposures(late_id)[-1].experiment_id is None
    # exactly one variant per member per experiment (the DB refuses a second row)
    u0 = users[0][1]
    with SessionLocal() as db, pytest.raises(IntegrityError):
        other = next(v for v in names if v != second[u0])
        vid = db.scalar(
            text("SELECT id FROM ab_variants WHERE experiment_id = :e AND name = :n"),
            {"e": exp["id"], "n": other},
        )
        db.add(AbAssignment(experiment_id=exp["id"], user_id=u0, variant_id=vid, variant=other, bucket=0))
        db.commit()


def test_admins_and_test_accounts_are_excluded(client, admin, cleanup, monkeypatch):
    exp = create(client, admin)
    assert client.post(f"{BASE}/{exp['key']}/start", headers=admin).status_code == 200
    assert client.get("/recommendations?limit=3", headers=admin).status_code == 200
    settings = get_settings()
    monkeypatch.setattr(settings, "experiments_excluded_email_patterns", "qa-*@example.com")
    h = register(client, f"qa-{uuid.uuid4().hex[:6]}@example.com")
    client.post("/movies/1/rate", headers=h, json={"rating": 4.0})
    assert client.get("/recommendations?limit=3", headers=h).status_code == 200
    with SessionLocal() as db:
        admin_id = db.scalar(select(User.id).where(User.email == "admin@example.com"))
        qa_id = db.scalar(select(User.id).where(User.email.like("qa-%")))
        enrolled_ids = set(
            db.scalars(select(AbAssignment.user_id).where(AbAssignment.experiment_id == exp["id"]))
        )
    assert admin_id not in enrolled_ids and qa_id not in enrolled_ids
    assert exposures(admin_id)[-1].experiment_id is None and exposures(qa_id)[-1].experiment_id is None


def test_variants_apply_their_serving_config(client, admin, cleanup, pipeline_result):
    """strategy_decision off -> no intelligence block; model_version -> the challenger serves."""
    from jev_ml.registry import read_registry

    reg = read_registry(get_settings().models_dir)
    challenger = next(v for v in reg["versions"] if v != reg["active"])
    variants = [
        {"name": "control", "is_control": True},
        {"name": "challenger", "config": {"model_version": challenger, "strategy_decision": False}},
    ]
    exp = create(client, admin, variants=variants)
    assert client.post(f"{BASE}/{exp['key']}/start", headers=admin).status_code == 200
    salt = salt_of(exp["key"])
    seen: dict[str, dict] = {}
    for _ in range(12):
        h, uid = member(client, "ab-cfg", 4)
        seen[["control", "challenger"][variant_index(salt, uid, [1.0, 1.0])]] = client.get(
            "/recommendations?limit=5", headers=h
        ).json()
        if len(seen) == 2:
            break
    assert set(seen) == {"control", "challenger"}
    assert seen["control"]["model_version"] == reg["active"] and seen["control"]["intelligence"] is not None
    assert seen["challenger"]["model_version"] == challenger and seen["challenger"]["intelligence"] is None
    assert all(i["decision_id"] is None for i in seen["challenger"]["items"])
    from jev_api.services.experiments import loaded_challengers

    assert challenger in loaded_challengers()


# --- outcomes and results ----------------------------------------------------------------------------
def test_outcome_attribution_window(client, admin, cleanup):
    exp = create(client, admin, attribution_window_hours=24)
    assert client.post(f"{BASE}/{exp['key']}/start", headers=admin).status_code == 200
    h1, u1 = member(client, "ab-attr")
    recs = client.get("/recommendations?limit=5", headers=h1).json()["items"]
    liked, rated, off_list = recs[0], recs[1], 60
    fb = {"movie_id": liked["movie_id"], "recommendation_id": liked["recommendation_id"], "feedback": "like"}
    assert client.post("/recommendations/feedback", headers=h1, json=fb).status_code == 201
    assert (
        client.post(f"/movies/{rated['movie_id']}/rate", headers=h1, json={"rating": 4.5}).status_code == 200
    )
    assert client.post(f"/movies/{off_list}/rate", headers=h1, json={"rating": 5.0}).status_code == 200
    # a second member whose exposure is older than the window: nothing is attributed
    h2, u2 = member(client, "ab-attr-old")
    old = client.get("/recommendations?limit=5", headers=h2).json()["items"][0]
    with SessionLocal() as db:
        db.execute(
            text("UPDATE ab_exposures SET served_at = :t WHERE user_id = :u"),
            {"t": datetime.now(UTC) - timedelta(hours=30), "u": u2},
        )
        db.commit()
    fb2 = {"movie_id": old["movie_id"], "recommendation_id": old["recommendation_id"], "feedback": "like"}
    assert client.post("/recommendations/feedback", headers=h2, json=fb2).status_code == 201
    res = client.get(f"{BASE}/{exp['key']}/results", headers=admin).json()
    with SessionLocal() as db:
        outs = db.scalars(select(AbOutcome).where(AbOutcome.experiment_id == exp["id"])).all()
    mine = {(o.movie_id, o.kind): o for o in outs if o.user_id == u1}
    assert mine[(liked["movie_id"], "like")].rank == liked["rank"]
    assert (
        mine[(rated["movie_id"], "rating")].value == 4.5
        and mine[(rated["movie_id"], "rating")].rank == rated["rank"]
    )
    assert (
        mine[(off_list, "rating")].rank is None
    )  # relevant but not on the list: counts toward the ideal DCG
    assert not [o for o in outs if o.user_id == u2]
    # idempotent: a second read attributes nothing new
    client.get(f"{BASE}/{exp['key']}/results", headers=admin)
    with SessionLocal() as db:
        assert db.scalar(
            select(text("count(*)")).select_from(AbOutcome).where(AbOutcome.experiment_id == exp["id"])
        ) == len(outs)
    variant = next(
        v
        for v in res["variants"]
        if v["exposed_users"] and v["rates"]["interaction_rate"]["users_with_event"]
    )
    assert variant["rates"]["positive_rate"]["users_with_event"] >= 1
    assert variant["means"]["ndcg_at_10"]["mean"] > 0
    assert res["label"] == "live traffic" and res["unit_of_analysis"] == "member"
    assert {"srm", "comparisons", "guardrails", "sample_size", "conclusion", "warnings"} <= set(res)


def test_srm_detection_blocks_a_winner(client, admin):
    exp = create(client, admin, analysis={"min_users_per_variant": 2})
    with SessionLocal() as db:
        vids = dict(
            db.execute(text("SELECT name, id FROM ab_variants WHERE experiment_id = :e"), {"e": exp["id"]})
            .tuples()
            .all()
        )
        tag = uuid.uuid4().hex[:6]
        users = [
            User(email=f"srm-{tag}-{i}@example.com", password_hash="x", display_name="s") for i in range(80)
        ]
        db.add_all(users)
        db.flush()
        for i, u in enumerate(users):  # 70 / 10 for a 50 / 50 design
            name = "control" if i < 70 else "diverse"
            db.add(
                AbAssignment(
                    experiment_id=exp["id"], user_id=u.id, variant_id=vids[name], variant=name, bucket=0
                )
            )
        db.commit()
    res = client.get(f"{BASE}/{exp['key']}/results", headers=admin).json()
    assert res["srm"]["assigned"]["observed"] == [70, 10] and res["srm"]["detected"] is True
    assert res["srm"]["assigned"]["p_value"] < 0.001
    assert res["conclusion"]["decision"] == "inconclusive" and res["conclusion"]["winner"] is None
    assert "sample ratio mismatch" in res["conclusion"]["reasons"][0]


# --- member decisions (P1 #7) --------------------------------------------------------------------------
def test_member_decisions_resolve_from_recommendation_rows(client):
    h, uid = member(client, "ab-dec", 8)
    r = client.get("/recommendations?limit=4", headers=h).json()
    decision_id = r["intelligence"]["decision_id"]
    with SessionLocal() as db:
        rec_ids = set(db.scalars(select(Recommendation.decision_id).where(Recommendation.user_id == uid)))
        assert rec_ids == {decision_id}
        rows = db.scalars(select(MemberDecision).where(MemberDecision.decision_id.in_(rec_ids))).all()
        assert rows and all(row.user_id == uid for row in rows)
        assert rows[0].spec_id == "recommendation_strategy" and rows[0].state_hash and rows[0].state
        assert (
            rows[0].served_strategy == r["intelligence"]["served_strategy"]
            and rows[0].source == "recommendations"
        )
    # the cache holds the decision, but the id resolves from the table whether or not it is cached
    client.app.state.cache.delete_prefix("strat:")
    got = client.get(f"/me/intelligence/decisions/{decision_id}", headers=h)
    assert got.status_code == 200, got.text
    assert got.json()["decision_id"] == decision_id and got.json()["confidence_kind"] == "margin"
    # /me/intelligence reports the same decision; it is not stored twice for the same state
    me = client.get("/me/intelligence", headers=h).json()
    assert me["strategy"]["id"] == decision_id
    with SessionLocal() as db:
        assert (
            db.scalar(
                select(text("count(*)"))
                .select_from(MemberDecision)
                .where(MemberDecision.decision_id == decision_id)
            )
            == 1
        )
    other, _ = member(client, "ab-dec-other", 0)
    assert client.get(f"/me/intelligence/decisions/{decision_id}", headers=other).status_code == 404
    assert client.get("/me/intelligence/decisions/not-a-decision", headers=h).status_code == 422
    # member feedback on a decision that is only in the table (not cached) is accepted
    client.app.state.cache.delete_prefix("strat:")
    fb = {"target_type": "strategy", "target_id": decision_id, "verdict": "accepted"}
    assert client.post("/me/intelligence/feedback", headers=h, json=fb).status_code == 201


# --- migration 0009 ------------------------------------------------------------------------------------
def _migration_0009(url: str) -> None:
    from alembic import command
    from alembic.config import Config

    from jev_ml.paths import ROOT

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "jev_api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "0008")
    eng = create_engine(url)
    now = "2026-09-01 00:00:00.000000"
    exp_sql = text(
        "INSERT INTO ab_experiments (key, name, surface, status, hypothesis, primary_metric, guardrails, "
        "traffic_percent, salt, attribution_window_hours, analysis, created_at, updated_at) VALUES "
        "(:k, 'n', 'recommendations', :s, '', 'interaction_rate', '[]', 100, 'salt', 24, '{}', :t, :t)"
    )
    try:
        assert "ab_experiments" not in inspect(eng).get_table_names()
        command.upgrade(cfg, "head")
        insp = inspect(eng)
        assert {
            "ab_experiments",
            "ab_variants",
            "ab_assignments",
            "ab_exposures",
            "ab_outcomes",
            "member_decisions",
        } <= set(insp.get_table_names())
        assert {"experiment_id", "variant"} <= {c["name"] for c in insp.get_columns("recommendations")}
        with Session(eng) as db:
            db.execute(exp_sql, {"k": "a", "s": "running", "t": now})
            db.execute(exp_sql, {"k": "b", "s": "draft", "t": now})
            db.execute(exp_sql, {"k": "c", "s": "stopped", "t": now})
            db.commit()
        with Session(eng) as db, pytest.raises(IntegrityError):  # a second active experiment on the surface
            db.execute(exp_sql, {"k": "d", "s": "paused", "t": now})
            db.commit()
        with Session(eng) as db, pytest.raises(IntegrityError):  # CHECK on status
            db.execute(exp_sql, {"k": "e", "s": "shadow", "t": now})
            db.commit()
        command.downgrade(cfg, "0008")
        insp = inspect(eng)
        assert not {"ab_experiments", "member_decisions", "ab_exposures"} & set(insp.get_table_names())
        assert "experiment_id" not in {c["name"] for c in insp.get_columns("recommendations")}
        command.upgrade(cfg, "head")
        command.downgrade(cfg, "base")
    finally:
        eng.dispose()


def test_migration_0009_roundtrip_sqlite(tmp_path):
    _migration_0009(f"sqlite:///{tmp_path / 'm0009.db'}")


@pytest.mark.skipif(not os.environ.get("JEV_TEST_POSTGRES_URL"), reason="JEV_TEST_POSTGRES_URL not set")
def test_migration_0009_roundtrip_postgres():
    _migration_0009(os.environ["JEV_TEST_POSTGRES_URL"])
