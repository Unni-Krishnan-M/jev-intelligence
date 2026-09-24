"""P2.4 replay/live separation, stale-warning auto-resolution, evidence lineage and API determinism.

Movie runs use the synthetic spike dataset of test_intel_api (last complete month 2017-12 carries a
Horror spike); generic runs use a private synthetic domain in a temporary configs/domains directory,
run *live* with the service clock set to 2020-07-25 (an explicit as_of would make them replays).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import yaml
from conftest import TMP_ROOT, admin_headers, register
from sqlalchemy import func, select
from test_intel_api import SPIKE_KEY, synthetic_frames
from test_platform_api import rates_config, rates_frame

from jev_ml.domains.movie import PipelineInputs

LIN = "generic:synth-lineage"
A_KEY = "warning:series:rate:A"
SYNTH_NOW = datetime(2020, 7, 25, tzinfo=UTC)
REPLAY_AS_OF = "2018-01-10"  # after the spike month: the replay raises the spike key too


@pytest.fixture(scope="module")
def lineage_root():
    root = TMP_ROOT / "lineage-root"
    (root / "configs" / "domains").mkdir(parents=True, exist_ok=True)
    rates_frame().to_csv(root / "rates.csv", index=False)
    (root / "configs" / "domains" / "synth-lineage.yaml").write_text(
        yaml.safe_dump(rates_config("synth-lineage", "rates.csv"))
    )
    return root


@pytest.fixture(scope="module")
def svc(client, lineage_root):
    inter, movies = synthetic_frames()
    s = client.app.state.intel
    original = s.inputs_factory, s.domains_root, s.clock
    s.inputs_factory = lambda as_of: PipelineInputs(
        interactions=inter.copy(),
        movies=movies.copy(),
        dataset_meta={"dataset_version": "synthetic-intel-v1"},
        as_of=as_of,
    )
    s.domains_root = lineage_root
    s.invalidate_domains()
    yield s
    s.inputs_factory, s.domains_root, s.clock = original
    s.invalidate_domains()


@pytest.fixture(scope="module")
def admin(client):
    return admin_headers(client)


def run(client, admin, **body):
    r = client.post("/intel/runs", headers=admin, json=body)
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "succeeded", r.json()["error"]
    return r.json()


def generic_live(client, admin, svc):
    original = svc.clock
    svc.clock = lambda: SYNTH_NOW
    try:
        return run(client, admin, domain=LIN)
    finally:
        svc.clock = original


def live_warnings(db, domain="movie"):
    from jev_api.models import IntelWarning, IntelWarningEvent

    rows = db.scalars(select(IntelWarning).where(IntelWarning.domain == domain).order_by(IntelWarning.id))
    snap = [
        (
            w.id,
            w.key,
            w.status,
            w.severity,
            w.occurrences,
            w.last_seen_run_id,
            w.first_seen_run_id,
            w.updated_at,
        )
        for w in rows
    ]
    events = db.scalar(select(func.count(IntelWarningEvent.id)))
    return snap, events


# --- replay isolation ------------------------------------------------------------------------------------
def test_replay_never_touches_live_warnings(client, svc, admin):
    from jev_api.db import SessionLocal

    live = run(client, admin)
    assert live["mode"] == "live"
    with SessionLocal() as db:
        before = live_warnings(db)
    assert any(k == SPIKE_KEY for _, k, *_ in before[0])
    replay = run(client, admin, as_of=REPLAY_AS_OF)
    assert replay["mode"] == "replay" and replay["as_of"].startswith(REPLAY_AS_OF)
    with SessionLocal() as db:
        assert live_warnings(db) == before  # no row created, updated, reopened or resolved; no event
    # the replay's warnings live in its result only
    page = client.get(f"/intel/runs/{replay['run_id']}/warnings", headers=admin).json()
    assert page["mode"] == "replay" and page["run_id"] == replay["run_id"]
    assert SPIKE_KEY in {w["key"] for w in page["items"]}
    assert all(
        w["last_seen_run_id"] != replay["run_id"]
        for w in client.get("/intel/warnings?limit=200", headers=admin).json()["items"]
    )
    # repeated replays still leave the live state alone
    run(client, admin, as_of="2016-06-01")
    with SessionLocal() as db:
        assert live_warnings(db) == before


def test_replay_is_not_the_default_latest_run(client, svc, admin):
    live = run(client, admin)
    replay = run(client, admin, as_of=REPLAY_AS_OF)
    for path in ("/intel/signals", "/intel/trends", "/intel/anomalies", "/intel/risks", "/intel/actions"):
        page = client.get(path, headers=admin).json()
        assert page["run_id"] == live["run_id"] and page["mode"] == "live", path
        assert client.get(path, headers=admin, params={"mode": "replay"}).json()["run_id"] == replay["run_id"]
        assert client.get(path, headers=admin, params={"mode": "any"}).json()["run_id"] == replay["run_id"]
        by_id = client.get(path, headers=admin, params={"run_id": replay["run_id"]}).json()
        assert by_id["run_id"] == replay["run_id"] and by_id["mode"] == "replay"
    assert client.get("/intel/signals?mode=bogus", headers=admin).status_code == 422
    assert client.get("/intel/predictions", headers=admin).json()["run_id"] == live["run_id"]
    assert client.get("/intel/evidence", headers=admin).json()["run_id"] == live["run_id"]
    assert client.get("/intel/decisions/batches", headers=admin).json()["run_id"] == live["run_id"]
    st = client.get("/intel/status", headers=admin).json()
    assert st["latest_run"]["run_id"] == live["run_id"] and st["mode"] == "live"
    assert (
        client.get("/intel/status?mode=replay", headers=admin).json()["latest_run"]["run_id"]
        == replay["run_id"]
    )
    # runs carry their mode; the list can be filtered
    runs = client.get("/intel/runs?limit=200", headers=admin).json()["items"]
    assert {r["run_id"]: r["mode"] for r in runs}[replay["run_id"]] == "replay"
    only = client.get("/intel/runs?mode=replay&limit=200", headers=admin).json()
    assert only["items"] and all(r["mode"] == "replay" for r in only["items"])
    assert client.get(f"/intel/runs/{replay['run_id']}", headers=admin).json()["mode"] == "replay"
    # decisions and history default to live runs
    decs = client.get("/intel/decisions?limit=200", headers=admin).json()["items"]
    assert decs and replay["run_id"] not in {d["run_id"] for d in decs}
    rdecs = client.get("/intel/decisions?limit=200&mode=replay", headers=admin).json()["items"]
    assert rdecs and all(d["run_id"] != live["run_id"] for d in rdecs)
    hist = client.get("/intel/history/trends?key=volume:all", headers=admin).json()["items"]
    assert hist and replay["run_id"] not in {p["run_id"] for p in hist}
    anyh = client.get("/intel/history/trends?key=volume:all&mode=any", headers=admin).json()["items"]
    assert replay["run_id"] in {p["run_id"] for p in anyh}
    # metrics and the domains listing describe the live run
    m = client.get("/admin/metrics", headers=admin).json()
    assert m["intel_latest_run"]["run_id"] == live["run_id"]
    doms = {d["key"]: d for d in client.get("/intel/domains", headers=admin).json()["items"]}
    assert doms["movie"]["latest_run"]["run_id"] == live["run_id"]


def test_startup_refresh_ignores_replays(client, svc, admin, monkeypatch):
    from jev_api.db import SessionLocal
    from jev_api.models import IntelRun

    run(client, admin)
    calls: list[str] = []
    monkeypatch.setattr(svc, "run", lambda trigger, *a, **k: calls.append(trigger))
    svc._startup_refresh()
    assert calls == []  # a fresh live run exists
    # age every live run past the refresh interval; a brand-new replay must not count as fresh
    with SessionLocal() as db:
        live_rows = db.scalars(
            select(IntelRun).where(IntelRun.domain == "movie", IntelRun.mode == "live")
        ).all()
        saved = {r.id: r.started_at for r in live_rows}
        old = datetime.now(UTC) - timedelta(hours=svc.settings.intel_min_interval_hours + 48)
        for r in live_rows:
            r.started_at = old
        db.commit()
    try:
        monkeypatch.undo()
        run(client, admin, as_of=REPLAY_AS_OF)  # the newest run is now a replay
        monkeypatch.setattr(svc, "run", lambda trigger, *a, **k: calls.append(trigger))
        svc._startup_refresh()
        assert calls == ["startup"]
    finally:
        with SessionLocal() as db:
            for r in db.scalars(select(IntelRun).where(IntelRun.id.in_(saved))):
                r.started_at = saved[r.id]
            db.commit()


def test_replay_uses_the_replay_clock_for_app_events(client, svc, admin):
    member = register(client, "replay-clock@example.com")
    assert client.post("/movies/1/rate", headers=member, json={"rating": 4.5}).status_code == 200
    live = client.get("/intel/runs/" + run(client, admin)["run_id"], headers=admin).json()
    replay = run(client, admin, as_of=REPLAY_AS_OF)
    app_live = next(s for s in svc.result_for_id(live["run_id"])["data"]["sources"] if s["source"] == "app")
    app_replay = next(
        s for s in svc.result_for_id(replay["run_id"])["data"]["sources"] if s["source"] == "app"
    )
    assert app_live["rows"] >= 1  # the 2026 rating is live data
    assert app_replay["rows"] == 0  # ... and does not exist yet in a 2018 replay


# --- auto-resolution ----------------------------------------------------------------------------------------
def _calm(root):
    df = rates_frame()
    calm = rates_frame(seed=1)
    df.loc[df["area"] == "A", "rate"] = calm.loc[calm["area"] == "B", "rate"].to_numpy()  # no deterioration
    df.to_csv(root / "rates.csv", index=False)


def test_auto_resolve_after_k_live_runs(client, svc, admin, lineage_root, monkeypatch):
    from jev_api.db import SessionLocal
    from jev_api.models import AuditLog, IntelWarning

    k = svc.settings.intel_auto_resolve_runs
    assert k == 3
    monkeypatch.setattr(svc.settings, "intel_auto_resolve_max_severity", "critical")
    first = generic_live(client, admin, svc)
    (w,) = [
        x
        for x in client.get("/intel/warnings", headers=admin, params={"domain": LIN, "key": A_KEY}).json()[
            "items"
        ]
    ]
    assert w["status"] == "new" and w["last_seen_run_id"] == first["run_id"]
    _calm(lineage_root)
    try:
        for i in range(k - 1):
            generic_live(client, admin, svc)
            cur = client.get(f"/intel/warnings/{w['id']}", headers=admin, params={"domain": LIN}).json()
            assert cur["status"] == "new", i  # absent from fewer than K live runs: still open
        # replays never count towards K (nor touch the warning)
        run(client, admin, domain=LIN, as_of="2020-07-01")
        assert (
            client.get(f"/intel/warnings/{w['id']}", headers=admin, params={"domain": LIN}).json()["status"]
            == "new"
        )
        last = generic_live(client, admin, svc)
        cur = client.get(f"/intel/warnings/{w['id']}", headers=admin, params={"domain": LIN}).json()
        assert cur["status"] == "resolved"
        ev = cur["history"][-1]
        assert ev["actor"] == "system" and ev["to_status"] == "resolved" and "auto-resolved" in ev["note"]
        with SessionLocal() as db:
            row = db.get(IntelWarning, w["id"])
            assert row.closed_at is not None and row.events[-1].run_id == last["run_id"]
            audits = db.scalars(
                select(AuditLog).where(
                    AuditLog.action == "warning.auto_resolve", AuditLog.target_id == str(w["id"])
                )
            ).all()
        assert len(audits) == 1 and audits[0].actor == "system"
        assert audits[0].detail["absent_live_runs"] >= k and audits[0].detail["run_id"] == last["run_id"]
        assert last["summary"] is not None
    finally:
        rates_frame().to_csv(lineage_root / "rates.csv", index=False)
    # the deterioration returns: a new warning reopens from the auto-resolved one
    generic_live(client, admin, svc)
    items = client.get("/intel/warnings", headers=admin, params={"domain": LIN, "key": A_KEY}).json()["items"]
    fresh = [x for x in items if x["status"] == "new"]
    assert len(fresh) == 1 and fresh[0]["reopened_from"] == w["id"]


def test_auto_resolve_leaves_severe_warnings_to_an_operator(client, svc, admin, lineage_root):
    items = client.get("/intel/warnings", headers=admin, params={"domain": LIN, "key": A_KEY}).json()["items"]
    (w,) = [x for x in items if x["status"] == "new"]
    assert svc.settings.intel_auto_resolve_max_severity == "medium"
    if w["severity"] in ("low", "medium"):
        pytest.skip("the synthetic warning is not severe enough for this check")
    _calm(lineage_root)
    try:
        for _ in range(svc.settings.intel_auto_resolve_runs + 1):
            generic_live(client, admin, svc)
    finally:
        rates_frame().to_csv(lineage_root / "rates.csv", index=False)
    cur = client.get(f"/intel/warnings/{w['id']}", headers=admin, params={"domain": LIN}).json()
    assert cur["status"] == "new"  # high / critical: an operator resolves it


# --- lineage ------------------------------------------------------------------------------------------------
def _check_graph(g: dict, mode: str) -> None:
    assert g["complete"] is True and g["unresolved"] == [], g["unresolved"]
    assert (
        g["sources"]
        and g["run"]["mode"] == mode
        and g["run"]["config_hash"]
        and g["run"]["input_fingerprint"]
    )
    assert "event_watermark" in g["run"]
    ids = {n["id"] for n in g["nodes"]}
    assert g["root"]["node"] in ids and g["run"]["id"] in ids
    for e in g["edges"]:
        assert e["from"] in ids and e["to"] in ids
    for item in g["evidence"]:
        assert item["ref"] is None or item["resolves_to"] in ids


@pytest.mark.parametrize("domain", ["movie", LIN])
def test_every_decision_and_warning_has_a_lineage(client, svc, admin, domain):
    live = run(client, admin) if domain == "movie" else generic_live(client, admin, svc)
    decs = client.get(
        "/intel/decisions", headers=admin, params={"domain": domain, "run_id": live["run_id"], "limit": 200}
    ).json()["items"]
    assert decs
    for d in decs:
        r = client.get(f"/intel/decisions/{d['db_id']}/lineage", headers=admin, params={"domain": domain})
        assert r.status_code == 200, r.text
        g = r.json()
        _check_graph(g, "live")
        assert g["root"]["id"] == d["id"] and g["root"]["db_id"] == d["db_id"]
        assert g["run"]["run_id"] == live["run_id"]
    warnings = client.get(
        "/intel/warnings", headers=admin, params={"domain": domain, "limit": 200, "status": "new"}
    ).json()["items"]
    assert warnings
    for w in warnings:
        r = client.get(f"/intel/warnings/{w['id']}/lineage", headers=admin, params={"domain": domain})
        assert r.status_code == 200, r.text
        g = r.json()
        _check_graph(g, "live")
        assert g["root"]["decision_id"] == w["decision_id"] and f"decision:{w['decision_id']}" in {
            n["id"] for n in g["nodes"]
        }
        assert g["root"]["decision_db_id"] is not None
    # contract id + run_id; domain scoping
    d = decs[0]
    g = client.get(
        f"/intel/decisions/{d['id']}/lineage",
        headers=admin,
        params={"domain": domain, "run_id": live["run_id"]},
    ).json()
    assert g["root"]["db_id"] == d["db_id"]
    other = LIN if domain == "movie" else "movie"
    assert (
        client.get(
            f"/intel/decisions/{d['db_id']}/lineage", headers=admin, params={"domain": other}
        ).status_code
        == 404
    )
    assert (
        client.get("/intel/decisions/999999/lineage", headers=admin, params={"domain": domain}).status_code
        == 404
    )


def test_replay_decision_lineage_and_auth(client, svc, admin):
    replay = run(client, admin, as_of=REPLAY_AS_OF)
    d = client.get("/intel/decisions", headers=admin, params={"run_id": replay["run_id"], "limit": 1}).json()[
        "items"
    ][0]
    g = client.get(f"/intel/decisions/{d['db_id']}/lineage", headers=admin).json()
    _check_graph(g, "replay")
    member = register(client, "lineage-member@example.com")
    assert client.get(f"/intel/decisions/{d['db_id']}/lineage", headers=member).status_code == 403
    assert client.get(f"/intel/decisions/{d['db_id']}/lineage").status_code == 401


# --- determinism ---------------------------------------------------------------------------------------
def _decision_view(client, admin, run_id: str, domain: str = "movie") -> list[tuple]:
    items = client.get(
        "/intel/decisions", headers=admin, params={"domain": domain, "run_id": run_id, "limit": 200}
    ).json()["items"]
    return sorted(
        (
            d["id"],
            d["key"],
            d["answer"],
            d["confidence"],
            d["confidence_kind"],
            d["policy_version"],
            d["abstained"],
        )
        for d in items
    )


def _hashes(client, admin, run_id: str, domain: str = "movie") -> dict:
    b = client.get(
        "/intel/decisions/batches", headers=admin, params={"domain": domain, "run_id": run_id}
    ).json()
    return {x["id"]: x.get("state_hash") for x in b["items"]}


def test_same_as_of_gives_identical_decisions_via_api(client, svc, admin):
    a = run(client, admin, as_of=REPLAY_AS_OF)
    b = run(client, admin, as_of=REPLAY_AS_OF)
    assert a["run_id"] != b["run_id"]
    assert _decision_view(client, admin, a["run_id"]) == _decision_view(client, admin, b["run_id"])
    ha, hb = _hashes(client, admin, a["run_id"]), _hashes(client, admin, b["run_id"])
    assert ha == hb and all(ha.values())
    ra, rb = (svc.result_for_id(x["run_id"])["run"] for x in (a, b))
    assert ra["config_hash"] == rb["config_hash"] and ra["input_fingerprint"] == rb["input_fingerprint"]
    # the same holds for two live runs of a generic domain on the same day
    ga, gb = generic_live(client, admin, svc), generic_live(client, admin, svc)
    assert _decision_view(client, admin, ga["run_id"], LIN) == _decision_view(
        client, admin, gb["run_id"], LIN
    )
    assert _hashes(client, admin, ga["run_id"], LIN) == _hashes(client, admin, gb["run_id"], LIN)


def test_knowledge_time_is_threaded_to_the_event_cut(client, svc, admin):
    r = client.post("/intel/runs", headers=admin, json={"knowledge_time": "2018-01-10"})
    assert r.status_code == 422 and "as_of" in r.json()["detail"]
    future = (datetime.now(UTC) + timedelta(days=3)).isoformat()
    r = client.post("/intel/runs", headers=admin, json={"as_of": REPLAY_AS_OF, "knowledge_time": future})
    assert r.status_code == 422 and "future" in r.json()["detail"]
    now_known = datetime.now(UTC).isoformat()
    replay = run(client, admin, as_of=REPLAY_AS_OF, knowledge_time=now_known)
    wm = replay["event_watermark"]
    assert replay["mode"] == "replay" and wm is not None
    assert str(wm.get("knowledge_time", ""))[:10] == now_known[:10]
    default = run(client, admin, as_of=REPLAY_AS_OF)
    assert str(default["event_watermark"].get("knowledge_time", ""))[:10] == REPLAY_AS_OF
