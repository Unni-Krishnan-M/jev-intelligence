"""WS1 ingestion (docs/STREAMING_ARCHITECTURE.md): generic-domain observations, events health and the
debounced near-real-time refresher.

The generic domain is the synthetic monthly `synth-rates` dataset of test_platform_api (a temporary
configs/domains directory; no downloaded data). Its file ends in 2020-06.
"""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime

import pytest
import yaml
from conftest import TMP_ROOT, admin_headers, register
from test_platform_api import rates_config, rates_frame

DOMAIN = "generic:ev-rates"
URL = f"/intel/domains/{DOMAIN}/observations"


@pytest.fixture(scope="module")
def generic(client):
    root = TMP_ROOT / "events-domains-root"
    cfg = root / "configs" / "domains"
    cfg.mkdir(parents=True, exist_ok=True)
    rates_frame().to_csv(root / "rates.csv", index=False)
    (cfg / "ev-rates.yaml").write_text(yaml.safe_dump(rates_config("ev-rates", "rates.csv")))
    svc = client.app.state.intel
    original = svc.domains_root
    svc.domains_root = root
    svc.invalidate_domains()
    yield svc
    svc.domains_root = original
    svc.invalidate_domains()


@pytest.fixture(scope="module")
def admin(client):
    return admin_headers(client)


def _obs(entity: str, when: str, value: float, key: str | None = None, **extra) -> dict:
    return {
        "entity": entity,
        "event_time": when,
        "value": value,
        "idempotency_key": key or uuid.uuid4().hex,
        **extra,
    }


def _series_last(client, admin, run_id: str, series_id: str) -> tuple[str, float]:
    r = client.get(f"/intel/series/{series_id}", headers=admin, params={"domain": DOMAIN, "run_id": run_id})
    assert r.status_code == 200, r.text
    last = [p for p in r.json()["series"]["points"] if p["v"] is not None][-1]  # the grid may run on
    return str(last["t"]), float(last["v"])


def test_observations_validation(client, admin, generic):
    batch = [
        _obs("A", "2020-07-01T00:00:00Z", 5.5, "ok-1"),
        _obs("Z", "2020-07-01T00:00:00Z", 5.5),  # unknown entity
        _obs("A", "2020-07-01T00:00:00Z", 500.0),  # outside value_range [0, 100]
        _obs("A", "2020-07-15T00:00:00Z", 5.5),  # not a month start
        _obs("A", "2020-07-01T06:00:00Z", 5.5),  # not midnight
        _obs("A", "2999-01-01T00:00:00Z", 5.5),  # future
        _obs("A", "2020-07-01T00:00:00Z", 5.5, attributes={"nope": "x"}),  # unknown attribute
        _obs("A", "2020-07-01T00:00:00Z", 5.5, "ok-1"),  # a repeat of the first: duplicate
    ]
    r = client.post(URL, headers=admin, json={"observations": batch})
    assert r.status_code == 200, r.text
    out = r.json()
    assert [i["status"] for i in out["items"]] == ["accepted"] + ["rejected"] * 6 + ["duplicate"]
    reasons = [i["reason"] for i in out["items"][1:7]]
    assert "unknown entity" in reasons[0] and "range" in reasons[1] and "first day" in reasons[2]
    assert "midnight" in reasons[3] and "future" in reasons[4] and "unknown attributes" in reasons[5]
    # structural errors are a 422 for the whole batch
    for bad in ({"observations": []}, {"observations": [{"entity": "A", "value": 1}]}, {"rows": []}):
        assert client.post(URL, headers=admin, json=bad).status_code == 422
    nan = '{"observations": [{"entity": "A", "event_time": "2020-07-01", "value": NaN, '
    nan += '"idempotency_key": "n"}]}'
    r = client.post(URL, headers={**admin, "content-type": "application/json"}, content=nan)
    assert r.status_code == 422


def test_observations_access_and_domain_checks(client, admin, generic):
    body = {"observations": [_obs("A", "2020-07-01T00:00:00Z", 5.0)]}
    assert client.post(URL, json=body).status_code == 401
    member = register(client, f"obs-{uuid.uuid4().hex[:6]}@example.com")
    assert client.post(URL, headers=member, json=body).status_code == 403
    assert client.post("/intel/domains/movie/observations", headers=admin, json=body).status_code == 422
    assert (
        client.post("/intel/domains/generic:nope/observations", headers=admin, json=body).status_code == 404
    )


def test_observations_flow_into_a_live_run_and_replays_stay_put(client, admin, generic):
    """Pushed values enter the next live run (published at ingestion: no availability lag); a replay
    at a date before their ingestion does not see them; the run records the watermark."""
    before = client.post("/intel/runs", headers=admin, json={"domain": DOMAIN})
    assert before.status_code == 200 and before.json()["status"] == "succeeded", before.text
    base_period, _ = _series_last(client, admin, before.json()["run_id"], "rate:C")
    assert base_period.startswith("2020-06")
    r = client.post(
        URL,
        headers=admin,
        json={
            "observations": [_obs("C", "2020-07-01T00:00:00Z", 9.7), _obs("C", "2020-08-01T00:00:00Z", 12.3)]
        },
    )
    assert r.json()["accepted"] == 2, r.text
    newest = max(i["seq"] for i in r.json()["items"])
    live = client.post("/intel/runs", headers=admin, json={"domain": DOMAIN}).json()
    assert live["status"] == "succeeded", live["error"]
    period, value = _series_last(client, admin, live["run_id"], "rate:C")
    assert period.startswith("2020-08") and value == pytest.approx(12.3)
    assert live["event_watermark"]["max_event_id"] >= newest
    stored = client.get(f"/intel/runs/{live['run_id']}", headers=admin, params={"domain": DOMAIN}).json()
    assert stored["event_watermark"] == live["event_watermark"]  # persisted on the run (lineage)
    assert live["data_version"] != before.json()["data_version"]  # the pushed rows are part of the data
    # a revision of a period replaces the earlier value (newest ingest wins)
    client.post(URL, headers=admin, json={"observations": [_obs("C", "2020-08-01T00:00:00Z", 11.1)]})
    revised = client.post("/intel/runs", headers=admin, json={"domain": DOMAIN}).json()
    assert _series_last(client, admin, revised["run_id"], "rate:C")[1] == pytest.approx(11.1)
    # a replay as of a date before the ingestion (knowledge time = as_of) sees only the file
    replay = client.post("/intel/runs", headers=admin, json={"domain": DOMAIN, "as_of": "2021-01-01"}).json()
    assert replay["status"] == "succeeded", replay["error"]
    assert _series_last(client, admin, replay["run_id"], "rate:C")[0].startswith("2020-06")
    assert replay["event_watermark"]["n_events"] == 0
    from sqlalchemy import func, select

    from jev_api.db import SessionLocal
    from jev_api.models import AuditLog

    with SessionLocal() as db:
        n = db.scalar(select(func.count()).select_from(AuditLog).where(AuditLog.action == "events.ingest"))
        assert n and n >= 2


def test_observed_adapter_keeps_the_generic_adapter_contract(client, generic):
    """ObservedAdapter overrides GenericAdapter._raw/_lag_days/_provenance: guard their shape."""
    import pandas as pd

    from jev_api.services import events

    base = generic.adapter(DOMAIN)
    obs = pd.DataFrame(
        [
            {
                "event": 1,
                "entity": "B",
                "event_time": datetime(2020, 7, 1, tzinfo=UTC),
                "value": 7.0,
                "attributes": {},
            }
        ]
    )
    adapter = events.adapter_with_observations(base, obs, 1)
    assert isinstance(adapter, events.ObservedAdapter)
    data = adapter.load(
        datetime(2020, 7, 2, tzinfo=UTC), datetime(2020, 7, 2, tzinfo=UTC), adapter.default_config()
    )
    rows = data.observations[data.observations["entity"] == "B"]
    assert rows["timestamp"].max() == datetime(2020, 7, 1, tzinfo=UTC).timestamp()  # lag 0 for pushed rows
    # the file's own rows keep their 20-day availability lag
    assert (
        not (data.observations["timestamp"] > datetime(2020, 6, 12, tzinfo=UTC).timestamp())
        .drop(rows.index)
        .any()
    )
    assert "pushed observations" in data.sources[0]["provenance"]
    assert events.adapter_with_observations(base, obs.iloc[0:0], None) is base


def test_events_health(client, admin, generic):
    member = register(client, f"health-{uuid.uuid4().hex[:6]}@example.com")
    client.post("/movies/2/watch", headers=member)
    client.post(URL, headers=admin, json={"observations": [_obs("A", "2020-09-01T00:00:00Z", 5.0, "h-1")]})
    client.post(URL, headers=admin, json={"observations": [_obs("A", "2020-09-01T00:00:00Z", 5.0, "h-1")]})
    client.post(URL, headers=admin, json={"observations": [_obs("Q", "2020-09-01T00:00:00Z", 5.0)]})
    assert client.get("/events/health", headers=member).status_code == 403
    r = client.get("/events/health", headers=admin)
    assert r.status_code == 200, r.text
    body = r.json()
    by = {d["domain"]: d for d in body["domains"]}
    assert {"movie", DOMAIN} <= set(by)
    g = by[DOMAIN]
    assert g["today"]["accepted"] >= 1 and g["today"]["duplicates"] >= 1 and g["today"]["rejected"] >= 1
    assert g["ingest_lag_s"] is not None and g["ingest_lag_s"] < 3600
    assert g["event_time_lag_s"] > 86400 * 365  # the observations describe 2020
    assert g["events_since_last_run"] >= 1  # the newest observation came after the last live run
    assert set(g) >= {"max_event_id", "max_ingested_at", "last_live_run", "watermark", "refresh"}
    assert by["movie"]["today"]["accepted"] >= 1
    assert body["refresher"]["enabled"] is False  # auto mode is off under pytest


# --- the refresher (unit) -----------------------------------------------------------------------------------
class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class BusyError(Exception):
    pass


def test_refresher_debounces_per_domain():
    from jev_api.services.events import EventRefresher

    clock, ran = FakeClock(), []
    r = EventRefresher(ran.append, debounce=30, max_delay=120, enabled=False, clock=clock)
    r.mark_dirty("movie")
    clock.t += 20
    r.mark_dirty("movie")  # more traffic: the quiet period restarts
    r.mark_dirty("generic:x")
    clock.t += 20
    assert r.tick() == [] and ran == []
    clock.t += 11  # 31 s since the last mark
    assert r.tick() == ["generic:x", "movie"]
    assert ran == ["generic:x", "movie"]
    assert r.tick() == [] and r.due() == []  # clean: nothing to do until the next mark
    assert r.state("movie")["dirty"] is False


def test_refresher_max_delay_under_steady_traffic():
    from jev_api.services.events import EventRefresher

    clock, ran = FakeClock(), []
    r = EventRefresher(ran.append, debounce=30, max_delay=100, enabled=False, clock=clock)
    for _ in range(12):  # a mark every 10 s never leaves 30 s of quiet
        r.mark_dirty("movie")
        clock.t += 10
        r.tick()
    assert ran == ["movie"]  # but the max delay forces one run after 100 s


def test_refresher_retries_when_the_run_lock_is_busy():
    from jev_api.services.events import EventRefresher

    clock, calls = FakeClock(), []

    def run(domain: str) -> None:
        calls.append(domain)
        if len(calls) == 1:
            raise BusyError

    r = EventRefresher(run, debounce=5, max_delay=60, enabled=False, busy=(BusyError,), clock=clock)
    r.mark_dirty("movie")
    clock.t += 6
    assert r.tick() == [] and r.state("movie")["dirty"]  # busy: still dirty
    clock.t += 6
    assert r.tick() == ["movie"] and calls == ["movie", "movie"]


def test_refresher_thread_runs_a_live_refresh():
    from jev_api.services.events import EventRefresher

    done = threading.Event()
    r = EventRefresher(lambda d: done.set(), debounce=0.05, max_delay=1, enabled=True)
    try:
        r.mark_dirty("movie")
        assert done.wait(5), "the worker thread did not refresh"
        assert r.summary()["running"]
    finally:
        r.stop()


def test_refresh_enabled_auto_mode(monkeypatch):
    from jev_api.config import Settings
    from jev_api.services.events import refresh_enabled

    assert refresh_enabled(Settings(events_refresh_enabled=True)) is True
    assert refresh_enabled(Settings(events_refresh_enabled=False)) is False
    assert refresh_enabled(Settings()) is False  # PYTEST_CURRENT_TEST is set
    monkeypatch.delenv("PYTEST_CURRENT_TEST")
    assert refresh_enabled(Settings(env="development")) is True
    assert refresh_enabled(Settings(env="test")) is False


def test_ingest_marks_the_domain_dirty(client, admin, generic):
    refresher = client.app.state.event_refresher
    client.post(URL, headers=admin, json={"observations": [_obs("B", "2020-10-01T00:00:00Z", 5.2)]})
    assert refresher.state(DOMAIN)["dirty"] is True
    assert refresher.summary()["running"] is False  # disabled under pytest: no background runs
