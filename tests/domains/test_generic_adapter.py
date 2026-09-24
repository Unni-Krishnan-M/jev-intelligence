"""The generic adapter end to end on synthetic CSVs (two different configs), leakage, availability
(publication lag), the registry, and the typed round trip of emitted objects."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from jev_ml.core import run_domain
from jev_ml.core.types import typed
from jev_ml.domains import available, get_adapter
from jev_ml.domains.generic import GenericAdapter, parse_config

NOW = datetime(2021, 1, 1, tzinfo=UTC)

RATES = {
    "key": "synthetic-rates",
    "name": "Synthetic rates",
    "description": "three areas, monthly level; A deteriorates",
    "source": {
        "name": "synth",
        "file": "rates.csv",
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

ORDERS = {
    "key": "synthetic-orders",
    "name": "Synthetic orders",
    "description": "an event stream: orders per category, adverse = fewer orders",
    "grid_end": "as_of",
    "source": {"name": "shop", "file": "orders.csv", "expected_update": None},
    "columns": {
        "timestamp": "ts",
        "entity": "customer",
        "value": "amount",
        "groups": ["category"],
        "entity_type_default": "customer",
    },
    "series": [
        {
            "id": "orders:all",
            "metric": "count",
            "unit": "orders/month",
            "adverse_direction": "down",
            "forecast": True,
            "min_volume": 5,
        },
        {"id": "customers:all", "metric": "nunique", "value_column": "entity_id", "unit": "customers/month"},
        {
            "id": "orders:{group}",
            "metric": "count",
            "group_by": "category",
            "entity_type": "category",
            "unit": "orders/month",
            "adverse_direction": "down",
            "min_volume": 5,
        },
        {
            "id": "share:{group}",
            "metric": "share",
            "group_by": "category",
            "entity_type": "category",
            "unit": "share",
            "adverse_direction": "down",
            "share_forecast": True,
            "min_volume": 5,
        },
        {
            "id": "basket:{group}",
            "metric": "mean",
            "group_by": "category",
            "entity_type": "category",
            "unit": "EUR",
            "min_count": 3,
        },
    ],
    "impact_weights": {"toys": 0.9, "books": 0.4},
    "thresholds": {"anomaly_scan_months": 6},
}


def rates_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.date_range("2010-01-01", "2020-06-01", freq="MS")
    rows = []
    for area, region in (("A", "N"), ("B", "N"), ("C", "S")):
        v = 5 + np.round(rng.normal(0, 0.1, len(months)), 1)
        if area == "A":
            v[-24:] += np.linspace(0, 1.5, 24)
            v[-1] += 2.0  # a jump in the last month
        for m, x in zip(months, v, strict=True):
            rows.append((m.strftime("%Y-%m-%d"), area, round(float(x), 2), region))
    return pd.DataFrame(rows, columns=["date", "area", "rate", "region"])


def orders_frame(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2016-01-01", tz="UTC")
    for m in range(54):
        m0 = (start + pd.DateOffset(months=m)).timestamp()
        for cat, lam in (("toys", 60 if m < 42 else 15), ("books", 40)):  # toys collapse late
            for _ in range(rng.poisson(lam)):
                rows.append(
                    (m0 + rng.uniform(0, 27 * 86400), f"c{rng.integers(0, 80)}", cat, rng.gamma(4, 5))
                )
    return pd.DataFrame(rows, columns=["ts", "customer", "category", "amount"])


def adapter(raw: dict, frame: pd.DataFrame, tmp: Path) -> GenericAdapter:
    return GenericAdapter(parse_config(raw, tmp), frame=frame)


@pytest.fixture(scope="module")
def rates_result(tmp_path_factory):
    ad = adapter(RATES, rates_frame(), tmp_path_factory.mktemp("rates"))
    return ad, run_domain(ad, as_of="2020-07-25", now=NOW).to_dict()


def test_rates_end_to_end(rates_result):
    ad, d = rates_result
    json.dumps(d, allow_nan=False)
    assert d["run"]["domain"] == "generic:synthetic-rates" and d["run"]["validation"] == "core"
    ids = [s["id"] for s in d["series"]]
    assert ids == ["rate:A", "rate:B", "rate:C", "rate:region:N", "rate:region:S"]
    # the June 2020 value is published 20 days after 2020-06-01, so it is visible on 2020-07-25
    assert d["run"]["last_complete_month"] == "2020-06-01"
    ewl = {x["situation"]: x for x in d["decisions"] if x["key"] == "early_warning_level"}
    assert ewl["series:rate:A"]["answer"] == "URGENT_ACTION"
    assert all(
        v["answer"] in ("NO_ACTION", "MONITOR") for k, v in ewl.items() if "rate:A" not in k and ":N" not in k
    )
    (w,) = [w for w in d["warnings"] if w["key"] == "warning:series:rate:A"]
    assert w["decision_id"] == ewl["series:rate:A"]["id"] and w["severity"] in ("high", "critical")
    kinds = {r["kind"] for r in d["risks"] if r["entity"] == "A"}
    assert {"adverse_trend", "adverse_anomaly"} <= kinds
    assert all(
        o["domain"] == ad.key
        for sec in ("signals", "trends", "risks", "decisions", "warnings")
        for o in d[sec]
    )
    src = d["data"]["sources"][0]
    assert (
        src["license"] == "synthetic (test data)" and src["checksum"].startswith("sha256:") and src["fresh"]
    )


def test_availability_lag_hides_unpublished_values(tmp_path):
    ad = adapter(RATES, rates_frame(), tmp_path)
    d = run_domain(ad, as_of="2020-06-15", now=NOW).to_dict()  # June not yet published (20-day lag)
    assert d["run"]["last_complete_month"] == "2020-05-01"
    assert d["data"]["excluded_after_as_of"] == 3  # three areas' June rows
    assert all(p["t"] <= "2020-05-01" for s in d["series"] for p in s["points"])


def _core_view(d: dict) -> dict:
    return {
        "series": [s["points"] for s in d["series"]],
        "trends": [(t["id"], t["direction"], t["slope"], t["p_value"]) for t in d["trends"]],
        "anomalies": [(a["id"], a["score"], a["suppressed"]) for a in d["anomalies"]],
        "risks": [(r["id"], r["score"]) for r in d["risks"] if r["kind"] != "data_quality"],
        "decisions": [(x["id"], x["answer"], x["confidence"]) for x in d["decisions"]],
        "warnings": [(w["key"], w["severity"]) for w in d["warnings"]],
        "forecasts": [(f["id"], f["points"]) for f in d["predictions"]["forecasts"]],
    }


def test_generic_adapter_is_leak_free(tmp_path):
    as_of = "2019-03-10"
    base = rates_frame()
    a = run_domain(adapter(RATES, base, tmp_path), as_of=as_of, now=NOW).to_dict()
    future = base.copy()
    ts = pd.to_datetime(future["date"], utc=True)
    # rows after as_of, and rows dated before as_of but published after it (lag 20 d), are rewritten
    late = ts + pd.Timedelta(days=20) > pd.Timestamp(as_of, tz="UTC")
    future.loc[late, "rate"] = 99.0
    extra = pd.DataFrame({"date": ["2019-12-01"], "area": ["D"], "rate": [50.0], "region": ["S"]})
    b = run_domain(adapter(RATES, pd.concat([future, extra]), tmp_path), as_of=as_of, now=NOW).to_dict()
    assert int(late.sum()) > 0
    assert _core_view(a) == _core_view(b)


def test_second_config_event_stream_with_declared_weights(tmp_path):
    ad = adapter(ORDERS, orders_frame(), tmp_path)
    d = run_domain(ad, as_of="2020-06-20", now=NOW).to_dict()
    ids = [s["id"] for s in d["series"]]
    assert ids[:2] == ["orders:all", "customers:all"]
    assert set(ids[2:]) == {f"{m}:{c}" for m in ("orders", "share", "basket") for c in ("books", "toys")}
    s = {x["id"]: x for x in d["series"]}
    assert s["orders:all"]["points"][-1]["partial"] is True  # June 2020 is in progress on 06-20
    assert d["run"]["config"]["anomaly_scan_months"] == 6
    toys = [r for r in d["risks"] if r["entity"] == "toys"]
    assert toys and all(r["impact"] == 0.9 and "declared" in r["factors"][1]["detail"] for r in toys)
    ewl = {x["situation"]: x for x in d["decisions"] if x["key"] == "early_warning_level"}
    assert ewl["series:orders:toys"]["answer"] in ("WARNING", "URGENT_ACTION")
    assert "series:orders:books" not in ewl or ewl["series:orders:books"]["answer"] in (
        "NO_ACTION",
        "MONITOR",
    )
    assert {w["decision_id"] for w in d["warnings"]} <= {x["id"] for x in ewl.values()}
    assert d["predictions"]["share_forecasts"]


def test_config_validation_rejects_typos(tmp_path):
    bad = json.loads(json.dumps(RATES))
    bad["series"][0]["adverse"] = "up"
    with pytest.raises(ValueError, match="unknown keys"):
        parse_config(bad, tmp_path)
    bad2 = json.loads(json.dumps(RATES))
    bad2["thresholds"] = {"anomaly_z": 3}
    with pytest.raises(ValueError, match="unknown thresholds"):
        parse_config(bad2, tmp_path)


def test_registry_lists_every_config_with_availability(tmp_path):
    (tmp_path / "configs" / "domains").mkdir(parents=True)
    (tmp_path / "configs" / "domains" / "rates.yaml").write_text(yaml.safe_dump(RATES))
    (tmp_path / "configs" / "domains" / "broken.yaml").write_text("key: x\n")
    items = {i["key"]: i for i in available(tmp_path)}
    assert items["movie"]["available"] is False and "processed" in items["movie"]["reason"]
    assert items["generic:synthetic-rates"]["available"] is False
    assert "download_domain_data.py" in items["generic:synthetic-rates"]["reason"]
    assert (
        items["generic:broken"]["available"] is False
        and "invalid config" in items["generic:broken"]["reason"]
    )
    (tmp_path / "rates.csv").write_text(rates_frame().to_csv(index=False))
    items = {i["key"]: i for i in available(tmp_path)}
    assert items["generic:synthetic-rates"]["available"] is True
    ad = get_adapter("generic:rates", root=tmp_path)
    assert run_domain(ad, as_of="2020-07-25", now=NOW).summary["status"] in ("watch", "alert")


def test_typed_round_trip(rates_result):
    _, d = rates_result
    views = typed(d)
    for sec in ("signals", "trends", "anomalies", "risks", "decisions", "warnings", "actions"):
        assert [o.to_dict() for o in views[sec]] == d[sec]
    t = views["trends"][0]
    assert t.to_dict()["supporting_observations"] == d["trends"][0]["n_points"]
