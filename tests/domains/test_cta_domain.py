"""The real configs/domains/cta-ridership.yaml on a synthetic CSV (no network): end to end, the
calendar check, seasonal forecasts, no weekend anomalies, a collapse raising warnings, leakage,
the plain-config derivation and the seasonal confirmation rule."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import yaml

from jev_ml.core import run_domain
from jev_ml.core.series import season_classes
from jev_ml.domains.generic import GenericAdapter, parse_config
from jev_ml.domains.generic.evaluation import (
    entity_units,
    plain_config,
    seasonal_adverse_move,
    weekly_yoy,
)
from jev_ml.paths import ROOT

NOW = datetime(2026, 9, 24, tzinfo=UTC)
FACTOR = {"weekday": 1.0, "saturday": 0.55, "sunday_holiday": 0.4}
ENTITIES = {"bus": ("mode", 800_000), "rail": ("mode", 700_000), "Clark/Lake": ("station", 20_000)}


def cta_raw() -> dict:
    raw: dict = yaml.safe_load((ROOT / "configs" / "domains" / "cta-ridership.yaml").read_text())
    return raw


def cta_frame(end: str = "2020-04-30", collapse: str | None = "2020-03-16", seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2016-01-01", end, freq="D")
    per = pd.PeriodIndex(days, freq="D")
    cls = season_classes(per, 7, "weekday_us_holidays")
    label = np.where(cls <= 4, "weekday", np.where(cls == 5, "saturday", "sunday_holiday"))
    code = pd.Series(label).map({"weekday": "W", "saturday": "A", "sunday_holiday": "U"}).to_numpy()
    fac = pd.Series(label).map(FACTOR).to_numpy()
    # reduced-ridership weekdays (MLK Day, Presidents' Day, the Friday after Thanksgiving, 24-31 Dec)
    special = (season_classes(per, 7, "weekday_us_special_days") == 7) & (label == "weekday")
    fac = np.where(special, 0.6, fac)
    shock = np.where(days >= pd.Timestamp(collapse), 0.25, 1.0) if collapse else np.ones(len(days))
    rows = []
    parts = {}
    for ent, (et, base) in ENTITIES.items():
        v = np.round(base * fac * shock * np.exp(rng.normal(0, 0.03, len(days))))
        parts[ent] = v
        rows.append(
            pd.DataFrame(
                {
                    "date": days.strftime("%Y-%m-%d"),
                    "entity": ent,
                    "entity_type": et,
                    "value": v,
                    "day_type": code,
                }
            )
        )
    tot = parts["bus"] + parts["rail"]
    rows.append(
        pd.DataFrame(
            {
                "date": days.strftime("%Y-%m-%d"),
                "entity": "total",
                "entity_type": "system",
                "value": tot,
                "day_type": code,
            }
        )
    )
    return pd.concat(rows, ignore_index=True)


def adapter(frame: pd.DataFrame, raw: dict | None = None) -> GenericAdapter:
    return GenericAdapter(parse_config(raw or cta_raw(), ROOT), frame=frame)


@pytest.fixture(scope="module")
def collapse_run():
    return run_domain(adapter(cta_frame()), as_of="2020-03-20", now=NOW).to_dict()


def test_cta_config_end_to_end(collapse_run):
    d = collapse_run
    assert d["run"]["frequency"] == "day" and d["run"]["last_complete_month"] == "2020-03-19"
    ids = {s["id"] for s in d["series"]}
    assert {"boardings:total", "boardings:bus", "boardings:rail", "boardings:Clark/Lake"} <= ids
    assert {"vs_last_year:total", "vs_last_year:bus"} <= ids
    types = {s["id"]: s["entity_type"] for s in d["series"]}
    assert types["boardings:total"] == "system" and types["boardings:bus"] == "mode"
    assert types["boardings:Clark/Lake"] == "station"
    cal = next(c for c in d["data"]["quality"]["checks"] if c["name"].endswith("_calendar"))
    assert cal["passed"] and cal["value"] == 1.0
    for f in d["predictions"]["forecasts"]:
        assert f["horizon_unit"] == "day" and f["backtest"]["benchmark"] == "seasonal_naive"
        assert f["seasonal_period"] == 7 and f["candidate_models"][0] == "naive"


def test_cta_collapse_raises_warnings(collapse_run):
    d = collapse_run
    warned = {w["key"] for w in d["warnings"]}
    for e in ("total", "bus", "rail", "Clark/Lake"):
        assert f"warning:series:boardings:{e}" in warned
    ewl = {x["situation"]: x["answer"] for x in d["decisions"] if x["key"] == "early_warning_level"}
    assert ewl["series:boardings:total"] == "URGENT_ACTION"


def test_cta_calm_period_weekends_are_not_anomalies():
    frame = cta_frame(collapse=None)
    weekend = lambda d: [  # noqa: E731
        a for a in d["anomalies"] if not a["suppressed"] and pd.Timestamp(a["detected_at"]).dayofweek >= 5
    ]
    aware = weekend(run_domain(adapter(frame), as_of="2019-06-01", now=NOW).to_dict())
    plain = weekend(
        run_domain(adapter(frame, plain_config(cta_raw())), as_of="2019-06-01", now=NOW).to_dict()
    )
    assert not [a for a in aware if pd.Timestamp(a["detected_at"]).dayofweek == 6]  # no Sunday
    assert len(aware) <= 1  # at most a noise hit (|z| >= 3.5 happens)
    assert len(plain) >= 8  # the trailing baseline flags every weekend of the scan window


def test_cta_as_of_is_leak_free():
    as_of = "2020-03-10"
    base = cta_frame()
    a = run_domain(adapter(base), as_of=as_of, now=NOW).to_dict()
    fut = base.copy()
    late = pd.to_datetime(fut["date"]) + pd.Timedelta(days=1) > pd.Timestamp(as_of)  # unpublished at as_of
    fut.loc[late, "value"] = 1.0
    more = cta_frame(end="2020-12-31").query("date > '2020-04-30'")
    b = run_domain(adapter(pd.concat([fut, more], ignore_index=True)), as_of=as_of, now=NOW).to_dict()
    assert int(late.sum()) > 0

    def view(d: dict) -> dict:
        return {
            "series": [s["points"] for s in d["series"]],
            "trends": [(t["id"], t["direction"], t["p_value"]) for t in d["trends"]],
            "anomalies": [(x["id"], x["score"]) for x in d["anomalies"]],
            "forecasts": [
                (f["id"], f["model"], f["points"], f["backtest"]) for f in d["predictions"]["forecasts"]
            ],
            "warnings": [(w["key"], w["severity"]) for w in d["warnings"]],
        }

    assert view(a) == view(b)


def test_plain_config_strips_only_seasonal_options():
    raw = cta_raw()
    plain = plain_config(raw)
    assert plain["thresholds"] == raw["thresholds"] and plain["source"] == raw["source"]
    assert all(not sp.get("ratio_lag") for sp in plain["series"])
    assert all(sp["anomaly_basis"] == "level" and sp["trend"] for sp in plain["series"])
    assert all("forecast_models" not in sp and "calendar" not in sp for sp in plain["series"])
    parse_config(plain, ROOT)


def test_seasonal_confirmation_rule():
    n = 1200
    per = pd.period_range("2016-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    base = (
        1000
        * np.array([1, 1, 1, 1, 1, 0.55, 0.4])[np.asarray(per.dayofweek)]
        * np.exp(rng.normal(0, 0.02, n))
    )
    last = 1000
    calm = weekly_yoy(base)
    assert seasonal_adverse_move(calm, last)[0] is False
    drop = base.copy()
    drop[last + 3 :] *= 0.7  # a sustained 30 % fall after the replay
    assert seasonal_adverse_move(weekly_yoy(drop), last)[0] is True
    blip = base.copy()
    blip[last + 8 : last + 12] *= 0.3  # a few bad days, recovered: not sustained
    assert seasonal_adverse_move(weekly_yoy(blip), last)[0] is False
    assert seasonal_adverse_move(calm, n - 10)[0] is None  # the future is not in the data


def test_entity_units_map_warnings_to_entities(collapse_run):
    units = entity_units({"result": collapse_run})
    assert {u["entity"] for u in units} == {"total", "bus", "rail", "Clark/Lake"}
    assert all(u["warned"] for u in units)
