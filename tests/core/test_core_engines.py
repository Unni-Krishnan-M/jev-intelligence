"""Core engines on synthetic data: series builder, trend, anomaly, forecast, risk, signals, the
early-warning decision (monotonicity, abstention, margins) and warnings downstream of it."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from jev_ml.core import risk as rk
from jev_ml.core.anomalies import scan_series
from jev_ml.core.config import EWL_LEVELS, CoreConfig
from jev_ml.core.early_warning import (
    Situation,
    aggregate,
    build_early_warning,
    build_situations,
    decide,
    level_of_points,
    option_scores,
    severity_points,
    situation_state,
)
from jev_ml.core.forecast import forecast_series
from jev_ml.core.series import Series, SeriesSpec, build_series
from jev_ml.core.signals import build_signals
from jev_ml.core.trends import analyse_trends
from jev_ml.core.warnings import build_warnings

CFG = CoreConfig()
KEY = "2020-01-01T00:00:00Z"


def series(values, adverse="up", metric="level", sid="level:X", count=False) -> Series:
    months = pd.period_range("2012-01", periods=len(values), freq="M")
    v = np.asarray(values, dtype=float)
    return Series(
        sid,
        metric,
        "X",
        "area",
        "%",
        months,
        v,
        False,
        np.ones(len(v)),
        count_series=count,
        adverse_direction=adverse,
        source="test",
        volume_guard=False,
        forecast=True,
    )


# ------------------------------------------------------------------------------------------ series


def _obs():
    rows = []
    base = pd.Timestamp("2019-01-01", tz="UTC")
    for m in range(6):
        t = (base + pd.DateOffset(months=m)).timestamp() + 3600
        for i in range(m + 1):  # month m has m+1 events
            rows.append((t + i, f"user:{i % 2}", "rating", float(i), ("a", "b") if i % 2 else ("a",), i % 2))
    return pd.DataFrame(rows, columns=["timestamp", "entity_id", "event_type", "value", "tag", "uid"])


def test_series_builder_metrics_and_partial_month():
    obs = _obs()
    specs = [
        SeriesSpec("n:all", "count", "events"),
        SeriesSpec("u:all", "nunique", "users", value_column="uid"),
        SeriesSpec("n:{group}", "count", "events", group_by="tag", entity_type="tag"),
        SeriesSpec("s:{group}", "share", "share", group_by="tag", entity_type="tag"),
        SeriesSpec("m:{group}", "mean", "avg", group_by="tag", entity_type="tag", min_count=2),
    ]
    as_of = datetime(2019, 6, 15, tzinfo=UTC)  # June is partial
    out = {s.id: s for s in build_series(obs, specs, as_of)}
    assert [s for s in out] == ["n:all", "u:all", "n:a", "s:a", "m:a", "n:b", "s:b", "m:b"]
    assert out["n:all"].partial_last and list(out["n:all"].values) == [1, 2, 3, 4, 5, 6]
    assert list(out["u:all"].values) == [1, 2, 2, 2, 2, 2]
    assert list(out["n:b"].values) == [0, 1, 1, 2, 2, 3]  # odd i carry tag b (lists count per group)
    assert np.allclose(out["s:b"].values, np.array([0, 1, 1, 2, 2, 3]) / np.arange(1, 7))
    m = out["m:b"].values  # mean of odd i, NaN when fewer than 2 events
    assert np.isnan(m[0]) and np.isnan(m[1]) and m[3] == 2.0
    complete, _, _ = out["n:all"].complete()
    assert str(complete[-1]) == "2019-05"
    # last_observation grid: nothing partial, ends at the latest observation
    lo = build_series(obs, specs[:1], datetime(2020, 1, 1, tzinfo=UTC), grid_end="last_observation")[0]
    assert not lo.partial_last and str(lo.months[-1]) == "2019-06"


def test_series_builder_weekly():
    obs = _obs()
    s = build_series(
        obs, [SeriesSpec("n:all", "count", "events")], datetime(2019, 7, 1, tzinfo=UTC), freq="W"
    )[0]
    assert s.freq == "W" and s.values.sum() == 21
    assert s.to_dict(5)["points"][0]["t"].count("-") == 2


# --------------------------------------------------------------------------- trend/anomaly/forecast


def test_trend_detects_adverse_rise_with_additive_fields():
    rng = np.random.default_rng(0)
    s = series(5 + 0.05 * np.arange(48) + rng.normal(0, 0.05, 48))
    (t,) = analyse_trends([s], CFG, KEY)
    assert t["direction"] == "up" and t["q_value"] < 0.01
    assert t["velocity"] == t["slope"] and t["baseline"] == t["prior_mean"]
    assert t["magnitude"] == pytest.approx(t["recent_mean"] - t["prior_mean"], abs=1e-5)
    assert t["confidence"] == pytest.approx(1 - t["q_value"]) and t["confidence_kind"] == "evidence"
    assert t["supporting_observations"] == 24


def test_anomaly_spike_scored_and_adverse():
    rng = np.random.default_rng(1)
    v = 5 + rng.normal(0, 0.1, 40)
    v[-1] = 8.0
    (a,) = [x for x in scan_series(series(v), CFG, KEY, {}) if x["detected_at"] == "2015-04-01"]
    assert a["kind"] == "series_spike" and a["adverse"] is True and a["severity"] == "critical"
    assert 0 < a["anomaly_score"] <= 1 and a["expected_value"] == a["baseline"]
    down = scan_series(series(v, adverse="down"), CFG, KEY, {})
    assert down[-1]["adverse"] is False


def test_forecast_backtest_and_horizon_unit():
    rng = np.random.default_rng(2)
    s = series(5 + np.cumsum(rng.normal(0, 0.1, 60)))
    out, state = forecast_series(s, CFG, KEY, "v")
    assert out["horizon_unit"] == "month" and len(out["points"]) == CFG.forecast_horizon
    assert out["backtest"]["origins"] > 0 and state.history.shape[0] == 60


# ------------------------------------------------------------------------------------------- risk


def test_risk_scoring_and_generic_trend_risk():
    r = rk.make_risk("k", "area", "X", "t", 0.8, 0.5, 1.0, 1.0, {}, [], "r", CFG, KEY)
    assert (
        r["score"] == 40.0
        and r["level"] == "high" == r["severity"]
        and r["contributing_factors"] == r["factors"]
    )
    rng = np.random.default_rng(3)
    s = series(5 + 0.05 * np.arange(48) + rng.normal(0, 0.05, 48))
    trends = analyse_trends([s], CFG, KEY)
    (tr,) = rk.trend_risks(trends, {s.id: s}, 1.0, CFG, KEY)
    assert tr["kind"] == "adverse_trend" and tr["series_id"] == s.id
    assert "measured" in tr["factors"][1]["detail"]
    declared = CoreConfig(impact_weights={"X": 0.3})
    (td,) = rk.trend_risks(trends, {s.id: s}, 1.0, declared, KEY)
    assert td["impact"] == 0.3 and "declared" in td["factors"][1]["detail"]
    # a favourable trend is not a risk
    s_down = series(s.values, adverse="down")
    assert rk.trend_risks(analyse_trends([s_down], CFG, KEY), {s.id: s_down}, 1.0, CFG, KEY) == []


def test_signals_carry_confidence_and_baseline():
    rng = np.random.default_rng(4)
    s = series(5 + 0.05 * np.arange(48) + rng.normal(0, 0.05, 48))
    trends = analyse_trends([s], CFG, KEY)
    sigs = build_signals(trends, [], [], {}, {s.id: s}, {"checks": []}, [], 1.6e9, KEY, KEY, CFG, KEY)
    (t,) = [x for x in sigs if x["kind"] == "trend"]
    assert t["confidence_kind"] == "evidence" and t["baseline"] == trends[0]["prior_mean"]
    assert t["entity_id"] == "area:X" and t["source"] == "test"


# ---------------------------------------------------------------------------- early-warning decision


def _risk_comp(score: float, **kw):
    return {
        "stage": "risk",
        "stage_group": "risk",
        "ref": "r",
        "points": severity_points(score, {"low": 5, "medium": 15, "high": 30, "critical": 50}, CFG),
        "detail": "",
        "title": "risk",
        "severity": "low",
        "warnable": False,
        "suppressed": False,
        **kw,
    }


def _comp(stage, points, group=None):
    return {
        "stage": stage,
        "stage_group": group or stage,
        "ref": stage,
        "points": points,
        "detail": "",
        "title": stage,
        "severity": "low",
        "warnable": False,
        "suppressed": False,
        "q_value": 0.0,
    }


def test_points_map_is_monotone_and_hits_band_edges():
    bands = {"low": 3.5, "medium": 5.0, "high": 7.0, "critical": 10.0}
    xs = np.linspace(0, 30, 601)
    pts = [severity_points(x, bands, CFG) for x in xs]
    assert all(b >= a for a, b in itertools.pairwise(pts))
    assert severity_points(5.0, bands, CFG) == 3.0 and severity_points(7.0, bands, CFG) == 5.0
    assert max(pts) <= CFG.ewl_max_points
    assert level_of_points(severity_points(4.99, bands, CFG), CFG) == "MONITOR"
    assert level_of_points(severity_points(5.0, bands, CFG), CFG) == "WARNING"


@pytest.mark.parametrize("seed", range(20))
def test_adding_or_strengthening_evidence_never_lowers_the_level(seed):
    rng = np.random.default_rng(seed)
    stages = ["trend", "anomaly", "forecast", "risk"]
    comps = [_comp(stages[i % 4], float(rng.uniform(0, 8))) for i in range(rng.integers(1, 5))]
    order = list(EWL_LEVELS)
    for corr in (0.0, 1.0):
        cfg = CoreConfig(ewl_corroboration=corr)
        p0, _, _ = aggregate(comps, cfg)
        lv0 = order.index(level_of_points(p0, cfg))
        more = [*comps, _comp(stages[int(rng.integers(0, 4))], float(rng.uniform(0, 8)))]
        p1, _, _ = aggregate(more, cfg)
        assert p1 >= p0 and order.index(level_of_points(p1, cfg)) >= lv0
        stronger = [dict(c, points=min(8.0, c["points"] + float(rng.uniform(0, 2)))) for c in comps]
        p2, _, _ = aggregate(stronger, cfg)
        assert p2 >= p0 and order.index(level_of_points(p2, cfg)) >= lv0


def test_option_scores_margin_and_boundaries():
    for p in np.linspace(0, 8, 81):
        sc = option_scores(float(p), CFG)
        assert abs(sum(sc.values()) - 1) < 1e-9 and min(sc.values()) >= 0
    mid = decide(situation_state(Situation("s", "area", "X", components=[_comp("anomaly", 4.0)])), CFG, KEY)
    edge = decide(situation_state(Situation("s", "area", "X", components=[_comp("anomaly", 3.0)])), CFG, KEY)
    assert mid["answer"] == edge["answer"] == "WARNING"
    assert mid["confidence"] == 1.0 and edge["confidence"] == 0.0 and mid["confidence_kind"] == "margin"


def test_corroboration_and_context_caps():
    two = [_comp("trend", 2.5), _comp("forecast", 2.0)]
    assert level_of_points(aggregate(two, CoreConfig(ewl_corroboration=0.0))[0], CFG) == "MONITOR"
    assert level_of_points(aggregate(two, CoreConfig(ewl_corroboration=1.0))[0], CFG) == "WARNING"
    # risks never count as corroboration (they are derived from the detection stages)
    assert aggregate([_comp("trend", 2.0), _risk_comp(10.0)], CoreConfig(ewl_corroboration=1.0))[2] == [
        "trend"
    ]


def test_abstains_when_only_skipped_stages():
    sit = Situation("series:x", "area", "X", "x", "up", "X", [], [{"stage": "trend", "reason": "short"}])
    d = decide(situation_state(sit), CFG, KEY)
    assert d["abstained"] and d["answer"] is None and "skipped" in d["fallback_reason"]
    assert d["confidence"] is None


def _pipeline_bits(v, adverse="up"):
    s = series(v, adverse=adverse)
    by = {s.id: s}
    trends = analyse_trends([s], CFG, KEY)
    anoms = scan_series(s, CFG, KEY, {})
    risks = rk.anomaly_risks(anoms, by, "2015-04-01", 1.0, CFG, KEY) + rk.trend_risks(
        trends, by, 1.0, CFG, KEY
    )
    sits = build_situations(by, trends, anoms, [], {}, risks, {}, "2015-04-01", CFG)
    return by, trends, anoms, risks, sits


def test_warnings_are_downstream_of_the_decision_and_carry_its_id():
    rng = np.random.default_rng(5)
    v = 5 + rng.normal(0, 0.1, 40)
    v[-1] = 8.0
    _, _, anoms, risks, sits = _pipeline_bits(v)
    decisions, batch = build_early_warning(sits, CFG, KEY)
    assert batch["name"] == "early_warning" and batch["status"] == "ok"
    (d,) = decisions
    assert d["answer"] == "URGENT_ACTION" and d["state"]["anomaly"]["severity"] == "critical"
    for stage in ("signal", "trend", "anomaly", "forecast", "risk"):
        assert stage in d["state"]
    (w,) = build_warnings(decisions, risks, anoms, {}, CFG)
    assert w["decision_id"] == d["id"] and w["early_warning_level"] == "URGENT_ACTION"
    assert w["severity"] == "critical" and w["key"] == f"warning:{d['situation']}"
    # the same spike in the favourable direction: context only (MONITOR), no warning
    _, _, anoms2, risks2, sits2 = _pipeline_bits(v, adverse="down")
    decisions2, _ = build_early_warning(sits2, CFG, KEY)
    assert decisions2[0]["answer"] == "MONITOR" and build_warnings(decisions2, risks2, anoms2, {}, CFG) == []
    # an operator dismissal at this severity suppresses the warning (not the decision)
    assert build_warnings(decisions, risks, anoms, {w["key"]: "critical"}, CFG) == []


# ------------------------------------------------------------------------------------------- evaluation


def test_series_adverse_move_outcome():
    from jev_ml.core.evaluation import series_adverse_move

    rng = np.random.default_rng(6)
    flat = 5 + np.round(rng.normal(0, 0.05, 40), 2)
    up = flat.copy()
    up[31:] += 1.0
    assert series_adverse_move(up, 30, 6, "up")[0] is True
    assert series_adverse_move(up, 30, 6, "down")[0] is False
    assert series_adverse_move(flat, 30, 6, "up")[0] is False
    assert series_adverse_move(flat, 36, 6, "up")[0] is None  # horizon beyond the data


def test_confusion_and_flip_rate():
    from jev_ml.core.evaluation import confusion, decision_consistency

    units = [
        {"warned": True, "confirmed": True},
        {"warned": True, "confirmed": False},
        {"warned": False, "confirmed": False},
        {"warned": False, "confirmed": True},
        {"warned": True, "confirmed": None},
    ]
    c = confusion(units)
    assert (c["tp"], c["fp"], c["fn"], c["tn"], c["observable"]) == (1, 1, 1, 1, 4)
    assert c["precision"] == 0.5 and c["false_positive_rate"] == 0.5

    def rep(levels):
        return {
            "result": {
                "decisions": [
                    {"key": "early_warning_level", "situation": k, "entity": k, "answer": v}
                    for k, v in levels.items()
                ]
            }
        }

    cons = decision_consistency([rep({"a": "WARNING"}), rep({"a": "WARNING", "b": "MONITOR"}), rep({})])
    # pairs: (a W->W, b N->M), (a W->N, b M->N) -> 3 flips of 4, 1 across the WARNING boundary
    assert cons["flips"] == 3 and cons["situation_pairs"] == 4 and cons["warning_boundary_flips"] == 1


def test_core_never_imports_the_recommender_or_a_domain():
    import ast
    from pathlib import Path

    core = Path(__file__).resolve().parents[2] / "ml" / "jev_ml" / "core"
    banned = (
        "jev_ml.models",
        "jev_ml.engine",
        "jev_ml.training",
        "jev_ml.calibration",
        "jev_ml.domains",
        "jev_ml.intel",
        "jev_ml.data",
        "jev_ml.features",
        "jev_ml.signals",
        "jev_ml.evaluation",
    )
    for py in core.glob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            )
            for n in names:
                assert not n.startswith(banned), f"{py.name} imports {n}"
