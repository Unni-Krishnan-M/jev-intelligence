"""Pipeline-level tests: contract shape, leakage, determinism, abstention, scenarios, validation."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from intel_synth import make_inputs

from jev_ml.intel import run_pipeline, run_scenario
from jev_ml.intel.config import IntelConfig

AS_OF = datetime(2013, 6, 15, tzinfo=UTC)


@pytest.fixture(scope="module")
def synth_inputs():
    return make_inputs(0)


@pytest.fixture(scope="module")
def synth_result(synth_inputs):
    return run_pipeline(synth_inputs)


def _normalise(d: dict) -> dict:
    """Drop fields that legitimately differ between runs of the same as_of: timings and the count
    of rows excluded *after* as_of (which is exactly what the leakage test changes)."""
    d = copy.deepcopy(d)
    d["run"].pop("stage_ms")
    d["data"].pop("excluded_after_as_of")
    for c in d["data"]["quality"]["checks"]:
        if c["name"].endswith("_after_as_of"):
            c["value"] = c["detail"] = None
    return d


def test_contract_top_level_and_json_strict(synth_result):
    d = synth_result.to_dict()
    for k in (
        "run",
        "data",
        "series",
        "signals",
        "trends",
        "anomalies",
        "predictions",
        "risks",
        "decisions",
        "warnings",
        "actions",
        "summary",
    ):
        assert k in d
    s = json.dumps(d, allow_nan=False)
    assert "NaN" not in s
    assert d["summary"]["status"] in ("nominal", "watch", "alert")
    assert set(d["summary"]["counts"]) >= {
        "signals",
        "trends_up",
        "trends_down",
        "anomalies",
        "risks_high",
        "warnings",
        "decisions",
        "actions",
    }
    assert d["run"]["pipeline_version"] == "intel-1.0.0"
    assert all(len(s["points"]) <= 120 for s in d["series"])
    for dec in d["decisions"]:
        assert dec["confidence_kind"] in ("probability", "margin", "rule")
        assert dec["kind"] in ("boolean", "choice", "score")
    for a in d["anomalies"]:
        assert a["severity"] in ("low", "medium", "high", "critical")
        assert a["method"] in ("robust_z", "isolation_forest", "rate_test")
    for t in d["trends"]:
        assert t["direction"] in ("up", "down", "flat") and len(t["slope_ci"]) == 2
    for f in d["predictions"]["forecasts"]:
        assert f["model"] in ("holt_damped", "moving_average", "naive")
        assert f["model_version"].startswith("fc-1.0.0-")
        assert {"origins", "mase", "smape", "mae", "coverage80", "naive_mase"} <= set(f["backtest"])
        for p in f["points"]:
            assert p["lo80"] is None or p["lo80"] <= p["mean"] <= p["hi80"]
    ev_keys = {"kind", "label", "value", "detail", "ref"}
    for r in d["risks"]:
        assert 0 <= r["score"] <= 100 and all(set(e) == ev_keys for e in r["evidence"])


def test_leakage_future_rows_do_not_change_result():
    base = make_inputs(0, as_of=AS_OF)
    r1 = run_pipeline(base).to_dict()
    inter = base.interactions
    future = inter[inter["timestamp"] <= AS_OF.timestamp()].sample(3000, random_state=1).copy()
    future["timestamp"] = int(AS_OF.timestamp()) + np.arange(len(future)) * 60 + 3600
    future["rating"] = 0.5  # extreme future behaviour must not leak into anything
    leaked = replace(base, interactions=pd.concat([inter, future], ignore_index=True))
    r2 = run_pipeline(leaked).to_dict()
    assert r2["data"]["excluded_after_as_of"] > r1["data"]["excluded_after_as_of"]
    assert _normalise(r1) == _normalise(r2)


def test_deterministic(synth_inputs):
    a = run_pipeline(synth_inputs).to_dict()
    b = run_pipeline(synth_inputs).to_dict()
    assert _normalise(a) == _normalise(b)
    assert [x["id"] for x in a["decisions"]] == [x["id"] for x in b["decisions"]]


def test_decisions_abstain_without_model_or_enough_data(synth_result):
    by = {d["key"]: d for d in synth_result.to_dict()["decisions"]}
    for key in ("retrain_model", "serving_model"):
        assert by[key]["abstained"] and by[key]["fallback_reason"] and by[key]["answer"] is None
    tiny = make_inputs(0)
    tiny = replace(tiny, interactions=tiny.interactions.head(400))
    d = {x["key"]: x for x in run_pipeline(tiny).to_dict()["decisions"]}
    assert d["reengagement_campaign"]["abstained"]
    lapse = run_pipeline(tiny).to_dict()["predictions"]["lapse"]
    assert lapse["status"] == "insufficient_data" and lapse["detail"]


def test_serving_decision_uses_bootstrap_probability(synth_inputs):
    rng = np.random.default_rng(0)
    per_user = {
        "hybrid": list(rng.normal(0.3, 0.1, 200)),
        "als": list(rng.normal(0.2, 0.1, 200)),
        "random": list(rng.normal(0.0, 0.01, 200)),
    }
    manifest = {
        "version": "m1",
        "model_type": "hybrid",
        "dataset_version": "synthetic-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "trained_on_rows": 1000,
        "data_cutoff_ts": synth_inputs.interactions["timestamp"].max(),
    }
    res = run_pipeline(replace(synth_inputs, model_manifest=manifest, per_user_ndcg=per_user)).to_dict()
    d = {x["key"]: x for x in res["decisions"]}
    s = d["serving_model"]
    assert s["answer"] == "hybrid" and s["confidence_kind"] == "probability" and s["confidence"] > 0.99
    assert "random" not in s["options"]
    assert abs(sum(s["option_scores"].values()) - 1) < 1e-6
    r = d["retrain_model"]
    assert r["answer"] == "no" and r["confidence_kind"] == "rule"
    kinds = {x["kind"] for x in res["risks"]}
    assert {"model_quality", "model_staleness"} <= kinds


def test_model_trained_after_as_of_is_not_replayed(synth_inputs):
    manifest = {
        "version": "m1",
        "model_type": "hybrid",
        "dataset_version": "synthetic-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "trained_on_rows": 1000,
        "data_cutoff_ts": synth_inputs.interactions["timestamp"].max(),
    }
    res = run_pipeline(replace(synth_inputs, model_manifest=manifest, as_of=AS_OF)).to_dict()
    d = {x["key"]: x for x in res["decisions"]}
    assert d["retrain_model"]["abstained"] and "after as_of" in d["retrain_model"]["fallback_reason"]
    assert not {"model_quality", "model_staleness"} & {r["kind"] for r in res["risks"]}


def test_scenario_math(synth_result):
    spec = {
        "series_id": "volume:all",
        "horizon_months": 12,
        "scenarios": [
            {"name": "c", "kind": "continue"},
            {"name": "r", "kind": "reverse", "trend_multiplier": -1.0},
            {"name": "s", "kind": "shock", "level_shift_pct": -20, "shock_month": 3},
        ],
    }
    out = run_scenario(synth_result, spec)
    sc = {s["name"]: s for s in out["scenarios"]}
    assert sc["c"]["points"] == out["baseline"]["points"] and sc["c"]["delta_vs_baseline"] == 0
    base = np.log1p([p["mean"] for p in out["baseline"]["points"]])
    rev = np.log1p([p["mean"] for p in sc["r"]["points"]])
    slope_b, slope_r = base[-1] - base[0], rev[-1] - rev[0]
    assert out["trend"]["slope_per_month"] and abs(slope_b) > 1e-9
    assert np.sign(slope_r) == -np.sign(slope_b)
    ratio = (1 + np.array([p["mean"] for p in sc["s"]["points"]])) / (
        1 + np.array([p["mean"] for p in out["baseline"]["points"]])
    )
    assert np.allclose(ratio[:2], 1.0) and np.allclose(ratio[2:], 0.8, atol=1e-3)
    assert [c["rank"] for c in out["comparison"]] == [1, 2, 3]
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize(
    "spec",
    [
        {"series_id": "volume:all", "horizon_months": 0},
        {"series_id": "volume:all", "horizon_months": 25},
        {"series_id": "nope", "horizon_months": 6},
        {
            "series_id": "volume:all",
            "horizon_months": 6,
            "scenarios": [{"kind": "continue", "name": str(i)} for i in range(7)],
        },
        {"series_id": "volume:all", "horizon_months": 6, "scenarios": [{"kind": "explode"}]},
        {"series_id": "volume:all", "horizon_months": 6, "scenarios": [{"kind": "shock"}]},
    ],
)
def test_scenario_validation(synth_result, spec):
    with pytest.raises(ValueError):
        run_scenario(synth_result, spec)


def test_validation_catches_bad_rows(synth_inputs):
    inter = synth_inputs.interactions
    bad = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 3, 4],
            "movie_id": [1, 99999, 2, 3, 5],
            "rating": [7.0, 4.0, np.nan, 3.3, 4.0],
            "timestamp": [1.3e9, 1.3e9, 1.3e9, 1.3e9, 1.0e8],
        }
    )
    dup = inter.head(5)
    res = run_pipeline(
        replace(synth_inputs, interactions=pd.concat([inter, bad, dup], ignore_index=True))
    ).to_dict()
    checks = {c["name"]: c for c in res["data"]["quality"]["checks"]}
    for name, n in (
        ("movielens_rating_range", 1),
        ("movielens_unknown_movies", 1),
        ("movielens_null_rate", None),
        ("movielens_rating_grid", 1),
        ("movielens_timestamp_range", 1),
        ("movielens_duplicates", 5),
    ):
        assert not checks[name]["passed"], name
        if n is not None:
            assert checks[name]["value"] == n, name
    assert res["data"]["quality"]["score"] < 1
    assert "data_quality" in {r["kind"] for r in res["risks"]}
    # invalid rows removed, duplicates kept once, the off-grid (3.3) rating kept (warning only)
    assert res["data"]["sources"][0]["rows"] == len(inter) + 1


def _feedback(synth_inputs):
    now = synth_inputs.now.timestamp()
    rng = np.random.default_rng(0)
    prior_ts = now - rng.uniform(8, 35, 400) * 86400
    rec_ts = now - rng.uniform(0, 7, 200) * 86400
    fb = pd.DataFrame(
        {
            "feedback": list(rng.choice(["like", "clicked", "dislike"], 400, p=[0.5, 0.4, 0.1]))
            + list(rng.choice(["like", "clicked", "dislike"], 200, p=[0.3, 0.3, 0.4])),
            "timestamp": np.r_[prior_ts, rec_ts],
        }
    )
    return fb


def test_live_feedback_rate_test(synth_inputs):
    res = run_pipeline(replace(synth_inputs, app_feedback=_feedback(synth_inputs))).to_dict()
    live = [a for a in res["anomalies"] if a["kind"] == "live_feedback"]
    assert live and live[0]["severity"] in ("high", "critical")
    app = next(s for s in res["data"]["sources"] if s["source"] == "app")
    assert app["fresh"] is True and app["expected_update"] == "daily"
    assert "recommendation_rejection" in {r["kind"] for r in res["risks"]}


def test_config_to_dict_is_json():
    json.dumps(IntelConfig().to_dict(), allow_nan=False)


def test_dismissed_warning_suppressed_unless_escalated(synth_inputs):
    inp = replace(synth_inputs, app_feedback=_feedback(synth_inputs))
    warns = {w["key"]: w for w in run_pipeline(inp).to_dict()["warnings"]}
    key = "anomaly:live_feedback:app"
    assert key in warns
    sev = warns[key]["severity"]
    same = run_pipeline(replace(inp, suppressed_keys={key: sev})).to_dict()
    assert key not in {w["key"] for w in same["warnings"]}
    assert next(a for a in same["anomalies"] if a["dedup_key"] == key)["suppressed"]
    lower = run_pipeline(replace(inp, suppressed_keys={key: "low"})).to_dict()
    assert key in {w["key"] for w in lower["warnings"]}  # escalated since dismissal
