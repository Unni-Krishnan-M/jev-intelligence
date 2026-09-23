"""Contract 9.1: score decisions (editorial_slot_share) and multi-question decision batches."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from intel_synth import make_inputs, make_series

from jev_ml.intel import run_pipeline
from jev_ml.intel.batches import Question, run_batch, state_hash
from jev_ml.intel.config import IntelConfig
from jev_ml.intel.decisions import POLICY_VERSIONS
from jev_ml.intel.forecast import forecast_series, window_mean_forecast

BATCH_KEYS = {
    "model_governance": ["retrain_model", "serving_model"],
    "genre_programming": ["genre_programming", "editorial_slot_share"],
    "audience": ["rater_action", "reengagement_campaign"],
}


@pytest.fixture(scope="module")
def result():
    return run_pipeline(make_inputs(0)).to_dict()


def _slots(d: dict) -> list[dict]:
    return [x for x in d["decisions"] if x["key"] == "editorial_slot_share"]


def test_score_decision_contract(result):
    slots = _slots(result)
    assert slots, "the synthetic data has four genres with share forecasts"
    answered = [x for x in slots if not x["abstained"]]
    assert answered
    for x in answered:
        assert x["kind"] == "score" and x["options"] == [] and x["option_scores"] == {}
        assert isinstance(x["answer"], float) and 0.0 <= x["answer"] <= 100.0
        assert x["scale"] == {"min": 0.0, "max": 100.0, "unit": "% of home-rail slots"}
        lo, hi = x["answer_interval"]
        assert 0.0 <= lo <= hi <= 100.0
        assert x["confidence_kind"] == "interval" and x["confidence"] == IntelConfig().forecast_interval
        st = x["state"]
        # the answer is exactly the forecast window mean of the share, in percent
        assert x["answer"] == pytest.approx(100 * min(1.0, max(0.0, st["forecast_mean_share"])), abs=0.006)
        assert len(st["window_months"]) == IntelConfig().slot_share_window_months
        assert 0.0 <= st["interval_backtest_coverage"] <= 1.0
        assert x["policy_version"] == "slot-share-1.0.0"
        assert any(e["ref"] == st["series_id"] for e in x["evidence"])
    # non-score decisions carry the new fields as null
    for x in result["decisions"]:
        if x["kind"] != "score":
            assert x["scale"] is None and x["answer_interval"] is None
    # share forecasts are published next to the volume forecasts
    sids = {f["series_id"] for f in result["predictions"]["share_forecasts"]}
    assert {x["state"]["series_id"] for x in answered} <= sids


def test_score_decision_abstains_on_untrustworthy_interval():
    cfg = IntelConfig(slot_share_min_coverage=1.01)  # no backtested coverage can reach this
    d = run_pipeline(make_inputs(0), cfg).to_dict()
    slots = _slots(d)
    assert slots and all(x["abstained"] for x in slots)
    for x in slots:
        assert x["answer"] is None and x["confidence"] is None and x["answer_interval"] is None
        assert "not trustworthy" in x["fallback_reason"]
        assert x["confidence_kind"] == "interval" and x["kind"] == "score"


def test_score_decision_abstains_without_forecast():
    cfg = IntelConfig(forecast_min_history=500)  # no series is long enough to forecast
    d = run_pipeline(make_inputs(0), cfg).to_dict()
    assert d["predictions"]["share_forecasts"] == []
    slots = _slots(d)
    assert slots and all(x["abstained"] and "no share forecast" in x["fallback_reason"] for x in slots)


def test_window_mean_interval_is_calibrated_on_a_stationary_series():
    rng = np.random.default_rng(3)
    share = 0.2 + rng.normal(0, 0.01, 90)
    s = make_series(share, metric="share", sid="share:genre:Test")
    cfg = IntelConfig()
    out = forecast_series(s, cfg, "k", "v")
    assert out is not None
    w = window_mean_forecast(out[1], 3, cfg)
    assert w is not None
    assert w["lo"] <= w["mean"] <= w["hi"] and 0.15 < w["mean"] < 0.25
    assert w["n_residuals"] >= cfg.forecast_min_residuals
    assert w["coverage"] is not None and w["coverage"] >= 0.6
    # count series (log scale) are not supported: the window mean of a log forecast is not a share
    vol = forecast_series(make_series(rng.poisson(100, 90).astype(float)), cfg, "k", "v")
    assert vol is not None and window_mean_forecast(vol[1], 3, cfg) is None


def test_batches_contract(result):
    batches = {b["name"]: b for b in result["decision_batches"]}
    assert set(batches) == set(BATCH_KEYS)
    by_id = {x["id"]: x for x in result["decisions"]}
    covered = []
    for name, b in batches.items():
        assert b["keys"] == BATCH_KEYS[name]
        assert b["id"].startswith("batch-") and b["status"] == "ok" and b["failure_reason"] is None
        assert len(b["state_hash"]) == 40 and int(b["state_hash"], 16) >= 0
        assert b["policy_versions"] == {k: POLICY_VERSIONS[k] for k in b["keys"]}
        for did in b["decision_ids"]:
            dec = by_id[did]
            assert dec["batch_id"] == b["id"] and dec["key"] in b["keys"]
            assert dec["policy_version"] == b["policy_versions"][dec["key"]]
        assert b["n_decisions"] == len(b["decision_ids"])
        covered += b["decision_ids"]
    # every decision belongs to exactly one batch
    assert sorted(covered) == sorted(by_id) and len(covered) == len(set(covered))
    json.dumps(result["decision_batches"], allow_nan=False)


def test_batches_deterministic_and_state_sensitive():
    a = run_pipeline(make_inputs(0)).to_dict()["decision_batches"]
    b = run_pipeline(make_inputs(0)).to_dict()["decision_batches"]
    assert a == b
    other = run_pipeline(make_inputs(1)).to_dict()["decision_batches"]
    ha = {x["name"]: x["state_hash"] for x in a}
    hb = {x["name"]: x["state_hash"] for x in other}
    assert ha["genre_programming"] != hb["genre_programming"]


def test_batch_ids_depend_on_as_of_only():
    inp = make_inputs(0)
    base = {x["name"]: x["id"] for x in run_pipeline(inp).to_dict()["decision_batches"]}
    # a different config changes answers but not the ids (ids = name + as_of)
    again = {
        x["name"]: x["id"]
        for x in run_pipeline(inp, IntelConfig(slot_share_min_coverage=1.01)).to_dict()["decision_batches"]
    }
    assert base == again
    shifted = replace(inp, as_of=pd.Timestamp("2013-06-15", tz="UTC"))
    assert {x["name"]: x["id"] for x in run_pipeline(shifted).to_dict()["decision_batches"]} != base


def _q(key: str, policy) -> Question:
    return Question(
        key, f"{key}-1.0.0", "boolean", "rule", f"{key}?", policy, lambda st: [("platform", "all")]
    )


def _ok(key: str):
    def policy(st):
        return [
            {
                "id": f"dec-{key}",
                "key": key,
                "abstained": False,
                "answer": "yes" if st["x"] > 0 else "no",
                "batch_id": None,
            }
        ]

    return policy


def test_run_batch_shares_one_snapshot_and_sets_ids():
    seen: list[int] = []

    def spy(st):
        seen.append(id(st))
        return _ok("b")(st)

    state = {"x": 1, "frame": pd.DataFrame({"a": [1.0, 2.0]})}
    ds, rec = run_batch(
        "t", "q", state, [_q("a", lambda st: (seen.append(id(st)), _ok("a")(st))[1]), _q("b", spy)], "k"
    )
    assert len(set(seen)) == 1, "every question must see the same snapshot object"
    assert rec["status"] == "ok" and rec["keys"] == ["a", "b"] and rec["decision_ids"] == ["dec-a", "dec-b"]
    assert all(d["batch_id"] == rec["id"] for d in ds)
    assert rec["state_hash"] == state_hash(state)


def test_run_batch_is_atomic_on_failure():
    def boom(st):
        raise RuntimeError("policy exploded")

    ds, rec = run_batch("t", "q", {"x": 1}, [_q("a", _ok("a")), _q("b", boom)], "k")
    assert rec["status"] == "failed" and "policy exploded" in rec["failure_reason"]
    # the answer of "a" (computed before "b" failed) is not kept: both questions abstain
    assert [d["key"] for d in ds] == ["a", "b"]
    assert all(d["abstained"] and d["answer"] is None and d["batch_id"] == rec["id"] for d in ds)
    assert all(d["fallback_reason"] == rec["failure_reason"] for d in ds)


def test_run_batch_detects_state_mutation_and_protects_inputs():
    def mutate(st):
        st["x"] = -5
        return _ok("b")(st)

    state = {"x": 1}
    ds, rec = run_batch("t", "q", state, [_q("a", _ok("a")), _q("b", mutate)], "k")
    assert rec["status"] == "failed" and "changed the shared state" in rec["failure_reason"]
    assert all(d["abstained"] for d in ds)
    assert state == {"x": 1}, "the caller's objects are never mutated (the snapshot is a copy)"


def test_state_hash_canonical():
    a = {"b": [1, 2.5, None], "a": {"y": np.float64(0.1), "x": np.int64(3)}}
    b = {"a": {"x": 3, "y": 0.1}, "b": [1, 2.5, None]}
    assert state_hash(a) == state_hash(b)
    f1 = pd.DataFrame({"p": [0.1, 0.2]})
    f2 = f1.copy()
    f2.loc[1, "p"] = 0.2000001
    assert state_hash({"f": f1}) != state_hash({"f": f2})
    assert state_hash({"f": f1}) == state_hash({"f": f1.copy()})
