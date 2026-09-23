"""Method-level tests with synthetic data whose ground truth is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
from intel_synth import make_ratings, make_series

from jev_ml.intel.anomalies import isolation_scores, rater_features, scan_series, suppression
from jev_ml.intel.config import IntelConfig
from jev_ml.intel.forecast import backtest_model, honest_coverage
from jev_ml.intel.trends import bh_qvalues, change_point, trend_of

CFG = IntelConfig()


def test_trend_up_and_down_detected():
    rng = np.random.default_rng(1)
    t = np.arange(48)
    up = trend_of(make_series(100 + 5 * t + rng.normal(0, 8, 48)), CFG, "k")
    down = trend_of(make_series(400 - 5 * t + rng.normal(0, 8, 48)), CFG, "k")
    assert up["direction"] == "up" and up["slope_ci"][0] > 0 and up["p_value"] < 0.001
    assert down["direction"] == "down" and down["slope_ci"][1] < 0
    assert up["window"]["months"] == CFG.trend_window_months


def test_flat_noise_is_not_a_trend():
    flags = 0
    for seed in range(40):
        x = np.random.default_rng(seed).normal(200, 20, 48)
        flags += trend_of(make_series(x), CFG, "k")["direction"] != "flat"
    assert flags <= 5  # nominal 5 % -> ~2 of 40


def test_bh_qvalues_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.03, 0.5, 0.9])
    q = bh_qvalues(p)
    assert np.all(q >= p) and np.all(q <= 1)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)


def test_change_point_detected_and_located():
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(0, 1, 12), rng.normal(3, 1, 12)]
    cp = change_point(x, 4, 499, seed=7)
    assert cp is not None and cp["p_value"] < 0.01 and abs(cp["index"] - 12) <= 2
    false_alarms = sum(
        (cp0 := change_point(np.random.default_rng(100 + i).normal(0, 1, 24), 4, 199, seed=i)) is not None
        and cp0["p_value"] < 0.01
        for i in range(40)
    )
    assert false_alarms <= 3  # iid noise: nominal 1 %
    # determinism
    assert change_point(x, 4, 499, seed=7) == cp


def test_spike_detected_and_min_volume_guard():
    rng = np.random.default_rng(0)
    base = rng.normal(300, 30, 36)
    base[-1] = 1500
    anoms = scan_series(make_series(base), CFG, "k", {}, scan_months=1)
    assert len(anoms) == 1 and anoms[0]["kind"] == "series_spike" and not anoms[0]["suppressed"]
    assert anoms[0]["severity"] in ("medium", "high", "critical")
    small = np.clip(rng.normal(8, 2, 36), 1, None)
    small[-1] = 40
    guarded = scan_series(make_series(small), CFG, "k", {}, scan_months=1)
    assert len(guarded) == 1 and guarded[0]["suppressed"]
    assert "min-volume" in guarded[0]["suppression_reason"]


def test_drop_detected_and_quiet_series_not_flagged():
    rng = np.random.default_rng(5)
    x = rng.normal(500, 40, 36)
    x[-1] = 60
    a = scan_series(make_series(x), CFG, "k", {}, scan_months=1)
    assert a and a[0]["kind"] == "series_drop"
    quiet = scan_series(make_series(rng.normal(500, 40, 36)), CFG, "k", {}, scan_months=12)
    assert len(quiet) <= 1


def test_suppression_respects_escalation():
    assert suppression("k", "medium", {"k": "medium"}) is not None
    assert suppression("k", "high", {"k": "medium"}) is None  # escalated
    assert suppression("k", "critical", {"k": "false positive"}) is not None
    assert suppression("other", "high", {"k": "medium"}) is None


def test_injected_attackers_rank_above_genuine_users():
    ratings, movies = make_ratings(seed=2)
    rng = np.random.default_rng(9)
    items = movies["movie_id"].to_numpy()
    target = int(items[-1])
    rows = []
    for a in range(8):
        uid = 10_000 + a
        for m in rng.choice(items[:-1], size=40, replace=False):
            rows.append((uid, int(m), float(rng.choice([0.5, 1.0, 5.0])), int(1.3e9 + rng.integers(0, 1e5))))
        rows.append((uid, target, 5.0, int(1.3e9)))
    data = pd.concat([ratings, pd.DataFrame(rows, columns=ratings.columns)], ignore_index=True)
    feats = rater_features(data, CFG)
    s = isolation_scores(feats, CFG)
    attackers = s[s.index >= 10_000]
    genuine = s[s.index < 10_000]
    assert attackers.min() > genuine.quantile(0.9)


def test_forecast_intervals_cover_ar_data_near_nominal():
    covs = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        e = np.zeros(120)
        for t in range(1, 120):
            e[t] = 0.5 * e[t - 1] + rng.normal(0, 0.3)
        raw = np.expm1(6.0 + e)
        bt = backtest_model("moving_average", raw, "log1p", CFG, 6, 48)
        covs.append(honest_coverage(bt, CFG, 6))
    assert 0.65 <= float(np.mean(covs)) <= 0.95
