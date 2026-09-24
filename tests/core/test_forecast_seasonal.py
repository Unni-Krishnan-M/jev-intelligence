"""Daily / weekly series building, season classes, seasonal forecasting (skill where it should and
should not exist), conformal coverage and the seasonal anomaly baseline, on synthetic data."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from jev_ml.core.anomalies import scan_series
from jev_ml.core.config import CoreConfig
from jev_ml.core.forecast import forecast_series, rolling_evaluation
from jev_ml.core.series import Series, SeriesSpec, build_series, derive, season_classes

DAY = 86400.0


def obs_frame(dates: pd.DatetimeIndex, values: np.ndarray, entity: str = "a") -> pd.DataFrame:
    ts = (dates.tz_localize("UTC") - pd.Timestamp("1970-01-01", tz="UTC")).total_seconds()
    return pd.DataFrame(
        {
            "timestamp": np.asarray(ts, dtype=float),
            "entity_id": f"x:{entity}",
            "entity_type": "x",
            "event_type": "observation",
            "value": values.astype(float),
            "source": "s",
            "entity": entity,
        }
    )


def daily_series(values: np.ndarray, start: str = "2018-01-01", **kw) -> Series:
    periods = pd.period_range(start, periods=len(values), freq="D")
    return Series(
        "s",
        "boardings",
        "a",
        "x",
        "n",
        periods,
        np.asarray(values, dtype=float),
        False,
        np.ones(len(values)),
        count_series=kw.pop("count", True),
        adverse_direction="down",
        forecast=True,
        **kw,
    )


CFG = CoreConfig(
    forecast_horizon=14,
    forecast_backtest_origins=56,
    forecast_history_months=364,
    forecast_min_history=56,
    anomaly_baseline_months=24,
    anomaly_min_baseline_months=12,
    anomaly_scan_months=28,
)
SEASONAL = ("naive", "seasonal_naive", "calendar_naive", "holt_winters", "holt_winters_calendar")
WEEKLY = np.array([1.0, 1.0, 1.0, 1.0, 0.95, 0.55, 0.4])  # Mon..Sun


def seasonal_counts(n: int, seed: int = 0, noise: float = 0.04, start: str = "2018-01-01") -> np.ndarray:
    rng = np.random.default_rng(seed)
    dow = pd.period_range(start, periods=n, freq="D").dayofweek
    level = 100_000 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    return np.round(level * WEEKLY[dow] * np.exp(rng.normal(0, noise, n)))


# ---------------------------------------------------------------------------------- series builder


def test_daily_grid_gaps_partial_and_duplicates():
    dates = pd.date_range("2020-01-01", "2020-03-31", freq="D")
    vals = np.arange(len(dates), dtype=float) + 1
    keep = ~dates.isin(pd.date_range("2020-02-10", "2020-02-12"))  # a 3-day gap
    obs = obs_frame(dates[keep], vals[keep])
    spec = SeriesSpec("b:{group}", "level", "n", group_by="entity", is_count=True)
    as_of = datetime(2020, 3, 20, 12, tzinfo=UTC)  # mid-day: 03-20 is the partial period
    (s,) = build_series(obs[obs["timestamp"] <= as_of.timestamp()], [spec], as_of, "D", "as_of")
    assert s.freq == "D" and s.partial_last and str(s.months[-1]) == "2020-03-20"
    m, v, _ = s.complete()
    assert str(m[-1]) == "2020-03-19" and len(m) == 79
    gap = (m >= pd.Period("2020-02-10", "D")) & (m <= pd.Period("2020-02-12", "D"))
    assert np.isnan(v[gap]).all() and np.isfinite(v[~gap]).all()  # gaps are NaN, never 0
    assert s.to_dict(10)["frequency"] == "day"
    # at midnight the day of as_of has not started: the previous day is the last, complete one
    (s2,) = build_series(obs, [spec], datetime(2020, 3, 20, tzinfo=UTC), "D", "as_of")
    assert not s2.partial_last and str(s2.months[-1]) == "2020-03-19"


def test_weekly_sum_needs_complete_weeks():
    dates = pd.date_range("2020-01-01", "2020-02-29", freq="D")  # starts on a Wednesday
    obs = obs_frame(dates, np.ones(len(dates)))
    obs = obs[obs["timestamp"] != pd.Timestamp("2020-01-22", tz="UTC").timestamp()]  # one missing day
    spec = SeriesSpec("w", "sum", "n", min_count=7)
    (s,) = build_series(obs, [spec], datetime(2020, 3, 2, tzinfo=UTC), "W", "as_of")
    m, v, c = s.complete()
    assert s.freq == "W" and s.to_dict(5)["frequency"] == "week"
    assert np.isnan(v[0])  # the first week has 5 days
    wk = int(np.flatnonzero(m == pd.Period("2020-01-22", "W"))[0])
    assert np.isnan(v[wk]) and c[wk] == 6  # a week with a missing day is not a low week
    assert np.isnan(v[-1])  # the data ends on Saturday 02-29: the last week has 6 days
    assert int(np.sum(v == 7.0)) == len(v) - 3


def test_derived_rolling_ratio():
    v = np.arange(1, 801, dtype=float)
    r = derive(v, 28, 364)
    assert np.isnan(r[: 364 + 27]).all()
    i = 500
    assert r[i] == pytest.approx(np.mean(v[i - 27 : i + 1]) / np.mean(v[i - 364 - 27 : i - 364 + 1]))


def test_season_classes_calendar():
    p = pd.period_range("2019-12-23", "2020-01-06", freq="D")
    plain = season_classes(p, 7, None)
    hol = season_classes(p, 7, "weekday_us_holidays")
    special = season_classes(p, 7, "weekday_us_special_days")
    assert list(plain[:7]) == [0, 1, 2, 3, 4, 5, 6]
    xmas, ny = p.get_loc(pd.Period("2019-12-25", "D")), p.get_loc(pd.Period("2020-01-01", "D"))
    assert plain[xmas] == 2 and hol[xmas] == 6 and hol[ny] == 6  # holidays run a Sunday service
    assert special[p.get_loc(pd.Period("2019-12-24", "D"))] == 7 and special[xmas] == 7
    assert special[p.get_loc(pd.Period("2020-01-02", "D"))] == 3
    # observance: 2021-07-04 is a Sunday -> Monday 2021-07-05; 2020-07-04 a Saturday -> Friday 07-03
    q = pd.PeriodIndex(["2021-07-05", "2020-07-03", "2020-07-04"], freq="D")
    assert list(season_classes(q, 7, "weekday_us_holidays")) == [6, 6, 5]
    w = pd.period_range("2019-12-23", periods=3, freq="W")
    assert list(season_classes(w, 52, None)) == [51, 0, 1]
    assert list(season_classes(pd.period_range("2020-11", periods=3, freq="M"), 12, None)) == [10, 11, 0]


def test_spec_validation():
    with pytest.raises(ValueError, match="naive benchmark"):
        SeriesSpec("a", "level", "n", seasonal_period=7, forecast_models=("holt_winters",))
    with pytest.raises(ValueError, match="need seasonal_period"):
        SeriesSpec("a", "level", "n", forecast_models=("naive", "holt_winters"))
    with pytest.raises(ValueError, match="unknown forecast models"):
        SeriesSpec("a", "level", "n", forecast_models=("naive", "prophet"))
    with pytest.raises(ValueError, match="calendar"):
        SeriesSpec("a", "level", "n", calendar="lunar")


# -------------------------------------------------------------------------------------- forecasts


def test_seasonal_models_beat_naive_on_seasonal_data():
    s = daily_series(seasonal_counts(1200), seasonal_period=7, forecast_models=SEASONAL)
    e = rolling_evaluation(s, CFG, start="2019-06-01", step=7)
    assert e is not None and e["issue_times"] >= 40
    assert e["relative_mae_vs_naive"] < 0.5
    assert e["relative_mae_vs_seasonal_naive"] < 1.0
    assert "naive" not in e["selected"]
    out, _ = forecast_series(s, CFG, "k", "v")
    assert out["model"] != "naive" and out["horizon_unit"] == "day" and len(out["points"]) == 14
    assert out["backtest"]["benchmark"] == "seasonal_naive"
    assert out["backtest"]["mase"] < out["backtest"]["random_walk_mase"]
    means = np.array([p["mean"] for p in out["points"]])
    dow = [pd.Timestamp(p["t"]).dayofweek for p in out["points"]]
    assert means[np.array(dow) == 6].mean() < 0.6 * means[np.array(dow) < 5].mean()  # Sundays low


def test_seasonal_models_do_not_beat_naive_without_seasonality():
    rng = np.random.default_rng(3)
    walk = 1000 + np.cumsum(rng.normal(0, 5, 1200))  # a random walk: naive is optimal
    s = daily_series(walk, count=False, seasonal_period=7, forecast_models=SEASONAL)
    e = rolling_evaluation(s, CFG, start="2019-06-01", step=7)
    assert e is not None
    assert e["relative_mae_vs_naive"] > 0.95  # no spurious skill from the seasonal candidates
    assert e["relative_mae_vs_seasonal_naive"] < 1.0  # seasonal naive is the worse benchmark here


def test_conformal_coverage_near_nominal():
    s = daily_series(seasonal_counts(1600, seed=5, noise=0.06), seasonal_period=7, forecast_models=SEASONAL)
    e = rolling_evaluation(s, CFG, start="2019-03-01", step=3)
    assert e is not None and e["interval_points"] > 3000
    assert 0.72 <= e["coverage80"] <= 0.88


def test_holiday_aware_models_on_holiday_data():
    n = 1400
    v = seasonal_counts(n, seed=9, noise=0.03)
    per = pd.period_range("2018-01-01", periods=n, freq="D")
    hol = season_classes(per, 7, "weekday_us_holidays") != per.dayofweek
    sunday_ratio = WEEKLY[6] / WEEKLY[np.asarray(per.dayofweek)]
    v = np.where(hol, np.round(v * sunday_ratio), v)
    s = daily_series(
        v,
        seasonal_period=7,
        calendar="weekday_us_holidays",
        forecast_models=("naive", "holt_winters", "holt_winters_calendar"),
    )
    both = rolling_evaluation(s, CFG, start="2019-01-01", step=1)
    plain = rolling_evaluation(s, CFG, start="2019-01-01", step=1, models=("naive", "holt_winters"))
    assert both is not None and plain is not None
    assert both["mae"] < plain["mae"]


def test_gaps_are_not_forecast_as_zero():
    v = seasonal_counts(700)
    v[600:630] = np.nan  # a month missing (the CTA data lacks July 2012)
    s = daily_series(v, seasonal_period=7, forecast_models=SEASONAL)
    out, _ = forecast_series(s, CFG, "k", "v")
    assert min(p["mean"] for p in out["points"]) > 20_000


def test_leak_free_forecast():
    v = seasonal_counts(900)
    s_a = daily_series(v[:800], seasonal_period=7, forecast_models=SEASONAL)
    later = v.copy()
    later[800:] = 1.0  # a collapse after the issue time
    s_b = daily_series(later, seasonal_period=7, forecast_models=SEASONAL)
    s_b.months, s_b.values, s_b.counts = s_b.months[:800], s_b.values[:800], s_b.counts[:800]
    assert forecast_series(s_a, CFG, "k", "v")[0] == forecast_series(s_b, CFG, "k", "v")[0]


# --------------------------------------------------------------------------------------- anomalies


def test_seasonal_anomaly_baseline_ignores_sundays():
    v = seasonal_counts(400, seed=2, noise=0.02)
    dow = pd.period_range("2018-01-01", periods=400, freq="D").dayofweek
    i = int(np.flatnonzero(dow == 2)[-1])  # a genuine drop on the last Wednesday
    v[i] = v[i] * 0.45
    level = daily_series(v, scan=True)
    seas = daily_series(v, scan=True, anomaly_basis="seasonal", seasonal_period=7)
    a_level = scan_series(level, CFG, "k", {})
    a_seas = scan_series(seas, CFG, "k", {})
    weekend = lambda a: pd.Timestamp(a["detected_at"]).dayofweek >= 5  # noqa: E731
    assert sum(weekend(a) for a in a_level) >= 6  # the trailing baseline flags weekends
    assert not any(weekend(a) for a in a_seas)
    assert [a["detected_at"] for a in a_seas] == [str(level.months[i])]
    assert a_seas[0]["kind"] == "series_drop" and a_seas[0]["adverse"] is True
