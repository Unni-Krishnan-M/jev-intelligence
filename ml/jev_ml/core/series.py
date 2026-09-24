"""UNDERSTAND: periodic series built from generic observations, driven by ``SeriesSpec``s.

Observation frame (see ``jev_ml.core.types.OBSERVATION_COLUMNS``): ``timestamp`` (UNIX seconds),
``entity_id``, ``entity_type``, ``event_type``, ``value`` (float, NaN allowed), ``source`` and any
group columns. A group column may hold a scalar or a list/tuple (an observation then belongs to
every listed group, e.g. a film's genres; each group counts it once).

Metrics per period and group:

* ``count``: observations in the group; ``share``: group count / all observations of the frame
  (after the spec's event filter) in the period, NaN when the period has none;
* ``mean``: mean ``value`` of the group's observations, NaN when fewer than ``min_count``;
* ``nunique``: distinct values of ``value_column`` (e.g. ``entity_id`` = active entities);
* ``level``: mean ``value`` (NaN when absent), for data that already is a level per period
  (e.g. an unemployment rate); equals ``mean`` with ``min_count = 1``;
* ``sum``: sum of ``value`` (e.g. daily boardings summed to weeks), NaN when fewer than
  ``min_count`` observations fall in the period (a week with a missing day is not a low week).

Derived values (opt-in, applied after the metric): ``rolling = k`` replaces each point by the mean
of the trailing k periods (NaN unless all k are defined), and ``ratio_lag = L`` then divides by the
same quantity L periods earlier (e.g. ``rolling: 28, ratio_lag: 364``: the last four weeks against
the same, weekday-aligned four weeks a year earlier; weekly and yearly seasonality cancel).

Periods: calendar months (``freq="M"``, required), weeks (``freq="W"``, Monday..Sunday) or days
(``freq="D"``). Grid end:

* ``"as_of"``: a period is *complete* when the first instant of the next period is <= as_of; the
  period containing as_of (if as_of is past its first instant) is the last point with
  ``partial: true`` and is excluded from every baseline, trend, anomaly and forecast;
* ``"last_observation"``: the grid ends at the period of the latest observation, which is complete
  (published statistics: a released monthly value is final for this purpose).

Series order: specs in order, except that consecutive specs with the same ``group_by`` form a block
emitted group by group (for each group, every spec of the block), so an entity's series are adjacent.
``counts`` holds the observations behind each point (the min-volume guard's input). Gaps are
NaN points (never zeros for ``level``/``mean``/``sum``); duplicates are removed by the core validator
before building.

Seasonality (opt-in per spec; every default reproduces the monthly behaviour byte for byte):
``seasonal_period`` (7 for daily data with a weekly cycle, 52 weekly, 12 monthly) and ``calendar``
define each period's *season class* (``season_classes``): day of week for days (with
``calendar: weekday_us_holidays`` the six US holidays a transit operator runs a Sunday service on
count as Sundays), week of year for weeks, month for months. Forecasting (``forecast_models``) and
``anomaly_basis: seasonal`` (a point is compared with the previous points of its own class, so a
Sunday is compared with Sundays) read them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import fnum, period_str

COUNT_METRICS = ("volume", "active_users", "count", "nunique")
METRICS = ("count", "share", "mean", "nunique", "level", "sum")
FREQS = {"M": "M", "month": "M", "W": "W", "week": "W", "D": "D", "day": "D"}
FREQ_NAMES = {"M": "month", "W": "week", "D": "day"}
ANOMALY_BASES = ("level", "change", "seasonal")
CALENDARS = ("weekday", "weekday_us_holidays")


@dataclass(frozen=True)
class SeriesSpec:
    """Declares one series (or one per group). ``id`` is a template; ``{group}`` is replaced by the
    group value (e.g. ``"share:genre:{group}"``)."""

    id: str
    metric: str  # count | share | mean | nunique | level
    unit: str
    name: str | None = None  # metric name in the output (default: ``metric``), e.g. "volume"
    group_by: str | None = None
    groups: tuple[str, ...] | None = None  # explicit group list (default: observed values, sorted)
    entity: str = "all"  # entity of an ungrouped series
    entity_type: str = "platform"
    adverse_direction: str = "none"  # up | down | both | none
    is_count: bool | None = None  # modelled on log1p scale (default: count/nunique metrics)
    value_column: str = "value"  # mean/level: the value; nunique: the column counted
    min_count: int = 1  # mean: NaN below this many observations in the period
    event_type: str | None = None  # only observations of this event type
    source: str | None = None
    scan_anomalies: bool = True
    trend: bool = True
    forecast: bool = False  # published in predictions.forecasts
    share_forecast: bool = False  # published in predictions.share_forecasts (eligibility applies)
    volume_guard: bool = True  # min-volume guard on anomalies (needs meaningful counts)
    min_volume: int | None = None  # guard threshold (None = CoreConfig.anomaly_min_volume)
    count_noun: str = "observations"  # what ``counts`` counts, in guard messages (e.g. "ratings")
    trend_unit: str | None = None  # unit of a trend signal (default "<metric>/<period>")
    # anomalies scored on the value itself ("level") or on its period-over-period change ("change":
    # for smooth level series, where a slow drift would otherwise be flagged month after month)
    anomaly_basis: str = "level"
    # published precision of the values (e.g. 0.1 for a rate given to one decimal): an anomaly whose
    # deviation is not larger than this is suppressed ("within data resolution")
    resolution: float | None = None
    # ---- opt-in seasonality and derived values (defaults: none, the v1.2 behaviour) ----------
    seasonal_period: int | None = None  # periods per seasonal cycle (7 = weekly cycle of days)
    calendar: str | None = None  # season classes of days: weekday | weekday_us_holidays
    forecast_models: tuple[str, ...] | None = None  # candidate models (None = the v1.2 trio)
    rolling: int | None = None  # trailing mean over this many periods
    ratio_lag: int | None = None  # divide by the (rolled) value this many periods earlier

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"series spec {self.id}: metric must be one of {METRICS}")
        if self.adverse_direction not in ("up", "down", "both", "none"):
            raise ValueError(f"series spec {self.id}: adverse_direction must be up|down|both|none")
        if self.anomaly_basis not in ANOMALY_BASES:
            raise ValueError(f"series spec {self.id}: anomaly_basis must be one of {ANOMALY_BASES}")
        if self.group_by is not None and "{group}" not in self.id:
            raise ValueError(f"series spec {self.id}: a grouped spec needs {{group}} in its id")
        if self.calendar is not None and self.calendar not in CALENDARS:
            raise ValueError(f"series spec {self.id}: calendar must be one of {CALENDARS}")
        for name in ("seasonal_period", "rolling", "ratio_lag"):
            v = getattr(self, name)
            if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
                raise ValueError(f"series spec {self.id}: {name} must be a positive integer")
        if self.anomaly_basis == "seasonal" and self.seasonal_period is None and self.calendar is None:
            raise ValueError(f"series spec {self.id}: anomaly_basis seasonal needs seasonal_period or calendar")
        if self.forecast_models is not None:
            from jev_ml.core.forecast import ALL_MODELS, SEASONAL_MODELS

            bad = [m for m in self.forecast_models if m not in ALL_MODELS]
            if bad or not self.forecast_models:
                raise ValueError(f"series spec {self.id}: unknown forecast models {bad} (known: {ALL_MODELS})")
            if "naive" not in self.forecast_models:
                raise ValueError(f"series spec {self.id}: forecast_models must include the naive benchmark")
            if any(m in SEASONAL_MODELS for m in self.forecast_models) and self.seasonal_period is None:
                raise ValueError(f"series spec {self.id}: seasonal forecast models need seasonal_period")

    @property
    def output_metric(self) -> str:
        return self.name or self.metric

    @property
    def count_like(self) -> bool:
        return self.is_count if self.is_count is not None else self.metric in ("count", "nunique")


@dataclass
class Series:
    id: str
    metric: str  # output metric name, e.g. volume | share | rating | active_users | level
    entity: str
    entity_type: str
    unit: str
    months: pd.PeriodIndex  # the period grid (monthly or weekly despite the name)
    values: np.ndarray  # float; NaN = undefined
    partial_last: bool
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0))  # observations behind each point
    count_series: bool | None = None  # None: inferred from the metric name
    adverse_direction: str = "none"
    source: str | None = None
    domain: str | None = None
    entity_id: str | None = None
    scan: bool = True
    trend: bool = True
    forecast: bool = False
    share_forecast: bool = False
    volume_guard: bool = True
    min_volume: int | None = None
    count_noun: str = "observations"
    trend_unit: str | None = None
    anomaly_basis: str = "level"
    resolution: float | None = None
    seasonal_period: int | None = None
    calendar: str | None = None
    forecast_models: tuple[str, ...] | None = None

    @property
    def is_count(self) -> bool:
        return self.count_series if self.count_series is not None else self.metric in COUNT_METRICS

    @property
    def freq(self) -> str:
        fs = self.months.freqstr
        return "W" if fs.startswith("W") else "D" if fs.startswith("D") else "M"

    @property
    def seasonal(self) -> bool:
        """Whether the series opted into the seasonal forecasting path (``forecast_models`` or
        ``seasonal_period`` declared)."""
        return self.forecast_models is not None or self.seasonal_period is not None

    def classes(self, periods: pd.PeriodIndex | None = None) -> np.ndarray:
        """Season class of each period (``season_classes``) of the grid, or of ``periods``."""
        return season_classes(self.months if periods is None else periods, self.seasonal_period, self.calendar)

    @property
    def periods(self) -> pd.PeriodIndex:
        return self.months

    def complete(self) -> tuple[pd.PeriodIndex, np.ndarray, np.ndarray]:
        """Complete periods only (drops the partial last period)."""
        if self.partial_last:
            return self.months[:-1], self.values[:-1], self.counts[:-1]
        return self.months, self.values, self.counts

    def to_dict(self, max_points: int) -> dict[str, Any]:
        n = len(self.months)
        start = max(0, n - max_points)
        pts = []
        for i in range(start, n):
            v = self.values[i]
            val: float | int | None
            if not np.isfinite(v):
                val = None
            elif self.is_count:
                val = round(float(v))
            else:
                val = fnum(v, 6)
            pts.append(
                {"t": period_str(self.months[i]), "v": val, "partial": bool(self.partial_last and i == n - 1)}
            )
        out: dict[str, Any] = {
            "id": self.id,
            "metric": self.metric,
            "entity": self.entity,
            "entity_type": self.entity_type,
            "unit": self.unit,
            "points": pts,
        }
        out["entity_id"] = self.entity_id or f"{self.entity_type}:{self.entity}"
        out["adverse_direction"] = self.adverse_direction
        out["frequency"] = FREQ_NAMES[self.freq]
        if self.domain is not None:
            out["domain"] = self.domain
        return out


# --------------------------------------------------------------------------------------------------
# season classes


def us_transit_holidays(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """The six US holidays on which transit operators commonly run a Sunday schedule: New Year's
    Day, Memorial Day, Independence Day, Labor Day, Thanksgiving and Christmas. Fixed-date holidays
    that fall on a weekend are observed on the nearest weekday (Saturday -> Friday, Sunday ->
    Monday), as for US federal holidays."""
    from pandas.tseries.holiday import (
        AbstractHolidayCalendar,
        Holiday,
        USLaborDay,
        USMemorialDay,
        USThanksgivingDay,
        nearest_workday,
    )

    class _Cal(AbstractHolidayCalendar):
        rules = [  # noqa: RUF012 (pandas API: a class attribute list)
            Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
            USMemorialDay,
            Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
            USLaborDay,
            USThanksgivingDay,
            Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
        ]

    out: pd.DatetimeIndex = _Cal().holidays(start - pd.Timedelta(days=7), end + pd.Timedelta(days=7))
    return out


def season_classes(periods: pd.PeriodIndex, seasonal_period: int | None, calendar: str | None) -> np.ndarray:
    """Season class per period: days -> day of week 0..6 (Monday = 0; with ``weekday_us_holidays``
    a holiday gets the Sunday class 6), weeks -> week of year 0..51 (ISO week 53 joins week 52),
    months -> month 0..11. Any other ``seasonal_period`` p -> period ordinal mod p."""
    if not len(periods):
        return np.zeros(0, dtype=np.int64)
    fs = periods.freqstr
    start = periods.start_time
    if fs.startswith("D") and (seasonal_period in (None, 7) or calendar is not None):
        cls = np.asarray(start.dayofweek, dtype=np.int64)
        if calendar == "weekday_us_holidays":
            hol = us_transit_holidays(start.min(), start.max())
            cls = np.where(start.normalize().isin(hol), 6, cls)
        return cls
    if fs.startswith("W") and seasonal_period in (None, 52):
        wk = np.asarray(periods.end_time.isocalendar().week, dtype=np.int64)
        return np.minimum(wk, 52) - 1
    if fs.startswith("M") and seasonal_period in (None, 12):
        return np.asarray(start.month, dtype=np.int64) - 1
    p = int(seasonal_period or 1)
    return (np.asarray(periods.asi8, dtype=np.int64) % p).astype(np.int64)  # type: ignore[attr-defined]


def seasonal_baseline_rows(classes: np.ndarray, i: int, n: int) -> np.ndarray:
    """Indices of the (at most) ``n`` periods before ``i`` with the same season class as ``i``, in
    time order: the anomaly baseline of ``anomaly_basis: seasonal``."""
    same = np.flatnonzero(classes[:i] == classes[i])
    return same[-n:]


# --------------------------------------------------------------------------------------------------
# builder


def period_grid(
    ts: np.ndarray, as_of: datetime, freq: str = "M", grid_end: str = "as_of"
) -> tuple[pd.PeriodIndex, bool]:
    """(period grid from the first observation to the grid end, whether the last period is partial)."""
    f = FREQS[freq]
    first = pd.Period(pd.Timestamp(float(np.min(ts)), unit="s"), freq=f)
    if grid_end == "last_observation":
        last = pd.Period(pd.Timestamp(float(np.max(ts)), unit="s"), freq=f)
        return pd.period_range(first, last, freq=f), False
    if grid_end != "as_of":
        raise ValueError("grid_end must be 'as_of' or 'last_observation'")
    a = pd.Timestamp(as_of)
    a = a.tz_convert(None) if a.tzinfo is not None else a
    cur = pd.Period(a, freq=f)
    partial = a > cur.start_time
    last = cur if partial else cur - 1
    return pd.period_range(first, last, freq=f), bool(partial)


def _group_rows(col: pd.Series, groups: Sequence[str] | None) -> dict[str, np.ndarray]:
    """Row positions per group value; list/tuple cells belong to every listed group."""
    values = col.to_numpy()
    s = pd.Series(values, index=np.arange(len(values)))
    if len(values) and any(isinstance(v, list | tuple) for v in values[: min(len(values), 1000)]):
        s = s.explode()
    s = s.dropna()
    rows = s.index.to_numpy(dtype=np.int64)
    codes, uniq = pd.factorize(s.to_numpy())
    order = np.argsort(codes, kind="stable")  # keeps row order within a group
    bounds = np.cumsum(np.bincount(codes, minlength=len(uniq)))
    out = {}
    start = 0
    for i, g in enumerate(uniq):
        out[str(g)] = rows[order[start : bounds[i]]]
        start = bounds[i]
    names = list(groups) if groups is not None else sorted(out)
    return {g: out.get(g, np.zeros(0, dtype=np.int64)) for g in names}


def _values(
    spec: SeriesSpec, rows: np.ndarray | None, code: np.ndarray, obs: pd.DataFrame, total: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_m = len(total)
    c = code if rows is None else code[rows]
    cnt = np.bincount(c, minlength=n_m).astype(float)
    if spec.metric == "count":
        return cnt, cnt
    if spec.metric == "share":
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total > 0, cnt / np.where(total > 0, total, 1), np.nan)
        return share, cnt
    if spec.metric == "sum":
        val = obs[spec.value_column].to_numpy(dtype=float)
        v = val if rows is None else val[rows]
        ok = np.isfinite(v)
        n_ok = np.bincount(c[ok], minlength=n_m).astype(float)
        tot = np.bincount(c[ok], weights=v[ok], minlength=n_m)
        return np.where(n_ok >= max(1, spec.min_count), tot, np.nan), cnt
    if spec.metric == "nunique":
        col = obs[spec.value_column].to_numpy()
        v = col if rows is None else col[rows]
        um = pd.DataFrame({"m": c, "u": v}).drop_duplicates()
        return np.bincount(um["m"].to_numpy(), minlength=n_m).astype(float), cnt
    # mean / level
    val = obs[spec.value_column].to_numpy(dtype=float)
    v = val if rows is None else val[rows]
    ok = np.isfinite(v)
    n_ok = np.bincount(c[ok], minlength=n_m).astype(float)
    s = np.bincount(c[ok], weights=v[ok], minlength=n_m)
    min_count = 1 if spec.metric == "level" else max(1, spec.min_count)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n_ok >= min_count, s / np.where(n_ok > 0, n_ok, 1), np.nan)
    return mean, cnt


def derive(values: np.ndarray, rolling: int | None, ratio_lag: int | None) -> np.ndarray:
    """``rolling`` trailing mean (NaN unless every period of the window is defined), then
    ``ratio_lag`` ratio to the value that many periods earlier (NaN when either is undefined or 0)."""
    v = np.asarray(values, dtype=float)
    if rolling is not None and rolling > 1:
        r = pd.Series(v).rolling(rolling, min_periods=rolling).mean().to_numpy()
        v = np.asarray(r, dtype=float)
    if ratio_lag is not None:
        prev = np.full(len(v), np.nan)
        if ratio_lag < len(v):
            prev[ratio_lag:] = v[:-ratio_lag]
        with np.errstate(invalid="ignore", divide="ignore"):
            v = np.where(np.isfinite(prev) & (prev != 0), v / np.where(prev != 0, prev, 1.0), np.nan)
    return v


def _blocks(specs: Sequence[SeriesSpec]) -> list[list[SeriesSpec]]:
    out: list[list[SeriesSpec]] = []
    for sp in specs:
        if (
            out
            and sp.group_by is not None
            and out[-1][0].group_by == sp.group_by
            and out[-1][0].groups == sp.groups
            and out[-1][0].event_type == sp.event_type
        ):
            out[-1].append(sp)
        else:
            out.append([sp])
    return out


def build_series(
    obs: pd.DataFrame,
    specs: Sequence[SeriesSpec],
    as_of: datetime,
    freq: str = "M",
    grid_end: str = "as_of",
    domain: str | None = None,
) -> list[Series]:
    """Build every spec's series from observations at or before ``as_of`` (callers filter)."""
    if not len(obs):
        return []
    ts = obs["timestamp"].to_numpy(dtype=float)
    months, partial = period_grid(ts, as_of, freq, grid_end)
    f = FREQS[freq]
    per = pd.PeriodIndex(pd.to_datetime(ts, unit="s"), freq=f)
    code = (np.asarray(per.asi8) - months[0].ordinal).astype(np.int64)  # type: ignore[attr-defined]
    n_m = len(months)
    inside = (code >= 0) & (code < n_m)
    if not inside.all():
        obs = obs.loc[inside].reset_index(drop=True)
        code = code[inside]
    else:
        obs = obs.reset_index(drop=True)
    event = obs["event_type"].to_numpy() if "event_type" in obs.columns else None
    out: list[Series] = []

    def make(sp: SeriesSpec, group: str | None, values: np.ndarray, counts: np.ndarray) -> Series:
        if sp.rolling is not None or sp.ratio_lag is not None:
            values = derive(values, sp.rolling, sp.ratio_lag)
        entity = group if group is not None else sp.entity
        sid = sp.id.format(group=group) if group is not None else sp.id
        return Series(
            sid,
            sp.output_metric,
            entity,
            sp.entity_type,
            sp.unit,
            months,
            values,
            partial,
            counts,
            count_series=sp.count_like,
            adverse_direction=sp.adverse_direction,
            source=sp.source,
            domain=domain,
            entity_id=f"{sp.entity_type}:{entity}",
            scan=sp.scan_anomalies,
            trend=sp.trend,
            forecast=sp.forecast,
            share_forecast=sp.share_forecast,
            volume_guard=sp.volume_guard,
            min_volume=sp.min_volume,
            count_noun=sp.count_noun,
            trend_unit=sp.trend_unit,
            anomaly_basis=sp.anomaly_basis,
            resolution=sp.resolution,
            seasonal_period=sp.seasonal_period,
            calendar=sp.calendar,
            forecast_models=tuple(sp.forecast_models) if sp.forecast_models is not None else None,
        )

    for block in _blocks(specs):
        head = block[0]
        if head.event_type is not None and event is not None:
            sel = np.flatnonzero(event == head.event_type)
            b_obs, b_code = obs.iloc[sel].reset_index(drop=True), code[sel]
        else:
            b_obs, b_code = obs, code
        total = np.bincount(b_code, minlength=n_m).astype(float)
        if head.group_by is None:
            for sp in block:
                v, c = _values(sp, None, b_code, b_obs, total)
                out.append(make(sp, None, v, c))
            continue
        rows_of = _group_rows(b_obs[head.group_by], head.groups)
        for g, rows in rows_of.items():
            for sp in block:
                v, c = _values(sp, rows, b_code, b_obs, total)
                out.append(make(sp, g, v, c))
    return out


def last_complete_period(series: list[Series]) -> str | None:
    """Start date of the last complete period of the grid (all series share one grid)."""
    if not series:
        return None
    m, _, _ = series[0].complete()
    return period_str(m[-1]) if len(m) else None
