"""UNDERSTAND: monthly series built from ratings <= as_of.

Months are calendar months (UTC). A month is *complete* when the first instant of the next month is
<= as_of. The month containing as_of (if as_of is past its first instant) is included as the last
point with ``partial: true`` and is excluded from every baseline, trend, anomaly and forecast.
Months without ratings are zero for volume/active users, undefined (NaN) for share/rating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from jev_ml.intel.common import fnum, month_str
from jev_ml.intel.config import IntelConfig
from jev_ml.intel.ingest import Prepared

COUNT_METRICS = ("volume", "active_users")


@dataclass
class Series:
    id: str
    metric: str  # volume | share | rating | active_users
    entity: str
    entity_type: str  # platform | genre
    unit: str
    months: pd.PeriodIndex
    values: np.ndarray  # float; NaN = undefined
    partial_last: bool
    counts: np.ndarray = field(default_factory=lambda: np.zeros(0))  # ratings behind each point

    @property
    def is_count(self) -> bool:
        return self.metric in COUNT_METRICS

    def complete(self) -> tuple[pd.PeriodIndex, np.ndarray, np.ndarray]:
        """Complete months only (drops the partial last month)."""
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
                {"t": month_str(self.months[i]), "v": val, "partial": bool(self.partial_last and i == n - 1)}
            )
        return {
            "id": self.id,
            "metric": self.metric,
            "entity": self.entity,
            "entity_type": self.entity_type,
            "unit": self.unit,
            "points": pts,
        }


def month_range(prep: Prepared) -> tuple[pd.PeriodIndex, bool]:
    ts = prep.ratings["timestamp"]
    first = pd.Period(pd.Timestamp(float(ts.min()), unit="s"), freq="M")
    as_of = pd.Timestamp(prep.as_of).tz_convert(None)
    cur = pd.Period(as_of, freq="M")
    partial = as_of > cur.start_time
    last = cur if partial else cur - 1
    return pd.period_range(first, last, freq="M"), bool(partial)


def build_series(prep: Prepared, cfg: IntelConfig) -> list[Series]:
    months, partial = month_range(prep)
    r = prep.ratings
    per = pd.PeriodIndex(pd.to_datetime(r["timestamp"].to_numpy(), unit="s"), freq="M")
    code = ((per.year - months[0].year) * 12 + (per.month - months[0].month)).to_numpy()
    n_m = len(months)
    vol = np.bincount(code, minlength=n_m).astype(float)
    # distinct raters per month
    um = pd.DataFrame({"m": code, "u": r["user_id"].to_numpy()}).drop_duplicates()
    users = np.bincount(um["m"].to_numpy(), minlength=n_m).astype(float)
    out = [
        Series("volume:all", "volume", "all", "platform", "ratings/month", months, vol, partial, vol.copy()),
        Series(
            "active_users:all",
            "active_users",
            "all",
            "platform",
            "users/month",
            months,
            users,
            partial,
            vol.copy(),
        ),
    ]
    # genre explode: one row per (rating, genre)
    mids = r["movie_id"].to_numpy()
    ratings = r["rating"].to_numpy()
    films_of: dict[str, list[int]] = {g: [] for g in prep.genres}
    for m in np.unique(mids):
        for g in prep.genre_of.get(int(m), ()):
            films_of[g].append(int(m))
    for g in prep.genres:
        mask = np.isin(mids, np.asarray(films_of[g], dtype=np.int64))
        gv = np.bincount(code[mask], minlength=n_m).astype(float)
        gs = np.bincount(code[mask], weights=ratings[mask], minlength=n_m)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(vol > 0, gv / np.where(vol > 0, vol, 1), np.nan)
            mean_r = np.where(gv >= cfg.rating_min_count, gs / np.where(gv > 0, gv, 1), np.nan)
        out.append(
            Series(f"volume:genre:{g}", "volume", g, "genre", "ratings/month", months, gv, partial, gv)
        )
        out.append(
            Series(f"share:genre:{g}", "share", g, "genre", "share of ratings", months, share, partial, gv)
        )
        out.append(
            Series(f"rating:genre:{g}", "rating", g, "genre", "mean rating", months, mean_r, partial, gv)
        )
    return out
