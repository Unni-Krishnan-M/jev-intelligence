"""VALIDATE: quality checks, the generic observation validator and source provenance rows.

``Checks`` accumulates checks ``{name, passed, value, threshold, severity, detail, source}``;
score = weighted share passed (weights per severity from ``CoreConfig.check_weights``).

``validate_observations`` is the core validation of a generic observation frame (used for every
adapter that does not validate its own raw data, see ``DomainData.core_checks``):

* schema: the required observation columns exist (error);
* leakage guard first: rows with ``timestamp > as_of`` are dropped *before* any other check and
  their number is reported (info);
* unparseable/null timestamps or entity ids (error), non-finite values where a value is required
  (error), timestamps before ``earliest_ts`` or after the wall clock (error);
* duplicate (timestamp, entity_id, event_type) rows, kept once (warning).
Invalid rows are dropped; the cleaned frame is sorted by (timestamp, entity_id).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import evidence, iso_from_epoch

OBSERVATION_REQUIRED = ("timestamp", "entity_id", "value")


class Checks:
    def __init__(self, weights: dict[str, float]) -> None:
        self.items: list[dict[str, Any]] = []
        self.weights = weights

    def add(
        self,
        name: str,
        passed: bool,
        value: float,
        threshold: float,
        severity: str,
        detail: str,
        source: str,
    ) -> None:
        self.items.append(
            {
                "name": name,
                "passed": bool(passed),
                "value": float(value),
                "threshold": float(threshold),
                "severity": severity,
                "detail": detail,
                "source": source,
            }
        )

    def extend(self, items: list[dict[str, Any]]) -> None:
        self.items.extend(items)

    def score(self) -> float:
        tot = sum(self.weights[c["severity"]] for c in self.items)
        ok = sum(self.weights[c["severity"]] for c in self.items if c["passed"])
        return float(ok / tot) if tot else 1.0


def to_epoch(col: pd.Series) -> pd.Series:
    """Seconds since epoch from numeric seconds or datetime-like values (NaN when unparseable)."""
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(np.float64)
    dt = pd.to_datetime(col, utc=True, errors="coerce")
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def validate_observations(
    obs: pd.DataFrame,
    source: str,
    as_of_ts: float,
    now_ts: float,
    checks: Checks,
    earliest_ts: float | None = None,
    value_required: bool = True,
) -> tuple[pd.DataFrame, int]:
    """Returns (clean rows <= as_of, number of rows dropped for being after as_of)."""
    missing = [c for c in OBSERVATION_REQUIRED if c not in obs.columns]
    checks.add(
        f"{source}_schema",
        not missing,
        len(missing),
        0,
        "error",
        f"missing columns: {missing}" if missing else "all required columns present",
        source,
    )
    if missing:
        return obs.iloc[0:0].copy(), 0
    d = obs.copy()
    d["timestamp"] = to_epoch(d["timestamp"])
    after = d["timestamp"] > as_of_ts
    n_after = int(after.sum())
    d = d[~after]
    n0 = len(d)
    bad_key = d["timestamp"].isna() | d["entity_id"].isna()
    checks.add(
        f"{source}_null_keys",
        not bad_key.any(),
        (int(bad_key.sum()) / n0) if n0 else 0.0,
        0,
        "error",
        f"{int(bad_key.sum())} of {n0} rows have an unparseable timestamp or no entity",
        source,
    )
    d = d[~bad_key]
    d["value"] = pd.to_numeric(d["value"], errors="coerce").astype(np.float64)
    if value_required:
        bad_val = ~np.isfinite(d["value"].to_numpy())
        checks.add(
            f"{source}_values",
            not bad_val.any(),
            int(bad_val.sum()),
            0,
            "error",
            f"{int(bad_val.sum())} rows without a finite value",
            source,
        )
        d = d[~bad_val]
    ts = d["timestamp"]
    lo = earliest_ts if earliest_ts is not None else -np.inf
    bad_ts = (ts < lo) | (ts > now_ts)
    checks.add(
        f"{source}_timestamp_range",
        not bad_ts.any(),
        int(bad_ts.sum()),
        0,
        "error",
        f"{int(bad_ts.sum())} timestamps outside the valid range (before "
        f"{iso_from_epoch(earliest_ts) if earliest_ts is not None else '-inf'} or after the wall clock)",
        source,
    )
    d = d[~bad_ts]
    keys = ["timestamp", "entity_id"] + (["event_type"] if "event_type" in d.columns else [])
    dup = d.duplicated(keys)
    checks.add(
        f"{source}_duplicates",
        not dup.any(),
        int(dup.sum()),
        0,
        "warning",
        f"{int(dup.sum())} duplicate ({', '.join(keys)}) rows (kept once)",
        source,
    )
    d = d[~dup]
    checks.add(
        f"{source}_after_as_of",
        True,
        n_after,
        0,
        "info",
        f"{n_after} rows after as_of excluded from this run (leakage guard)",
        source,
    )
    d = d.sort_values(["timestamp", "entity_id"], kind="mergesort").reset_index(drop=True)
    return d, n_after


def source_row(
    source: str,
    kind: str,
    rows: int,
    first: float | None,
    last: float | None,
    now_ts: float,
    as_of_ts: float | None,
    expected: str | None,
    fresh: bool | None,
    detail: str | None,
) -> dict[str, Any]:
    """A DataSource row (contract section 4, ``data.sources[]``)."""
    return {
        "source": source,
        "kind": kind,
        "rows": int(rows),
        "first_event": iso_from_epoch(first),
        "last_event": iso_from_epoch(last),
        "age_days": round((now_ts - last) / 86400.0, 2) if last is not None else None,
        "lag_days": round((as_of_ts - last) / 86400.0, 2)
        if (last is not None and as_of_ts is not None)
        else None,
        "expected_update": expected,
        "fresh": fresh,
        "detail": detail,
    }


def quality_evidence(quality: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        evidence("test", c["name"], c["value"], c["detail"])
        for c in quality["checks"]
        if not c["passed"] and c["severity"] != "info"
    ]
