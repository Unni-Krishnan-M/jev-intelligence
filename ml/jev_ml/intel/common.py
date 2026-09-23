"""Small shared helpers: stable ids, ISO timestamps, JSON sanitising and Evidence records."""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd


def stable_id(prefix: str, *parts: object) -> str:
    """Deterministic id: prefix + 12 hex chars of sha1(parts). Same inputs -> same id."""
    raw = "|".join(str(p) for p in parts)
    return f"{prefix}-{hashlib.sha1(raw.encode()).hexdigest()[:12]}"  # noqa: S324 (not security)


def short_hash(obj: Any, n: int = 8) -> str:
    raw = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:n]  # noqa: S324


def sub_seed(seed: int, key: str) -> int:
    """Per-object seed that does not depend on processing order."""
    return (seed * 1_000_003 + zlib.crc32(key.encode())) % (2**32)


def to_utc(dt: datetime | pd.Timestamp | str | None) -> datetime | None:
    if dt is None:
        return None
    ts = pd.Timestamp(dt)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    out: datetime = ts.to_pydatetime()
    return out


def iso(dt: datetime | pd.Timestamp | None) -> str | None:
    if dt is None:
        return None
    d = to_utc(dt)
    assert d is not None
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_from_epoch(sec: float | None) -> str | None:
    if sec is None or (isinstance(sec, float) and math.isnan(sec)):
        return None
    return iso(datetime.fromtimestamp(float(sec), tz=UTC))


def month_str(p: pd.Period | pd.Timestamp) -> str:
    return pd.Period(p, freq="M").start_time.strftime("%Y-%m-01")


def fnum(x: Any, nd: int | None = 6) -> float | None:
    """Float for JSON: None for NaN/inf/None; optionally rounded."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd) if nd is not None else v


def clean(obj: Any) -> Any:
    """Recursively convert numpy/pandas scalars to Python and NaN/inf to None (JSON-safe)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, bool | np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, datetime | pd.Timestamp):
        return iso(obj)
    return obj


def evidence(
    kind: str, label: str, value: Any = None, detail: str | None = None, ref: str | None = None
) -> dict[str, Any]:
    """Evidence = {kind: metric|series|record|test|model, label, value, detail, ref}."""
    if isinstance(value, float | np.floating):
        value = fnum(value)
    elif isinstance(value, np.integer):
        value = int(value)
    return {"kind": kind, "label": label, "value": value, "detail": detail, "ref": ref}


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))
