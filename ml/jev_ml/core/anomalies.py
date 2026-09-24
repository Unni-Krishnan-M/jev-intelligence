"""DETECT (series anomalies): modified z-score of each point against its trailing baseline.

z = 0.6745 * (x - median) / MAD over the ``anomaly_baseline_months`` complete periods *before* the
point (Iglewicz & Hoaglin 1993; the point never contaminates its own baseline). Count-like series
are scored on log1p scale (multiplicative bursts), others on their natural scale. If MAD = 0 the
mean-absolute-deviation variant (x - median) / (1.2533 * MeanAD) is used. Only the last
``anomaly_scan_months`` complete periods are scanned (new events, not history), and only series
whose spec sets ``scan_anomalies``. With ``anomaly_basis = "change"`` the scored quantity is the
period-over-period change (x_t - x_{t-1}) against the trailing changes, on the raw scale; ``value`` is
then the change, ``baseline`` the median change, and ``level`` the series value at the point.

Resolution guard (series with ``resolution``): a deviation not larger than the published precision
of the values is kept but suppressed. Min-volume guard (series with ``volume_guard``): an anomaly
on fewer than ``min_volume`` observations
(point and baseline median) is kept but marked suppressed. Keys an operator dismissed are
suppressed too unless the severity escalated (``suppression``).

Additive platform fields: ``observed_value`` (= value), ``expected_value`` (= baseline median),
``anomaly_score`` = min(1, |z| / (2 * threshold)) (a normalised evidence strength, kind ``margin``,
not a probability; ``confidence`` carries the same number), ``baseline_points``, ``entity_id``,
``source``, ``adverse`` (whether the deviation points in the series' adverse direction) and
``severity_basis`` (the measure and bands the severity came from; the early-warning decision
reads it).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jev_ml.core.common import clip01, evidence, fnum, period_str, stable_id
from jev_ml.core.config import SEVERITIES, CoreConfig, severity_rank
from jev_ml.core.series import Series, seasonal_baseline_rows


def suppression(key: str, severity: str, suppressed: dict[str, str]) -> str | None:
    """Reason a key is suppressed by a prior operator dismissal, or None.

    The value is the severity at dismissal (a later *higher* severity is an escalation and fires
    again) or a free-text reason (suppressed at every severity)."""
    if key not in suppressed:
        return None
    val = suppressed[key]
    if val in SEVERITIES:
        if severity_rank(severity) > severity_rank(val):
            return None  # escalated since the dismissal
        return f"dismissed by operator at severity {val}"
    return f"dismissed by operator: {val}"


def band(value: float, bands: dict[str, float]) -> str:
    sev = "low"
    for name in ("medium", "high", "critical"):
        if value >= bands[name]:
            sev = name
    return sev


_band = band  # historical name


def robust_z(x: float, baseline: np.ndarray) -> float | None:
    med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - med)))
    if mad > 1e-12:
        return 0.6745 * (x - med) / mad
    mean_ad = float(np.mean(np.abs(baseline - med)))
    if mean_ad > 1e-12:
        return (x - med) / (1.2533 * mean_ad)
    return None


def is_adverse(direction: str, adverse_direction: str) -> bool | None:
    """Whether a move up/down is adverse for a series (None when the series declares no direction)."""
    if adverse_direction == "none":
        return None
    if adverse_direction == "both":
        return True
    return direction == adverse_direction


def scan_series(
    s: Series, cfg: CoreConfig, as_of_key: str, suppressed: dict[str, str], scan_months: int | None = None
) -> list[dict[str, Any]]:
    months, levels, counts = s.complete()
    change = s.anomaly_basis == "change"
    values = np.r_[np.nan, np.diff(levels)] if change else levels
    n = len(values)
    scan = scan_months or cfg.anomaly_scan_months
    out: list[dict[str, Any]] = []
    tf = np.log1p if (s.is_count and not change) else (lambda a: a)
    min_volume = s.min_volume if s.min_volume is not None else cfg.anomaly_min_volume
    classes = s.anomaly_classes(months) if s.anomaly_basis == "seasonal" else np.zeros(0, dtype=np.int64)
    for i in range(max(0, n - scan), n):
        x = values[i]
        if not np.isfinite(x):
            continue
        if s.anomaly_basis == "seasonal":  # WS4b hook: baseline = earlier periods of the same season class
            rows = seasonal_baseline_rows(classes, i, cfg.anomaly_baseline_months)
            base_raw = values[rows]
            base_cnt = counts[rows] if len(counts) == n else counts[:0]
        else:
            lo = max(0, i - cfg.anomaly_baseline_months)
            base_raw = values[lo:i]
            base_cnt = counts[lo:i]
        ok = np.isfinite(base_raw)
        if ok.sum() < cfg.anomaly_min_baseline_months:
            continue
        base = np.asarray(tf(base_raw[ok]), dtype=float)
        z = robust_z(float(tf(np.asarray([x]))[0]), base)
        if z is None or abs(z) < cfg.anomaly_z_threshold:
            continue
        kind = "series_spike" if z > 0 else "series_drop"
        base_med = float(np.median(base_raw[ok]))
        severity = band(abs(z), cfg.anomaly_severity_bands)
        key = f"anomaly:{kind}:{s.id}"
        reason = None
        vol_now = float(counts[i]) if len(counts) > i else float("nan")
        vol_base = float(np.median(base_cnt[ok])) if len(base_cnt) == len(base_raw) else float("nan")
        if s.volume_guard and max(vol_now, vol_base) < min_volume:
            reason = (
                f"min-volume guard: {vol_now:.0f} {s.count_noun} (baseline median {vol_base:.0f}) "
                f"< {min_volume}"
            )
        if reason is None and s.resolution is not None and abs(x - base_med) <= s.resolution + 1e-9:
            reason = f"within data resolution: |deviation| {abs(x - base_med):.4g} <= {s.resolution:g}"
        reason = reason or suppression(key, severity, suppressed)
        t = period_str(months[i])
        # from the published (rounded) z, exactly as the v1.1 signal/warning strength was computed
        score = clip01(abs(float(fnum(z, 3) or 0.0)) / (2 * cfg.anomaly_z_threshold))
        label = t[:7] if s.freq == "M" else t
        prev = levels[i - 1] if i > 0 else np.nan
        rel_dev = (
            (x / abs(prev) if np.isfinite(prev) and prev != 0 else None)
            if change
            else ((x - base_med) / abs(base_med) if base_med != 0 else None)
        )
        what = f"{s.id} change" if change else s.id
        out.append(
            {
                "id": stable_id("anom", kind, s.id, t, as_of_key),
                "dedup_key": key,
                "kind": kind,
                "entity_type": s.entity_type,
                "entity": s.entity,
                "series_id": s.id,
                "metric": s.metric,
                "detected_at": t,
                "value": fnum(x),
                "baseline": fnum(base_med),
                "deviation": fnum(x - base_med),
                "score": fnum(z, 3),
                "method": "robust_z",
                "severity": severity,
                "suppressed": reason is not None,
                "suppression_reason": reason,
                "features": {},
                "evidence": [
                    evidence("series", f"{what} in {label}", fnum(x), f"{s.unit}", s.id),
                    evidence(
                        "metric",
                        "baseline median change" if change else "baseline median",
                        fnum(base_med),
                        f"median of {int(ok.sum())} complete months before {label}"
                        if s.freq == "M"
                        else f"median of {int(ok.sum())} complete periods before {label}",
                        s.id,
                    ),
                    evidence(
                        "test",
                        "robust z",
                        fnum(z, 3),
                        f"|z| >= {cfg.anomaly_z_threshold} ({'log1p scale' if s.is_count else 'raw scale'})",
                    ),
                ],
                "observed_value": fnum(x),
                "expected_value": fnum(base_med),
                "anomaly_score": fnum(score, 4),
                "confidence": fnum(score, 4),
                "confidence_kind": "margin",
                "baseline_points": int(ok.sum()),
                "anomaly_basis": s.anomaly_basis,
                "level": fnum(levels[i]),
                "relative_deviation": fnum(rel_dev, 6),
                "entity_id": s.entity_id or f"{s.entity_type}:{s.entity}",
                "source": s.source,
                "adverse": is_adverse("up" if z > 0 else "down", s.adverse_direction),
                "severity_basis": {
                    "measure": "|robust z|",
                    "value": fnum(abs(z), None),  # unrounded: the band above used this exact value
                    "bands": {"low": cfg.anomaly_z_threshold, **cfg.anomaly_severity_bands},
                },
            }
        )
    return out


def scan_all_series(
    series: list[Series], cfg: CoreConfig, as_of_key: str, suppressed: dict[str, str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in series:
        if s.scan:
            out.extend(scan_series(s, cfg, as_of_key, suppressed))
    return out
