"""DETECT (anomalies): series spikes/drops, rater behaviour and live feedback rate.

(a) Series: modified z-score (Iglewicz & Hoaglin 1993) of each point against its trailing
    baseline: z = 0.6745 * (x - median) / MAD over the `anomaly_baseline_months` complete months
    *before* the point (the point never contaminates its own baseline). Count series are scored on
    log1p scale because monthly rating counts are multiplicative/bursty; share and rating series on
    their natural scale. If MAD = 0 the mean-absolute-deviation variant (x - median)/(1.2533*MeanAD)
    is used. Only the last `anomaly_scan_months` complete months are scanned (new events, not
    history). Per-genre *volume* series are not scanned: they move with platform volume, so a single
    burst would be reported ~15 times; genre-specific behaviour is caught on the share series.
    Min-volume guard: an anomaly on fewer than `anomaly_min_volume` ratings (month and baseline
    median) is kept but marked suppressed.
(b) Raters: IsolationForest (seeded) on per-user behavioural features computed from ratings
    <= as_of. The deviation from item means is leave-one-out (the user's own rating is removed from
    the item mean) so a user cannot pull the reference towards themself. The top `contamination`
    share is flagged; `score` is the anomaly-score percentile (0..100) and severity comes from the
    raw IsolationForest score s (Liu et al.: s ~ 0.5 ordinary, s -> 1 anomalous).
(c) Live feedback: negative-feedback share in the last 7 days vs the prior 28 days, one-sided
    two-proportion z test. Skipped (with the reason) when either window has too few events.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from jev_ml.intel.common import evidence, fnum, iso, iso_from_epoch, month_str, stable_id
from jev_ml.intel.config import SEVERITIES, IntelConfig, severity_rank
from jev_ml.intel.ingest import NEGATIVE_FEEDBACK, Prepared
from jev_ml.intel.series import Series

SCAN_PREFIXES = ("volume:all", "active_users:all", "share:genre:", "rating:genre:")
RATER_FEATURES = (
    "ratings_per_active_day",
    "max_ratings_one_day",
    "busiest_day_share",
    "extreme_share",
    "rating_std",
    "item_mean_abs_dev",
    "longtail_share",
)
_LOG_FEATURES = ("ratings_per_active_day", "max_ratings_one_day")


def suppression(key: str, severity: str, suppressed: dict[str, str]) -> str | None:
    """Reason a key is suppressed by a prior operator dismissal, or None (see PipelineInputs)."""
    if key not in suppressed:
        return None
    val = suppressed[key]
    if val in SEVERITIES:
        if severity_rank(severity) > severity_rank(val):
            return None  # escalated since the dismissal
        return f"dismissed by operator at severity {val}"
    return f"dismissed by operator: {val}"


def _band(value: float, bands: dict[str, float]) -> str:
    sev = "low"
    for name in ("medium", "high", "critical"):
        if value >= bands[name]:
            sev = name
    return sev


def robust_z(x: float, baseline: np.ndarray) -> float | None:
    med = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - med)))
    if mad > 1e-12:
        return 0.6745 * (x - med) / mad
    mean_ad = float(np.mean(np.abs(baseline - med)))
    if mean_ad > 1e-12:
        return (x - med) / (1.2533 * mean_ad)
    return None


def scan_series(
    s: Series, cfg: IntelConfig, as_of_key: str, suppressed: dict[str, str], scan_months: int | None = None
) -> list[dict[str, Any]]:
    months, values, counts = s.complete()
    n = len(values)
    scan = scan_months or cfg.anomaly_scan_months
    out: list[dict[str, Any]] = []
    tf = np.log1p if s.is_count else (lambda a: a)
    for i in range(max(0, n - scan), n):
        x = values[i]
        if not np.isfinite(x):
            continue
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
        severity = _band(abs(z), cfg.anomaly_severity_bands)
        key = f"anomaly:{kind}:{s.id}"
        reason = None
        vol_now = float(counts[i])
        vol_base = float(np.median(base_cnt[ok]))
        if max(vol_now, vol_base) < cfg.anomaly_min_volume:
            reason = (
                f"min-volume guard: {vol_now:.0f} ratings (baseline median {vol_base:.0f}) "
                f"< {cfg.anomaly_min_volume}"
            )
        reason = reason or suppression(key, severity, suppressed)
        t = month_str(months[i])
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
                    evidence("series", f"{s.id} in {t[:7]}", fnum(x), f"{s.unit}", s.id),
                    evidence(
                        "metric",
                        "baseline median",
                        fnum(base_med),
                        f"median of {int(ok.sum())} complete months before {t[:7]}",
                        s.id,
                    ),
                    evidence(
                        "test",
                        "robust z",
                        fnum(z, 3),
                        f"|z| >= {cfg.anomaly_z_threshold} ({'log1p scale' if s.is_count else 'raw scale'})",
                    ),
                ],
            }
        )
    return out


def scan_all_series(
    series: list[Series], cfg: IntelConfig, as_of_key: str, suppressed: dict[str, str]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in series:
        if s.id.startswith(SCAN_PREFIXES):
            out.extend(scan_series(s, cfg, as_of_key, suppressed))
    return out


# --------------------------------------------------------------------------------------------------
# raters


def rater_features(ratings: pd.DataFrame, cfg: IntelConfig) -> pd.DataFrame:
    """Per-user behavioural features from the given rows (callers pass rows <= as_of)."""
    r = ratings[["user_id", "movie_id", "rating", "timestamp"]]
    u = r["user_id"].to_numpy()
    m = r["movie_id"].to_numpy()
    x = r["rating"].to_numpy(dtype=float)
    day = (r["timestamp"].to_numpy(dtype=float) // 86400).astype(np.int64)
    item_sum = pd.Series(x).groupby(m).transform("sum").to_numpy()
    item_n = pd.Series(x).groupby(m).transform("size").to_numpy().astype(float)
    others = item_n - 1
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = np.where(others > 0, (item_sum - x) / np.where(others > 0, others, 1), np.nan)
    df = pd.DataFrame(
        {
            "u": u,
            "day": day,
            "x": x,
            "extreme": ((x <= 1.0) | (x >= 5.0)).astype(float),
            "absdev": np.abs(x - loo),
            "longtail": (others <= cfg.rater_longtail_max_count).astype(float),
        }
    )
    g = df.groupby("u")
    per_day = df.groupby(["u", "day"]).size()
    days = per_day.groupby(level=0).size()
    max_day = per_day.groupby(level=0).max()
    n = g.size()
    feats = pd.DataFrame(
        {
            "n_ratings": n,
            "ratings_per_active_day": n / days,
            "max_ratings_one_day": max_day,
            "busiest_day_share": max_day / n,
            "extreme_share": g["extreme"].mean(),
            "rating_std": g["x"].std(ddof=0),
            "item_mean_abs_dev": g["absdev"].mean(),  # NaN-skipping mean over items with others
            "longtail_share": g["longtail"].mean(),
        }
    )
    feats["item_mean_abs_dev"] = feats["item_mean_abs_dev"].fillna(feats["item_mean_abs_dev"].median())
    return feats.fillna(0.0)


def isolation_scores(feats: pd.DataFrame, cfg: IntelConfig, seed: int | None = None) -> pd.Series:
    """IsolationForest anomaly score s in (0, 1] per user (higher = more anomalous)."""
    X = feats[list(RATER_FEATURES)].copy()
    for c in _LOG_FEATURES:
        X[c] = np.log1p(X[c])
    model = IsolationForest(
        n_estimators=cfg.rater_estimators,
        contamination="auto",
        random_state=cfg.seed if seed is None else seed,
    )
    model.fit(X.to_numpy())
    return pd.Series(-model.score_samples(X.to_numpy()), index=feats.index, name="if_score")


def detect_raters(
    prep: Prepared, cfg: IntelConfig, as_of_key: str, suppressed: dict[str, str]
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Returns (anomalies for flagged raters, per-user frame with features + score + percentile)."""
    feats = rater_features(prep.ratings, cfg)
    feats = feats[feats["n_ratings"] >= cfg.rater_min_ratings]
    if len(feats) < 20:
        return [], feats
    s = isolation_scores(feats, cfg)
    feats = feats.assign(if_score=s)
    feats["percentile"] = 100.0 * s.rank(method="max") / len(s)
    last = prep.ratings.groupby("user_id")["timestamp"].max()
    feats["last_ts"] = last.reindex(feats.index)
    n_flag = max(1, int(np.ceil(cfg.rater_contamination * len(feats))))
    order = feats.iloc[np.lexsort((feats.index.to_numpy(), -feats["if_score"].to_numpy()))]
    flagged = order.head(n_flag)
    feats["flagged"] = feats.index.isin(flagged.index)
    median_s = float(s.median())
    pct = feats[list(RATER_FEATURES)].rank(pct=True)
    out = []
    for uid_raw, row in flagged.iterrows():
        uid = int(uid_raw)  # type: ignore[call-overload]
        severity = _band(float(row["if_score"]), cfg.rater_severity_bands)
        key = f"anomaly:rater_behaviour:user:{int(uid)}"
        idle_days = (prep.as_of_ts - float(row["last_ts"])) / 86400.0
        reason = None
        if idle_days > cfg.rater_active_days:
            last_day = (iso_from_epoch(row["last_ts"]) or "")[:10]
            reason = f"inactive: last rating {last_day} ({idle_days:.0f} days before as_of)"
        reason = reason or suppression(key, severity, suppressed)
        fp = pct.loc[uid]
        extreme_feats = sorted(RATER_FEATURES, key=lambda c: -abs(float(fp[c]) - 0.5))[:3]
        features: dict[str, Any] = {c: fnum(row[c], 4) for c in RATER_FEATURES}
        features.update(
            {
                "n_ratings": int(row["n_ratings"]),
                "if_score": fnum(row["if_score"], 4),
                "last_rating": iso_from_epoch(row["last_ts"]),
                "days_inactive": fnum(idle_days, 1),
            }
        )
        out.append(
            {
                "id": stable_id("anom", "rater_behaviour", int(uid), as_of_key),
                "dedup_key": key,
                "kind": "rater_behaviour",
                "entity_type": "user",
                "entity": str(int(uid)),
                "series_id": None,
                "metric": None,
                "detected_at": iso(prep.as_of),
                "value": fnum(row["if_score"], 4),
                "baseline": fnum(median_s, 4),
                "deviation": fnum(float(row["if_score"]) - median_s, 4),
                "score": fnum(row["percentile"], 2),
                "method": "isolation_forest",
                "severity": severity,
                "suppressed": reason is not None,
                "suppression_reason": reason,
                "features": features,
                "evidence": [
                    evidence(
                        "model",
                        "isolation-forest score",
                        fnum(row["if_score"], 4),
                        f"population median {median_s:.3f}; top {100 * cfg.rater_contamination:.0f} % "
                        "flagged",
                    )
                ]
                + [
                    evidence(
                        "metric",
                        c,
                        fnum(row[c], 4),
                        f"population percentile {100 * float(fp[c]):.0f}",
                    )
                    for c in extreme_feats
                ],
            }
        )
    return out, feats


# --------------------------------------------------------------------------------------------------
# live feedback


def live_feedback(
    prep: Prepared, cfg: IntelConfig, as_of_key: str, suppressed: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fb = prep.app_feedback
    now = prep.now_ts
    recent_lo = now - cfg.live_recent_days * 86400
    prior_lo = recent_lo - cfg.live_prior_days * 86400
    status: dict[str, Any] = {"status": "skipped", "reason": None}
    if fb is None or not len(fb):
        status["reason"] = "no app feedback recorded"
        return [], status
    ts = fb["timestamp"].to_numpy(dtype=float)
    neg = fb["feedback"].isin(NEGATIVE_FEEDBACK).to_numpy()
    rec = (ts > recent_lo) & (ts <= now)
    pri = (ts > prior_lo) & (ts <= recent_lo)
    n1, n0 = int(rec.sum()), int(pri.sum())
    k1, k0 = int(neg[rec].sum()), int(neg[pri].sum())
    status.update({"recent_n": n1, "recent_negative": k1, "prior_n": n0, "prior_negative": k0})
    if n1 < cfg.live_min_feedback or n0 < cfg.live_min_feedback:
        status["reason"] = (
            f"too little feedback to test: {n1} events in the last {cfg.live_recent_days} d, "
            f"{n0} in the prior {cfg.live_prior_days} d (need >= {cfg.live_min_feedback} each)"
        )
        return [], status
    p1, p0 = k1 / n1, k0 / n0
    pool = (k1 + k0) / (n1 + n0)
    se = np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se if se > 0 else 0.0
    p = float(stats.norm.sf(z))
    status.update({"status": "ok", "recent_rate": p1, "prior_rate": p0, "z": z, "p_value": p})
    if p >= cfg.live_alpha:
        return [], status
    lift = p1 / p0 if p0 > 0 else float("inf")
    severity = "low"
    for name, thr in (("medium", 1.2), ("high", 1.5), ("critical", 2.0)):
        if lift >= thr:
            severity = name
    key = "anomaly:live_feedback:app"
    reason = suppression(key, severity, suppressed)
    anom = {
        "id": stable_id("anom", "live_feedback", "app", (iso(prep.now) or "")[:10], as_of_key),
        "dedup_key": key,
        "kind": "live_feedback",
        "entity_type": "platform",
        "entity": "app",
        "series_id": None,
        "metric": "negative_feedback_rate",
        "detected_at": iso(prep.now),
        "value": fnum(p1),
        "baseline": fnum(p0),
        "deviation": fnum(p1 - p0),
        "score": fnum(z, 3),
        "method": "rate_test",
        "severity": severity,
        "suppressed": reason is not None,
        "suppression_reason": reason,
        "features": {"recent_n": n1, "recent_negative": k1, "prior_n": n0, "prior_negative": k0},
        "evidence": [
            evidence("metric", "negative rate, last 7 d", fnum(p1), f"{k1}/{n1}"),
            evidence("metric", "negative rate, prior 28 d", fnum(p0), f"{k0}/{n0}"),
            evidence("test", "two-proportion z (one-sided)", fnum(z, 3), f"p = {p:.4f}"),
        ],
    }
    return [anom], status
