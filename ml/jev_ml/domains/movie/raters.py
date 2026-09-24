"""DETECT (movie domain): rater behaviour and live recommendation-feedback anomalies.

(b) Raters: IsolationForest (seeded) on per-user behavioural features computed from ratings
    <= as_of. The deviation from item means is leave-one-out (the user's own rating is removed from
    the item mean) so a user cannot pull the reference towards themself. The top `contamination`
    share is flagged; `score` is the anomaly-score percentile (0..100) and severity comes from the
    raw IsolationForest score s (Liu et al.: s ~ 0.5 ordinary, s -> 1 anomalous).
(c) Live feedback: negative-feedback share in the last 7 days vs the prior 28 days, one-sided
    two-proportion z test. Skipped (with the reason) when either window has too few events.

Platform fields (additive): ``anomaly_score`` (raters: clip((s - 0.5) / 0.3); live: min(1, |z| / 4);
the same normalised strengths v1.1 used for signals and warning confidence, kind ``margin``),
``confidence``/``confidence_kind``, ``observed_value``/``expected_value`` (= value/baseline),
``entity_id``, ``source`` and ``severity_basis`` (IsolationForest score bands with the ordinary
s = 0.5 as the low edge; feedback lift bands 1.0 / 1.2 / 1.5 / 2.0).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from jev_ml.core.anomalies import band as _band
from jev_ml.core.anomalies import suppression
from jev_ml.core.common import clip01, evidence, fnum, iso, iso_from_epoch, stable_id
from jev_ml.domains.movie.config import IntelConfig
from jev_ml.domains.movie.ingest import NEGATIVE_FEEDBACK, Prepared

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
LIVE_LIFT_BANDS = {"low": 1.0, "medium": 1.2, "high": 1.5, "critical": 2.0}


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
        strength = clip01((float(fnum(row["if_score"], 4) or 0.0) - 0.5) / 0.3)
        out[-1].update(
            {
                "observed_value": out[-1]["value"],
                "expected_value": out[-1]["baseline"],
                "anomaly_score": fnum(strength, 4),
                "confidence": fnum(strength, 4),
                "confidence_kind": "margin",
                "entity_id": f"user:{uid}",
                "source": "movielens",
                "adverse": None,
                "severity_basis": {
                    "measure": "isolation-forest score",
                    "value": fnum(row["if_score"], None),
                    "bands": {"low": 0.5, **cfg.rater_severity_bands},
                },
            }
        )
    return out, feats


# --------------------------------------------------------------------------------------------------
# live feedback


def live_feedback(
    prep: Prepared, cfg: IntelConfig, as_of_key: str, suppressed: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fb = prep.app_feedback
    # the replay clock in a replay (P2.4), the wall clock in a live run
    now = prep.event_clock_ts if prep.event_clock_ts is not None else prep.now_ts
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
    for name in ("medium", "high", "critical"):
        if lift >= LIVE_LIFT_BANDS[name]:
            severity = name
    key = "anomaly:live_feedback:app"
    reason = suppression(key, severity, suppressed)
    clock = iso_from_epoch(now)  # = iso(prep.now) in a live run; the as_of in a replay (P2.4)
    anom: dict[str, Any] = {
        "id": stable_id("anom", "live_feedback", "app", (clock or "")[:10], as_of_key),
        "dedup_key": key,
        "kind": "live_feedback",
        "entity_type": "platform",
        "entity": "app",
        "series_id": None,
        "metric": "negative_feedback_rate",
        "detected_at": clock,
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
    strength = clip01(abs(float(fnum(z, 3) or 0.0)) / 4.0)  # z of 4 = full strength (as v1.1)
    anom.update(
        {
            "observed_value": anom["value"],
            "expected_value": anom["baseline"],
            "anomaly_score": fnum(strength, 4),
            "confidence": fnum(strength, 4),
            "confidence_kind": "margin",
            "entity_id": "platform:app",
            "source": "app",
            "adverse": True,
            "severity_basis": {
                "measure": "negative-feedback lift (recent / prior rate)",
                "value": fnum(min(lift, 1e6), None),
                "bands": dict(LIVE_LIFT_BANDS),
            },
        }
    )
    return [anom], status


def anomaly_warning_text(a: dict[str, Any], cfg: IntelConfig) -> dict[str, Any] | None:
    """Warning wording for the movie domain's anomalies (the v1.1 texts)."""
    if a["method"] == "robust_z":
        from jev_ml.core.warnings import robust_z_text

        t = robust_z_text(a, cfg)
        t["action"] = (
            "Check whether the change comes from a few users or a catalogue event before acting on it."
        )
        return t
    if a["method"] == "isolation_forest":
        band = cfg.rater_severity_bands.get(a["severity"])
        return {
            "condition": f"isolation-forest score {a['value']:.3f} >= {band}",
            "observed": a["value"],
            "threshold": band if band is not None else 0.0,
            "title": f"Rater {a['entity']} behaves anomalously",
            "description": (
                f"User {a['entity']} is in the top {100 - a['score']:.1f} % most anomalous raters "
                f"(score {a['value']:.3f})."
            ),
            "action": "Review the rater's history; see the rater_action decision.",
        }
    if a["method"] == "rate_test":
        return {
            "condition": f"one-sided two-proportion test p < {cfg.live_alpha}",
            "observed": a["value"],
            "threshold": a["baseline"],
            "title": "Negative recommendation feedback rising",
            "description": (
                f"Negative feedback share {a['value']:.2%} in the last 7 days vs {a['baseline']:.2%} before."
            ),
            "action": "Inspect which reason codes attract dislikes and adjust those rails.",
        }
    return None
