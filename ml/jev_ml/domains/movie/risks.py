"""ASSESS (movie domain): the movie-specific risk kinds, scored with the core risk framework.

``genre_demand_decline`` (share down-trend surviving the FDR; impact = share / genre_impact_share_ref;
attached to its share series), ``audience_lapse`` (lapse model), ``rating_manipulation`` (flagged
raters; impact = exact top-50 trending churn when they are removed), ``model_staleness`` and
``model_quality`` (model artefact; skipped on replays that would leak the future) and
``recommendation_rejection`` (live feedback). Score, level and factors: ``jev_ml.core.risk``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import clip01, evidence, fnum, iso_from_epoch
from jev_ml.core.risk import make_risk
from jev_ml.domains.movie.config import IntelConfig
from jev_ml.domains.movie.ingest import Prepared
from jev_ml.domains.movie.modelstats import influence, new_events_since_training
from jev_ml.domains.movie.raters import RATER_FEATURES


def genre_risks(
    trends: list[dict[str, Any]],
    forecasts: dict[str, dict[str, Any]],
    dq: float,
    cfg: IntelConfig,
    as_of_key: str,
) -> list[dict[str, Any]]:
    out = []
    for t in trends:
        if t["metric"] != "share" or t["direction"] != "down" or (t["q_value"] or 1.0) > cfg.trend_fdr:
            continue
        g = t["entity"]
        share = t["recent_mean"] or 0.0
        impact = share / cfg.genre_impact_share_ref
        coverage = t["n_points"] / t["window"]["months"]
        conf = coverage * (1.0 - (t["q_value"] or 0.0))
        fc = forecasts.get(f"volume:genre:{g}")
        ev = [
            evidence(
                "test",
                "Mann–Kendall p",
                t["p_value"],
                f"tau {t['kendall_tau']}, BH q {t['q_value']}",
                t["id"],
            ),
            evidence(
                "metric",
                "Theil–Sen slope (share/month)",
                t["slope"],
                f"95 % CI {t['slope_ci']}; {t['change_rate_pct_per_month']} %/month",
                t["series_id"],
            ),
        ]
        if fc:
            nxt = sum(p["mean"] or 0 for p in fc["points"])
            ev.append(
                evidence(
                    "model",
                    f"forecast volume next {fc['horizon_months']} m",
                    fnum(nxt, 1),
                    fc["model"],
                    fc["id"],
                )
            )
        out.append(
            make_risk(
                "genre_demand_decline",
                "genre",
                g,
                f"{g} demand declining",
                float(t["evidence_strength"] or 0.0),
                impact,
                conf,
                dq,
                {
                    "likelihood": "1 - Mann–Kendall p of the share down-trend",
                    "impact": f"recent share {share:.3f} / {cfg.genre_impact_share_ref}",
                    "confidence": f"months with data {coverage:.2f} x (1 - BH q)",
                },
                ev,
                f"Review {g} programming: fewer {g} slots on the home rails, or refresh the {g} catalogue.",
                cfg,
                as_of_key,
                series_id=t["series_id"],
            )
        )
    return out


def lapse_risk(
    lapse: dict[str, Any], scored: pd.DataFrame, prep: Prepared, dq: float, cfg: IntelConfig, as_of_key: str
) -> list[dict[str, Any]]:
    pop = lapse.get("population")
    if lapse.get("status") != "ok" or not pop or not pop.get("n_scored") or scored is None or not len(scored):
        return []
    m = lapse["metrics"]
    look = prep.ratings[prep.ratings["timestamp"] > prep.as_of_ts - cfg.lapse_lookback_days * 86400]
    vol = look.groupby("user_id").size()
    high = scored.index[scored["p"] >= cfg.lapse_high_risk]
    impact = float(vol.reindex(high).fillna(0).sum() / max(vol.sum(), 1))
    conf = clip01(2 * ((m["auc"] or 0.5) - 0.5)) * (1 - (m["ece"] or 0.0))
    ev = [
        evidence(
            "model", "expected lapses", pop["expected_lapses"], f"of {pop['n_scored']} recently active users"
        ),
        evidence("metric", "high-risk users", pop["high_risk"], f"p >= {cfg.lapse_high_risk}"),
        evidence(
            "metric",
            "historical lapse rate (test cutoffs)",
            m["base_rate"],
            "observed share lapsing in 180 d",
        ),
        evidence("test", "holdout AUC", m["auc"], f"recency baseline {m['baseline_auc']}; ECE {m['ece']}"),
    ]
    return [
        make_risk(
            "audience_lapse",
            "platform",
            "all",
            f"{pop['high_risk']} of {pop['n_scored']} active raters likely to lapse",
            float(pop["mean_p"]),
            impact,
            conf,
            dq,
            {
                "likelihood": "mean calibrated P(no rating in 180 d) over recently active users",
                "impact": "share of the last-365-day rating volume from high-risk users",
                "confidence": "2*(AUC-0.5) x (1-ECE) on the temporal holdout",
            },
            ev,
            "Target the high-risk users with a re-engagement message (new films in their top genres).",
            cfg,
            as_of_key,
        )
    ]


def manipulation_risk(
    raters: list[dict[str, Any]],
    feats: pd.DataFrame,
    prep: Prepared,
    dq: float,
    cfg: IntelConfig,
    as_of_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    active = [a for a in raters if not a["suppressed"]]
    if not active:
        return [], None
    users = [int(a["entity"]) for a in active]
    infl = influence(prep.ratings, prep.as_of_ts, cfg, users)
    strength = float(np.mean([clip01((a["value"] - 0.5) / 0.3) for a in active]))
    pct = feats[list(RATER_FEATURES)].rank(pct=True)
    extreme = ((pct <= 0.05) | (pct >= 0.95)).mean(axis=1)
    agree = float(extreme.reindex(users).mean())
    ev = [
        evidence("record", f"user {a['entity']}", a["value"], f"IF percentile {a['score']}", a["id"])
        for a in active
    ] + [
        evidence(
            "test",
            f"top-{infl['top_n']} trending films changed without these users",
            infl["changed"],
            f"left: {infl['left'][:10]}",
        )
    ]
    risk = make_risk(
        "rating_manipulation",
        "platform",
        "all",
        f"{len(active)} active raters with anomalous behaviour",
        strength,
        infl["changed"] / infl["top_n"],
        agree,
        dq,
        {
            "likelihood": (
                "mean of clip((IF score - 0.5)/0.3) over flagged active raters "
                "(anomaly strength, not a probability of fraud)"
            ),
            "impact": (
                f"exact: {infl['changed']} of the top-{infl['top_n']} trending films change "
                "when their ratings are removed"
            ),
            "confidence": "share of the flagged raters' features beyond the population 5th/95th percentile",
        },
        ev,
        "Review the flagged raters; exclude quarantined profiles from popularity/trending signals.",
        cfg,
        as_of_key,
    )
    return [risk], infl


def model_risks(
    prep: Prepared,
    manifest: dict[str, Any] | None,
    boot: dict[str, Any] | None,
    dq: float,
    cfg: IntelConfig,
    as_of_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if not manifest or not prep.model_replayable:
        return [], None
    out = []
    created = pd.Timestamp(manifest["created_at"]).timestamp() if manifest.get("created_at") else None
    new = new_events_since_training(prep.ratings, prep.app_ratings, prep.model_cutoff_ts, created)
    rows = int(manifest.get("trained_on_rows") or 0)
    frac = new["total"] / rows if rows else 0.0
    stale = {
        "new_events": new,
        "trained_on_rows": rows,
        "new_frac": frac,
        "cutoff": iso_from_epoch(prep.model_cutoff_ts),
    }
    if prep.model_cutoff_ts is not None and rows:
        out.append(
            make_risk(
                "model_staleness",
                "model",
                str(manifest.get("version")),
                (
                    f"{new['total']:,} new ratings since the model was trained ({frac:.1%} of training data)"
                    if new["total"]
                    else "Model is up to date: no new ratings since training"
                ),
                frac / cfg.retrain_new_event_frac,
                cfg.risk_declared_impact["model_staleness"],
                1.0,
                dq,
                {
                    "likelihood": (
                        f"new-event share {frac:.4f} / retrain threshold {cfg.retrain_new_event_frac}"
                    ),
                    "impact": "declared impact weight",
                    "confidence": "exact counts",
                },
                [
                    evidence(
                        "metric",
                        "new MovieLens ratings since training cutoff",
                        new["movielens"],
                        str(stale["cutoff"]),
                    ),
                    evidence(
                        "metric",
                        "new app ratings since model creation",
                        new["app"],
                        manifest.get("created_at"),
                    ),
                    evidence("record", "trained on rows", rows, str(manifest.get("dataset_version"))),
                ],
                "Retrain once enough new interactions accumulate (see the retrain decision).",
                cfg,
                as_of_key,
                confidence_kind="rule",
            )
        )
    if boot and not boot.get("insufficient") and boot.get("hybrid"):
        h = boot["hybrid"]
        width = h["lead_ci95"][1] - h["lead_ci95"][0]
        conf = clip01(1.0 - width / (2 * max(abs(h["lead"]), cfg.model_min_lead)))
        out.append(
            make_risk(
                "model_quality",
                "model",
                str(manifest.get("version")),
                f"Hybrid lead over {h['best_single']} is {100 * h['lead']:.1f} %",
                h["p_lead_below_min"],
                cfg.risk_declared_impact["model_quality"],
                conf,
                1.0,
                {
                    "likelihood": (
                        f"bootstrap P(hybrid lead < {cfg.model_min_lead:.0%}) "
                        f"over {boot['n_users']} test users"
                    ),
                    "impact": "declared impact weight",
                    "confidence": "1 - bootstrap CI width / (2 x max(|lead|, min lead))",
                },
                [
                    evidence(
                        "test",
                        "hybrid NDCG@10 lead (95 % CI)",
                        fnum(h["lead"], 4),
                        f"{[round(x, 4) for x in h['lead_ci95']]}",
                    ),
                    evidence(
                        "model",
                        "mean NDCG@10",
                        fnum(boot["mean_ndcg"].get("hybrid"), 4),
                        f"best single {h['best_single']}: {boot['mean_ndcg'][h['best_single']]:.4f}",
                    ),
                ],
                "If the lead is small, revisit hybrid weights or fall back to the best single model.",
                cfg,
                as_of_key,
            )
        )
    return out, stale


def rejection_risk(live: dict[str, Any], dq: float, cfg: IntelConfig, as_of_key: str) -> list[dict[str, Any]]:
    if live.get("status") != "ok":
        return []
    return [
        make_risk(
            "recommendation_rejection",
            "platform",
            "app",
            f"{100 * live['recent_rate']:.0f} % of recent recommendation feedback is negative",
            live["recent_rate"],
            cfg.risk_declared_impact["recommendation_rejection"],
            1.0 - float(live["p_value"]),
            dq,
            {
                "likelihood": "negative share of feedback in the last 7 days",
                "impact": "declared impact weight",
                "confidence": "1 - p of the increase vs the prior 28 days",
            },
            [
                evidence(
                    "metric",
                    "negative rate 7 d",
                    fnum(live["recent_rate"], 4),
                    f"{live['recent_negative']}/{live['recent_n']}",
                ),
                evidence(
                    "metric",
                    "negative rate prior 28 d",
                    fnum(live["prior_rate"], 4),
                    f"{live['prior_negative']}/{live['prior_n']}",
                ),
            ],
            "Inspect which reason codes attract dislikes / not-interested and adjust those rails.",
            cfg,
            as_of_key,
        )
    ]
