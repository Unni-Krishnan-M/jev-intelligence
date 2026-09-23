"""ASSESS: risks computed from the detected/predicted evidence.

score = 100 * likelihood * impact * confidence * data_quality (0..100), i.e. the expected-loss
product shrunk towards 0 when the evidence is weak or the data is poor.
level: score >= risk_levels.critical -> critical, >= high -> high, >= medium -> medium, else low.

Factors: every multiplicative input is listed with weight 1.0 (multiplicative) and
``contribution`` = the points of score the factor removes (score with the factor at 1 minus the
actual score); a factor at 1.0 removes nothing. Composite inputs list their parts in ``detail``.

A risk kind is only produced when its evidence exists: no down-trend -> no genre risk, no app
data -> no staleness/rejection risk, model trained after as_of -> no model risks (replay guard).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.intel.common import clip01, evidence, fnum, iso_from_epoch, stable_id
from jev_ml.intel.config import IntelConfig
from jev_ml.intel.ingest import Prepared, quality_evidence
from jev_ml.intel.modelstats import influence, new_events_since_training


def level_of(score: float, cfg: IntelConfig) -> str:
    lv = "low"
    for name in ("medium", "high", "critical"):
        if score >= cfg.risk_levels[name]:
            lv = name
    return lv


def make_risk(
    kind: str,
    entity_type: str,
    entity: str,
    title: str,
    likelihood: float,
    impact: float,
    confidence: float,
    data_quality: float,
    details: dict[str, str],
    ev: list[dict[str, Any]],
    response: str,
    cfg: IntelConfig,
    as_of_key: str,
) -> dict[str, Any]:
    lik, imp, conf, qual = clip01(likelihood), clip01(impact), clip01(confidence), clip01(data_quality)
    score = 100.0 * lik * imp * conf * qual
    vals = {"likelihood": lik, "impact": imp, "confidence": conf, "data_quality": qual}
    factors = []
    for name, v in vals.items():
        without = 100.0 * np.prod([1.0 if k == name else x for k, x in vals.items()])
        factors.append(
            {
                "name": name,
                "value": fnum(v, 4),
                "weight": 1.0,
                "contribution": fnum(without - score, 3),
                "detail": details.get(name),
            }
        )
    return {
        "id": stable_id("risk", kind, entity, as_of_key),
        "kind": kind,
        "title": title,
        "entity_type": entity_type,
        "entity": entity,
        "likelihood": fnum(lik, 4),
        "impact": fnum(imp, 4),
        "exposure": fnum(lik * imp, 4),
        "confidence": fnum(conf, 4),
        "data_quality": fnum(qual, 4),
        "score": fnum(score, 2),
        "level": level_of(score, cfg),
        "factors": factors,
        "evidence": ev,
        "recommended_response": response,
    }


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
    from jev_ml.intel.anomalies import RATER_FEATURES

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


def quality_risk(prep: Prepared, cfg: IntelConfig, as_of_key: str) -> list[dict[str, Any]]:
    failed = [c for c in prep.quality["checks"] if not c["passed"] and c["severity"] != "info"]
    if not failed:
        return []
    q = prep.quality["score"]
    err = any(c["severity"] == "error" for c in failed)
    return [
        make_risk(
            "data_quality",
            "source",
            ",".join(sorted({c["source"] for c in failed})),
            f"{len(failed)} data-quality checks failed",
            1.0 - q,
            cfg.risk_declared_impact["data_quality"] * (1.0 if err else 0.5),
            1.0,
            1.0,
            {
                "likelihood": "1 - weighted share of checks passed",
                "impact": "declared impact weight (halved when only warning-level checks fail)",
                "confidence": "deterministic checks",
            },
            quality_evidence(prep.quality),
            "Inspect the failing checks; invalid rows are already excluded from this run.",
            cfg,
            as_of_key,
        )
    ]


def staleness_risks(prep: Prepared, cfg: IntelConfig, as_of_key: str) -> list[dict[str, Any]]:
    out = []
    for s in prep.sources:
        if s["expected_update"] is None or s["fresh"] is not False:
            continue  # no SLA, or no events to judge
        age = float(s["age_days"])
        lik = 1.0 - float(np.exp(-(age - cfg.live_fresh_days) / cfg.live_fresh_days))
        out.append(
            make_risk(
                "data_staleness",
                "source",
                s["source"],
                f"{s['source']} data stale ({age:.1f} days since last event)",
                lik,
                cfg.risk_declared_impact["data_staleness"],
                1.0,
                1.0,
                {
                    "likelihood": f"1 - exp(-(age - {cfg.live_fresh_days}) / {cfg.live_fresh_days})",
                    "impact": "declared impact weight",
                    "confidence": "exact timestamps",
                },
                [evidence("record", "last event", s["last_event"], f"expected {s['expected_update']}")],
                "Check the app's event ingestion (API health, background jobs).",
                cfg,
                as_of_key,
            )
        )
    return out


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
