"""DECIDE: bounded, typed decisions with versioned policies.

Each decision answers one fixed question with a fixed option set. The policy only combines evidence
computed upstream; confidence carries its kind:

* ``retrain_model`` (boolean, rule): yes iff new-event share since training >= threshold or the
  model was trained on another dataset version. confidence 1.0 because it is a deterministic rule.
* ``serving_model`` (choice, probability): option scores = bootstrap P(model is best) on per-user
  NDCG@10; confidence = P(best beats runner-up) under the paired bootstrap.
* ``genre_programming`` (choice boost|hold|reduce, margin): standardised Theil–Sen slope of the
  genre's share, z = slope / se (se from the 95 % CI). Scores: boost = max(0, z - c),
  reduce = max(0, -z - c), hold = c, c = 1.96; normalised to sum 1. A boost/reduce answer is
  downgraded to hold unless the trend survives the BH FDR (q <= trend_fdr). confidence =
  (best - second) / (best + second): a normalised margin, not a probability.
* ``rater_action`` (choice ignore|monitor|quarantine, margin): a = clip((IF score - 0.5) / 0.3),
  infl = min(1, top-50 films changed by removing the user / rater_influence_ref);
  ignore = 1 - a, monitor = a * (1 - infl), quarantine = a * infl (inactive users: ignore = 1).
* ``reengagement_campaign`` (boolean, probability): yes iff the lapse model is usable
  (AUC >= reengage_min_auc) and P(at least reengage_min_users of the high-risk users lapse) >= 0.5,
  that probability computed exactly (Poisson-binomial) from the calibrated lapse probabilities.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.intel.common import clip01, evidence, fnum, stable_id
from jev_ml.intel.config import IntelConfig

Z_CRIT = 1.96


def _decision(
    key: str,
    policy: str,
    question: str,
    kind: str,
    options: list[str],
    answer: str | None,
    scores: dict[str, float],
    confidence: float | None,
    confidence_kind: str,
    state: dict[str, Any],
    rationale: list[str],
    ev: list[dict[str, Any]],
    entity_type: str,
    entity: str,
    as_of_key: str,
    fallback: str | None = None,
) -> dict[str, Any]:
    return {
        "id": stable_id("dec", key, entity, as_of_key),
        "key": key,
        "spec_id": key,
        "policy_version": policy,
        "question": question,
        "kind": kind,
        "options": options,
        "answer": answer,
        "option_scores": {k: fnum(v, 4) for k, v in scores.items()},
        "confidence": fnum(confidence, 4),
        "confidence_kind": confidence_kind,
        "state": state,
        "rationale": rationale,
        "evidence": ev,
        "abstained": fallback is not None,
        "fallback_reason": fallback,
        "entity_type": entity_type,
        "entity": entity,
    }


def retrain_decision(
    manifest: dict[str, Any] | None,
    stale: dict[str, Any] | None,
    data_version: str,
    replayable: bool,
    cfg: IntelConfig,
    as_of_key: str,
) -> dict[str, Any]:
    q = "Should the recommendation model be retrained now?"
    ent = str(manifest.get("version")) if manifest else "none"
    args = ("retrain_model", "retrain-1.0.0", q, "boolean", ["yes", "no"])
    if not manifest:
        return _decision(
            *args, None, {}, None, "rule", {}, [], [], "model", ent, as_of_key, "no model manifest available"
        )
    if not replayable or stale is None:
        return _decision(
            *args,
            None,
            {},
            None,
            "rule",
            {},
            [],
            [],
            "model",
            ent,
            as_of_key,
            "model was trained on data after as_of (replay): a retrain decision would use future information",
        )
    frac = stale["new_frac"]
    version_mismatch = manifest.get("dataset_version") != data_version
    fire = frac >= cfg.retrain_new_event_frac or version_mismatch
    state = {
        "new_events": stale["new_events"],
        "trained_on_rows": stale["trained_on_rows"],
        "new_event_frac": fnum(frac, 6),
        "threshold": cfg.retrain_new_event_frac,
        "dataset_version_model": manifest.get("dataset_version"),
        "dataset_version_data": data_version,
        "training_cutoff": stale["cutoff"],
    }
    rationale = [
        f"{stale['new_events']['total']} new ratings since training = {100 * frac:.2f} % of the "
        f"{stale['trained_on_rows']} training rows (threshold {100 * cfg.retrain_new_event_frac:.0f} %)",
        "model dataset version matches the data"
        if not version_mismatch
        else "model was trained on a different dataset version",
    ]
    return _decision(
        *args,
        "yes" if fire else "no",
        {"yes": 1.0 if fire else 0.0, "no": 0.0 if fire else 1.0},
        1.0,
        "rule",
        state,
        rationale,
        [evidence("metric", "new-event share", fnum(frac, 6), f"threshold {cfg.retrain_new_event_frac}")],
        "model",
        ent,
        as_of_key,
    )


def serving_decision(
    manifest: dict[str, Any] | None,
    boot: dict[str, Any] | None,
    replayable: bool,
    cfg: IntelConfig,
    as_of_key: str,
) -> dict[str, Any]:
    q = "Which evaluated model should serve recommendations?"
    ent = str(manifest.get("version")) if manifest else "none"
    options = boot["models"] if boot and not boot.get("insufficient") else []
    args = ("serving_model", "serving-1.0.0", q, "choice", options)
    if not manifest or not replayable:
        reason = (
            "no model manifest available" if not manifest else "model evaluated on data after as_of (replay)"
        )
        return _decision(*args, None, {}, None, "probability", {}, [], [], "model", ent, as_of_key, reason)
    if not boot:
        return _decision(
            *args,
            None,
            {},
            None,
            "probability",
            {},
            [],
            [],
            "model",
            ent,
            as_of_key,
            "no per-user evaluation results",
        )
    if boot.get("insufficient"):
        return _decision(
            *args,
            None,
            {},
            None,
            "probability",
            {"n_users": boot["n_users"]},
            [],
            [],
            "model",
            ent,
            as_of_key,
            f"only {boot['n_users']} evaluated users (need >= {cfg.bootstrap_min_users})",
        )
    best, runner = boot["best"], boot["runner_up"]
    conf = boot["p_best_beats_runner_up"]
    state = {
        "mean_ndcg10": {k: fnum(v, 5) for k, v in boot["mean_ndcg"].items()},
        "n_users": boot["n_users"],
        "bootstrap_samples": cfg.bootstrap_samples,
        "current_model_type": manifest.get("model_type"),
    }
    rationale = [
        f"{best} has the highest mean NDCG@10 ({boot['mean_ndcg'][best]:.4f}) vs {runner} "
        f"({boot['mean_ndcg'][runner]:.4f})",
        f"paired bootstrap over {boot['n_users']} users: P({best} > {runner}) = {conf:.3f}",
    ]
    if manifest.get("model_type") and manifest["model_type"] != best:
        rationale.append(f"currently serving {manifest['model_type']}")
    return _decision(
        *args,
        best,
        boot["p_best"],
        conf,
        "probability",
        state,
        rationale,
        [evidence("test", f"P({best} > {runner})", fnum(conf, 4), "paired bootstrap on per-user NDCG@10")],
        "model",
        ent,
        as_of_key,
    )


def genre_decisions(
    trends: list[dict[str, Any]], forecasts: dict[str, dict[str, Any]], cfg: IntelConfig, as_of_key: str
) -> list[dict[str, Any]]:
    out = []
    shares = {t["entity"]: t for t in trends if t["metric"] == "share"}
    for g, t in sorted(shares.items()):
        q = f"How should {g} be programmed on the home rails?"
        args = ("genre_programming", "genre-1.0.0", q, "choice", ["boost", "hold", "reduce"])
        if (t["recent_mean"] or 0.0) < cfg.genre_min_share:
            continue
        lo, hi = t["slope_ci"]
        slope = t["slope"] or 0.0
        if lo is None or hi is None or hi - lo <= 0:
            out.append(
                _decision(
                    *args,
                    None,
                    {},
                    None,
                    "margin",
                    {"slope": slope},
                    [],
                    [],
                    "genre",
                    g,
                    as_of_key,
                    "degenerate trend estimate (constant series)",
                )
            )
            continue
        se = (hi - lo) / (2 * Z_CRIT)
        z = slope / se
        scores = {"boost": max(0.0, z - Z_CRIT), "hold": Z_CRIT, "reduce": max(0.0, -z - Z_CRIT)}
        tot = sum(scores.values())
        scores = {k: v / tot for k, v in scores.items()}
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        answer = ranked[0][0]
        qv = t["q_value"] if t["q_value"] is not None else 1.0
        rationale = [
            f"share trend {t['direction']}: slope {slope:.5f}/month (95 % CI {lo:.5f}..{hi:.5f}), z = "
            f"{z:.2f}",
            f"Mann–Kendall p = {t['p_value']:.4f}, BH q = {qv:.4f}",
        ]
        if answer != "hold" and qv > cfg.trend_fdr:
            rationale.append(f"downgraded to hold: q {qv:.3f} > FDR {cfg.trend_fdr}")
            answer = "hold"
        second = max(v for k, v in scores.items() if k != answer)
        conf = (scores[answer] - second) / (scores[answer] + second) if scores[answer] + second > 0 else 0.0
        fc = forecasts.get(f"volume:genre:{g}")
        state = {
            "series_id": t["series_id"],
            "recent_share": t["recent_mean"],
            "prior_share": t["prior_mean"],
            "slope": slope,
            "slope_ci": [lo, hi],
            "z": fnum(z, 4),
            "p_value": t["p_value"],
            "q_value": qv,
            "forecast_next_total": fnum(sum(p["mean"] or 0 for p in fc["points"]), 2) if fc else None,
            "forecast_model": fc["model"] if fc else None,
        }
        out.append(
            _decision(
                *args,
                answer,
                scores,
                max(0.0, conf),
                "margin",
                state,
                rationale,
                [evidence("series", f"{g} share trend", t["slope"], t["direction"], t["id"])],
                "genre",
                g,
                as_of_key,
            )
        )
    return out


def rater_decisions(
    raters: list[dict[str, Any]], infl: dict[str, Any] | None, cfg: IntelConfig, as_of_key: str
) -> list[dict[str, Any]]:
    out = []
    per_user = (infl or {}).get("per_user", {})
    # only raters still active (inactive ones are reported as suppressed anomalies; deciding on
    # them would add noise), highest anomaly score first
    active = [a for a in raters if not (a["suppression_reason"] or "").startswith("inactive")]
    ranked = sorted(active, key=lambda a: (-(a["value"] or 0.0), a["entity"]))
    for a in ranked[: cfg.rater_decisions_max]:
        uid = int(a["entity"])
        q = f"What should happen to rater {uid}'s ratings?"
        args = ("rater_action", "rater-1.0.0", q, "choice", ["ignore", "monitor", "quarantine"])
        strength = clip01(((a["value"] or 0.5) - 0.5) / 0.3)
        inactive = bool(a["suppression_reason"] and a["suppression_reason"].startswith("inactive"))
        changed = per_user.get(uid)
        if not inactive and changed is None:
            out.append(
                _decision(
                    *args,
                    None,
                    {},
                    None,
                    "margin",
                    {"if_score": a["value"]},
                    [],
                    [],
                    "user",
                    str(uid),
                    as_of_key,
                    "influence not computed for this rater",
                )
            )
            continue
        infl_n = clip01((changed or 0) / cfg.rater_influence_ref)
        if inactive:
            scores = {"ignore": 1.0, "monitor": 0.0, "quarantine": 0.0}
        else:
            scores = {
                "ignore": 1 - strength,
                "monitor": strength * (1 - infl_n),
                "quarantine": strength * infl_n,
            }
        ranked_s = sorted(
            scores.items(), key=lambda kv: (-kv[1], ["ignore", "monitor", "quarantine"].index(kv[0]))
        )
        (best, s1), (_, s2) = ranked_s[0], ranked_s[1]
        conf = (s1 - s2) / (s1 + s2) if s1 + s2 > 0 else 0.0
        state = {
            "if_score": a["value"],
            "percentile": a["score"],
            "anomaly_strength": fnum(strength, 4),
            "top50_changed_if_removed": changed,
            "influence_norm": fnum(infl_n, 4),
            "inactive": inactive,
            "features": a["features"],
        }
        rationale = [
            f"IsolationForest score {a['value']:.3f} (percentile {a['score']})",
            "inactive for longer than the review window"
            if inactive
            else f"removing this user changes {changed} of the top-{cfg.influence_top_n} trending films",
        ]
        out.append(
            _decision(
                *args,
                best,
                scores,
                conf,
                "margin",
                state,
                rationale,
                [evidence("record", f"anomaly {a['id']}", a["value"], a["severity"], a["id"])],
                "user",
                str(uid),
                as_of_key,
            )
        )
    return out


def poisson_binomial_tail(p: np.ndarray, k: int) -> float:
    """P(X >= k) for X = sum of independent Bernoulli(p_i), exact DP."""
    dist = np.zeros(len(p) + 1)
    dist[0] = 1.0
    for pi in p:
        dist[1:] = dist[1:] * (1 - pi) + dist[:-1] * pi
        dist[0] *= 1 - pi
    return float(dist[k:].sum())


def reengagement_decision(
    lapse: dict[str, Any], scored: pd.DataFrame | None, cfg: IntelConfig, as_of_key: str
) -> dict[str, Any]:
    q = "Should a re-engagement campaign target users at high lapse risk now?"
    args = ("reengagement_campaign", "reengage-1.0.0", q, "boolean", ["yes", "no"])
    if lapse.get("status") != "ok" or scored is None or not len(scored):
        return _decision(
            *args,
            None,
            {},
            None,
            "probability",
            {},
            [],
            [],
            "platform",
            "all",
            as_of_key,
            f"lapse model unavailable: {lapse.get('detail') or lapse.get('status')}",
        )
    auc = lapse["metrics"]["auc"] or 0.0
    if auc < cfg.reengage_min_auc:
        return _decision(
            *args,
            None,
            {},
            None,
            "probability",
            {"auc": auc},
            [],
            [],
            "platform",
            "all",
            as_of_key,
            f"lapse model holdout AUC {auc:.3f} < {cfg.reengage_min_auc}: not trustworthy enough to "
            "target users",
        )
    high = scored[scored["p"] >= cfg.lapse_high_risk]["p"].to_numpy()
    k = cfg.reengage_min_users
    p_yes = poisson_binomial_tail(high, k) if len(high) >= k else 0.0
    state = {
        "n_scored": len(scored),
        "high_risk": len(high),
        "threshold": cfg.lapse_high_risk,
        "min_users": k,
        "expected_lapses_high_risk": fnum(high.sum(), 3),
        "auc": auc,
        "ece": lapse["metrics"]["ece"],
    }
    answer = "yes" if p_yes >= 0.5 else "no"
    return _decision(
        *args,
        answer,
        {"yes": p_yes, "no": 1 - p_yes},
        p_yes if answer == "yes" else 1 - p_yes,
        "probability",
        state,
        [
            f"{len(high)} of {len(scored)} recently active users have P(lapse) >= {cfg.lapse_high_risk}",
            f"P(at least {k} of them lapse within {lapse['horizon_days']} d) = {p_yes:.3f} (calibrated "
            f"lapse model, AUC {auc:.3f})",
        ],
        [
            evidence(
                "model", "expected lapses among high-risk users", fnum(high.sum(), 2), lapse["model_version"]
            )
        ],
        "platform",
        "all",
        as_of_key,
    )
