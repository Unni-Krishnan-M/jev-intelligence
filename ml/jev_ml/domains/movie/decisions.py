"""DECIDE (movie domain): the movie decisions, built with the core decision framework.

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
* ``editorial_slot_share`` (score, interval): the recommended % of home-rail slots that should carry
  a film of the genre next quarter. Policy (``slot-share-1.0.0``): *demand-proportional* allocation,
  i.e. slots follow the forecast share of rating activity. answer = 100 x the mean forecast
  ``share:genre:<G>`` over the next ``slot_share_window_months`` months (the first months after the
  last complete month); ``answer_interval`` = 100 x the conformal 80 % interval *of that window mean*
  (backtest residuals of the window mean, see ``forecast.window_mean_forecast``); both clipped to
  [0, 100]. confidence = the nominal coverage (0.8), kind ``interval``: it is the coverage the
  interval is built for, not a probability that the answer is right; the honestly backtested
  coverage is in ``state``. A film carries several genres, so slot shares of different genres do not
  sum to 100. Editorial boosts/reductions are the separate ``genre_programming`` question (same
  batch); this answer is not adjusted by it. Abstains when the genre has no share forecast
  (history too short), too few complete backtest windows for an interval, or a backtested interval
  coverage below ``slot_share_min_coverage`` (an interval that covered less than that cannot be
  reported as an 80 % interval).

Every decision is produced inside a *decision batch* (``jev_ml.core.batches``); ``batch_id`` links it to
the shared, hashed state snapshot it was answered from. Non-score decisions carry
``scale: null`` and ``answer_interval: null``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.common import clip01, evidence, fnum
from jev_ml.core.config import EWL_POLICY_VERSION
from jev_ml.core.decisions import decision as _decision
from jev_ml.domains.movie.config import IntelConfig

Z_CRIT = 1.96

# policy version per decision key (also recorded per batch in decision_batches[].policy_versions)
POLICY_VERSIONS = {
    "retrain_model": "retrain-1.0.0",
    "serving_model": "serving-1.0.0",
    "genre_programming": "genre-1.0.0",
    "editorial_slot_share": "slot-share-1.0.0",
    "rater_action": "rater-1.0.0",
    "reengagement_campaign": "reengage-1.0.0",
    "early_warning_level": EWL_POLICY_VERSION,  # core decision, produced for every domain
}
SLOT_SCALE = {"min": 0.0, "max": 100.0, "unit": "% of home-rail slots"}


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


# --------------------------------------------------------------------------------------------------
# score decisions


def share_window_inputs(
    share_forecasts: list[dict[str, Any]], states: dict[str, Any], cfg: IntelConfig
) -> dict[str, dict[str, Any]]:
    """Plain-data summary per ``share:genre:*`` forecast (the slot-share decision's input; it goes
    into the batch's hashed state snapshot, so it holds numbers only, no fitted objects)."""
    from jev_ml.core.forecast import window_mean_forecast

    out: dict[str, dict[str, Any]] = {}
    for f in share_forecasts:
        st = states.get(f["series_id"])
        w = window_mean_forecast(st, cfg.slot_share_window_months, cfg) if st is not None else None
        base = {
            "forecast_id": f["id"],
            "model": f["model"],
            "model_version": f["model_version"],
            "backtest_mase": f["backtest"].get("mase"),
            "backtest_naive_mase": f["backtest"].get("naive_mase"),
        }
        if w is None:
            out[f["series_id"]] = {
                **base,
                "status": "insufficient",
                "reason": f"fewer than {cfg.forecast_min_residuals} complete "
                f"{cfg.slot_share_window_months}-month backtest windows for an interval",
            }
            continue
        out[f["series_id"]] = {
            **base,
            "status": "ok",
            "months": w["months"],
            "mean": fnum(w["mean"], 6),
            "lo": fnum(w["lo"], 6),
            "hi": fnum(w["hi"], 6),
            "n_residuals": w["n_residuals"],
            "coverage": fnum(w["coverage"], 4),
            "coverage_origins": w["coverage_origins"],
            "nominal": w["nominal"],
        }
    return out


def _pct(x: float) -> float:
    return round(100.0 * clip01(x), 2)


def slot_share_decisions(
    trends: list[dict[str, Any]], windows: dict[str, dict[str, Any]], cfg: IntelConfig, as_of_key: str
) -> list[dict[str, Any]]:
    """``editorial_slot_share`` per genre (policy in the module docstring)."""
    out = []
    shares = {t["entity"]: t for t in trends if t["metric"] == "share"}
    for g, t in sorted(shares.items()):
        if (t["recent_mean"] or 0.0) < cfg.genre_min_share:
            continue
        sid = f"share:genre:{g}"
        q = f"What share of home-rail slots should carry {g} films next quarter?"
        args: tuple[str, str, str, str, list[str]] = (
            "editorial_slot_share",
            POLICY_VERSIONS["editorial_slot_share"],
            q,
            "score",
            [],
        )
        w = windows.get(sid)
        reason = None
        if w is None:
            reason = "no share forecast for this genre (history too short)"
        elif w["status"] != "ok":
            reason = w["reason"]
        elif w["coverage"] is None:
            reason = "the window interval's coverage cannot be backtested (too few past windows)"
        elif w["coverage"] < cfg.slot_share_min_coverage:
            reason = (
                f"backtested coverage of the {int(100 * w['nominal'])} % interval is "
                f"{w['coverage']:.2f} < {cfg.slot_share_min_coverage}: the interval is not trustworthy"
            )
        if reason is not None:
            out.append(
                _decision(
                    *args,
                    None,
                    {},
                    None,
                    "interval",
                    {"series_id": sid, "window": w},
                    [],
                    [],
                    "genre",
                    g,
                    as_of_key,
                    reason,
                    scale=SLOT_SCALE,
                )
            )
            continue
        assert w is not None
        answer = _pct(w["mean"])
        interval: list[float | None] = [_pct(w["lo"]), _pct(w["hi"])]
        months = w["months"]
        state = {
            "series_id": sid,
            "forecast_id": w["forecast_id"],
            "forecast_model": w["model"],
            "forecast_model_version": w["model_version"],
            "window_months": months,
            "forecast_mean_share": w["mean"],
            "forecast_interval_share": [w["lo"], w["hi"]],
            "recent_share": t["recent_mean"],
            "trend_direction": t["direction"],
            "interval_nominal": w["nominal"],
            "interval_backtest_coverage": w["coverage"],
            "interval_backtest_origins": w["coverage_origins"],
            "window_residuals": w["n_residuals"],
            "backtest_mase": w["backtest_mase"],
            "backtest_naive_mase": w["backtest_naive_mase"],
            "allocation_rule": "demand-proportional: slot share = forecast share of rating activity",
        }
        rationale = [
            f"{w['model']} forecast of the {g} share of ratings for {months[0][:7]}..{months[-1][:7]}: "
            f"{answer:.1f} % (80 % interval {interval[0]:.1f}–{interval[1]:.1f} %)",
            f"recent {cfg.trend_window_months}-month share {100 * (t['recent_mean'] or 0):.1f} %, "
            f"trend {t['direction']}",
            f"interval from {w['n_residuals']} backtest window residuals; backtested coverage "
            f"{100 * w['coverage']:.0f} % over {w['coverage_origins']} origins (nominal "
            f"{100 * w['nominal']:.0f} %)",
        ]
        out.append(
            _decision(
                *args,
                answer,
                {},
                w["nominal"],
                "interval",
                state,
                rationale,
                [
                    evidence("series", f"{g} share forecast", w["mean"], w["model"], sid),
                    evidence(
                        "test",
                        "window-interval backtest coverage",
                        w["coverage"],
                        f"nominal {w['nominal']}",
                        w["forecast_id"],
                    ),
                ],
                "genre",
                g,
                as_of_key,
                scale=SLOT_SCALE,
                answer_interval=interval,
            )
        )
    return out


# --------------------------------------------------------------------------------------------------
# decision batches (multi-question calls, contract section 9.1)


def build_decision_batches(
    *,
    manifest: dict[str, Any] | None,
    stale: dict[str, Any] | None,
    boot: dict[str, Any] | None,
    data_version: str,
    replayable: bool,
    trends: list[dict[str, Any]],
    forecasts_by_series: dict[str, dict[str, Any]],
    share_windows: dict[str, dict[str, Any]],
    rater_anoms: list[dict[str, Any]],
    influence: dict[str, Any] | None,
    lapse: dict[str, Any],
    lapse_scored: pd.DataFrame | None,
    cfg: IntelConfig,
    as_of_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All decisions of a run, grouped into three batches:

    * ``model_governance``: retrain_model + serving_model (one model/evaluation snapshot),
    * ``genre_programming``: genre_programming + editorial_slot_share for every eligible genre (one
      trends/forecasts snapshot),
    * ``audience``: rater_action + reengagement_campaign (one anomalies/lapse snapshot).
    Returns (decisions in the historical order, batch records)."""
    from jev_ml.core.batches import Question, run_batch

    pv = POLICY_VERSIONS
    model_ent = [("model", str(manifest.get("version")) if manifest else "none")]

    def genre_entities(st: Any) -> list[tuple[str, str]]:
        return sorted(
            ("genre", t["entity"])
            for t in st["share_trends"]
            if (t["recent_mean"] or 0.0) >= cfg.genre_min_share
        )

    def rater_entities(st: Any) -> list[tuple[str, str]]:
        active = [
            a for a in st["rater_anomalies"] if not (a["suppression_reason"] or "").startswith("inactive")
        ]
        ranked = sorted(active, key=lambda a: (-(a["value"] or 0.0), a["entity"]))
        return [("user", str(a["entity"])) for a in ranked[: cfg.rater_decisions_max]]

    gov_state = {
        "manifest": manifest,
        "staleness": stale,
        "bootstrap": boot,
        "data_version": data_version,
        "model_replayable": replayable,
    }
    gov_q = [
        Question(
            "retrain_model",
            pv["retrain_model"],
            "boolean",
            "rule",
            "Should the recommendation model be retrained now?",
            lambda st: [
                retrain_decision(
                    st["manifest"],
                    st["staleness"],
                    st["data_version"],
                    st["model_replayable"],
                    cfg,
                    as_of_key,
                )
            ],
            lambda st: model_ent,
        ),
        Question(
            "serving_model",
            pv["serving_model"],
            "choice",
            "probability",
            "Which evaluated model should serve recommendations?",
            lambda st: [
                serving_decision(st["manifest"], st["bootstrap"], st["model_replayable"], cfg, as_of_key)
            ],
            lambda st: model_ent,
        ),
    ]
    genre_state = {
        "share_trends": [t for t in trends if t["metric"] == "share"],
        "genre_volume_forecasts": {
            k: v for k, v in forecasts_by_series.items() if k.startswith("volume:genre:")
        },
        "share_windows": share_windows,
    }
    genre_q = [
        Question(
            "genre_programming",
            pv["genre_programming"],
            "choice",
            "margin",
            "How should each genre be programmed on the home rails?",
            lambda st: genre_decisions(st["share_trends"], st["genre_volume_forecasts"], cfg, as_of_key),
            genre_entities,
        ),
        Question(
            "editorial_slot_share",
            pv["editorial_slot_share"],
            "score",
            "interval",
            "What share of home-rail slots should carry each genre next quarter?",
            lambda st: slot_share_decisions(st["share_trends"], st["share_windows"], cfg, as_of_key),
            genre_entities,
        ),
    ]
    audience_state = {
        "rater_anomalies": rater_anoms,
        "influence": influence,
        "lapse": lapse,
        "lapse_scored": lapse_scored,
    }
    audience_q = [
        Question(
            "rater_action",
            pv["rater_action"],
            "choice",
            "margin",
            "What should happen to each flagged rater's ratings?",
            lambda st: rater_decisions(st["rater_anomalies"], st["influence"], cfg, as_of_key),
            rater_entities,
        ),
        Question(
            "reengagement_campaign",
            pv["reengagement_campaign"],
            "boolean",
            "probability",
            "Should a re-engagement campaign target users at high lapse risk now?",
            lambda st: [reengagement_decision(st["lapse"], st["lapse_scored"], cfg, as_of_key)],
            lambda st: [("platform", "all")],
        ),
    ]
    decisions: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for name, question, state, qs in (
        (
            "model_governance",
            "Is the recommendation model current, and which model should serve?",
            gov_state,
            gov_q,
        ),
        (
            "genre_programming",
            "How should each genre be programmed on the home rails, and with what slot share next quarter?",
            genre_state,
            genre_q,
        ),
        ("audience", "Which audience interventions are warranted now?", audience_state, audience_q),
    ):
        ds, rec = run_batch(name, question, state, qs, as_of_key)
        decisions.extend(ds)
        batches.append(rec)
    return decisions, batches
