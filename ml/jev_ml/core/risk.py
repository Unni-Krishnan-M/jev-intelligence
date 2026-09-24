"""ASSESS: the risk framework and the domain-independent risk families.

score = 100 * likelihood * impact * confidence * data_quality (0..100), i.e. the expected-loss
product shrunk towards 0 when the evidence is weak or the data is poor.
level: score >= risk_levels.critical -> critical, >= high -> high, >= medium -> medium, else low.

Factors: every multiplicative input is listed with weight 1.0 (multiplicative) and
``contribution`` = the points of score the factor removes (score with the factor at 1 minus the
actual score). Additive platform fields: ``severity`` (= level), ``contributing_factors``
(= factors), ``entity_id``, ``series_id`` (when the risk is about one series; null otherwise) and
``severity_basis`` (score and level bands, read by the early-warning decision).

Generic families (``CoreConfig.generic_risks``), for series that declare an adverse direction:

* ``adverse_trend``: the trend moves in the adverse direction (Mann–Kendall p < alpha, Theil–Sen CI
  excluding 0) and survives the FDR (q <= trend_fdr), and is not reversing (core-1.1.0,
  ``trend.recent_move``). likelihood = 1 - p; confidence = share of window periods with data x (1 - q).
* ``adverse_forecast``: the selected forecast model beats naive in its backtest (skill =
  1 - MASE / naive MASE > 0) and the mean of the next horizon moves >= forecast_risk_min_change
  (relative) against the last horizon, in the adverse direction. likelihood = share of forecast
  steps whose 80 % interval lies entirely on the adverse side of the last observed value;
  confidence = skill.
* ``adverse_anomaly``: an active (not suppressed) series anomaly in the last
  ``warning_anomaly_recent_months`` complete periods, deviating in the adverse direction.
  likelihood = anomaly_score; confidence = baseline periods used / anomaly_baseline_months.

Impact is *declared* when the domain config gives the entity a weight (``impact_weights``, labelled
"declared" in ``factors[].detail``); otherwise it is *measured* as |relative change| /
``risk_relative_change_ref`` (window mean vs prior window, forecast vs last horizon, or deviation vs
baseline), capped at 1. A risk whose impact can be neither declared nor measured is not produced.

``quality_risk`` and ``staleness_risks`` work on any domain's quality checks and sources.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jev_ml.core.anomalies import is_adverse
from jev_ml.core.common import clip01, evidence, fnum, stable_id
from jev_ml.core.config import CoreConfig
from jev_ml.core.quality import quality_evidence
from jev_ml.core.series import Series
from jev_ml.core.trends import is_reversing

# confidence_kind of a risk's ``confidence`` factor (core-1.1.0; decisions.CONFIDENCE_KINDS): "rule"
# for deterministic checks and exact counts, "margin" for backtest skill, "evidence" (a strength of
# support in [0, 1] from a test or from data coverage) otherwise. Never a probability.
RISK_CONFIDENCE_KINDS = {
    "data_quality": "rule",
    "data_staleness": "rule",
    "adverse_trend": "evidence",
    "adverse_forecast": "margin",
    "adverse_anomaly": "evidence",
}


def level_of(score: float, cfg: CoreConfig) -> str:
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
    cfg: CoreConfig,
    as_of_key: str,
    series_id: str | None = None,
    id_part: str | None = None,
    confidence_kind: str | None = None,
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
    level = level_of(score, cfg)
    return {
        "id": stable_id("risk", kind, id_part if id_part is not None else entity, as_of_key),
        "kind": kind,
        "title": title,
        "entity_type": entity_type,
        "entity": entity,
        "likelihood": fnum(lik, 4),
        "impact": fnum(imp, 4),
        "exposure": fnum(lik * imp, 4),
        "confidence": fnum(conf, 4),
        "confidence_kind": confidence_kind or RISK_CONFIDENCE_KINDS.get(kind, "evidence"),
        "data_quality": fnum(qual, 4),
        "score": fnum(score, 2),
        "level": level,
        "factors": factors,
        "evidence": ev,
        "recommended_response": response,
        "severity": level,
        "contributing_factors": [dict(f) for f in factors],
        "entity_id": f"{entity_type}:{entity}",
        "series_id": series_id,
        "severity_basis": {
            "measure": "risk score",
            "value": fnum(score, None),  # unrounded: the level above used this exact value
            # a risk earns MONITOR points from a third of the medium threshold (linear from 0)
            "bands": {"low": cfg.risk_levels["medium"] / 3.0, **cfg.risk_levels},
        },
    }


def quality_risk(quality: dict[str, Any], cfg: CoreConfig, as_of_key: str) -> list[dict[str, Any]]:
    failed = [c for c in quality["checks"] if not c["passed"] and c["severity"] != "info"]
    if not failed:
        return []
    q = quality["score"]
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
            quality_evidence(quality),
            "Inspect the failing checks; invalid rows are already excluded from this run.",
            cfg,
            as_of_key,
        )
    ]


def staleness_risks(sources: list[dict[str, Any]], cfg: CoreConfig, as_of_key: str) -> list[dict[str, Any]]:
    """One risk per source with an update expectation that is not fresh. Each source row carries
    its SLA (``sla_days``) and the days it is behind (``staleness_days``: age for live sources,
    lag behind as_of for datasets)."""
    out = []
    for s in sources:
        if s["expected_update"] is None or s["fresh"] is not False or s.get("sla_days") is None:
            continue  # no SLA, or no events to judge
        sla = float(s["sla_days"])
        age = float(s.get("staleness_days", s["age_days"]))
        lik = 1.0 - float(np.exp(-(age - sla) / sla))
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
                    "likelihood": f"1 - exp(-(age - {sla}) / {sla})",
                    "impact": "declared impact weight",
                    "confidence": "exact timestamps",
                },
                [evidence("record", "last event", s["last_event"], f"expected {s['expected_update']}")],
                s.get("stale_response") or "Check the source's ingestion (API health, background jobs).",
                cfg,
                as_of_key,
            )
        )
    return out


# --------------------------------------------------------------------------------------------------
# generic families


def recent_from(last_complete: str | None, n_periods: int, freq: str = "M") -> str | None:
    """Start date of the first of the last ``n_periods`` complete periods."""
    if not last_complete:
        return None
    if freq == "M":
        return (pd.Period(last_complete[:7], freq="M") - (n_periods - 1)).strftime("%Y-%m-01")
    p = pd.Period(last_complete, freq=freq) - (n_periods - 1)  # W or D
    return p.start_time.strftime("%Y-%m-%d")


def _impact(entity: str, rel: float | None, what: str, cfg: CoreConfig) -> tuple[float, str] | None:
    if entity in cfg.impact_weights:
        return float(cfg.impact_weights[entity]), f"declared impact weight for {entity} (domain config)"
    if rel is None or not np.isfinite(rel):
        return None
    ref = cfg.risk_relative_change_ref
    return clip01(abs(rel) / ref), f"measured: |relative change of {what}| {abs(rel):.4f} / {ref}"


def trend_risks(
    trends: list[dict[str, Any]], series_by_id: dict[str, Series], dq: float, cfg: CoreConfig, as_of_key: str
) -> list[dict[str, Any]]:
    out = []
    for t in trends:
        s = series_by_id.get(t["series_id"])
        if s is None or t["direction"] == "flat" or not is_adverse(t["direction"], s.adverse_direction):
            continue
        if is_reversing(t):  # core-1.1.0: the last periods already reverse the trend (core.trends)
            continue
        q = t["q_value"] if t["q_value"] is not None else 1.0
        if q > cfg.trend_fdr:
            continue
        prior = t["prior_mean"]
        rel = (t["recent_mean"] - prior) / abs(prior) if prior not in (None, 0) else None
        imp = _impact(s.entity, rel, "the window mean vs the prior window", cfg)
        if imp is None:
            continue
        coverage = t["n_points"] / t["window"]["months"]
        word = "rising" if t["direction"] == "up" else "falling"
        out.append(
            make_risk(
                "adverse_trend",
                s.entity_type,
                s.entity,
                f"{s.entity} {s.metric} {word} (adverse)",
                float(t["evidence_strength"] or 0.0),
                imp[0],
                coverage * (1.0 - q),
                dq,
                {
                    "likelihood": "1 - Mann–Kendall p of the adverse trend",
                    "impact": imp[1],
                    "confidence": f"periods with data {coverage:.2f} x (1 - BH q)",
                },
                [
                    evidence(
                        "test", "Mann–Kendall p", t["p_value"], f"tau {t['kendall_tau']}, BH q {q}", t["id"]
                    ),
                    evidence(
                        "metric",
                        f"Theil–Sen slope ({s.unit} per period)",
                        t["slope"],
                        f"95 % CI {t['slope_ci']}; window mean {t['recent_mean']} vs prior {prior}",
                        s.id,
                    ),
                ],
                f"Investigate what drives the adverse {s.metric} trend of {s.entity}.",
                cfg,
                as_of_key,
                series_id=s.id,
                id_part=s.id,
            )
        )
    return out


def forecast_risks(
    forecasts: list[dict[str, Any]],
    states: dict[str, Any],
    series_by_id: dict[str, Series],
    dq: float,
    cfg: CoreConfig,
    as_of_key: str,
) -> list[dict[str, Any]]:
    out = []
    for f in forecasts:
        s = series_by_id.get(f["series_id"])
        st = states.get(f["series_id"])
        if s is None or st is None or s.adverse_direction == "none":
            continue
        bt = f["backtest"]
        if bt.get("mase") is None or not bt.get("naive_mase") or bt["mase"] >= bt["naive_mase"]:
            continue
        h = len(f["points"])
        hist = np.asarray(st.history, dtype=float)
        last_h = hist[-h:]
        last_h = last_h[np.isfinite(last_h)]
        means = [p["mean"] for p in f["points"]]
        if not len(last_h) or any(m is None for m in means):
            continue
        base = float(np.mean(last_h))
        nxt = float(np.mean(means))
        if base == 0:
            continue
        rel = (nxt - base) / abs(base)
        direction = "up" if rel > 0 else "down"
        if abs(rel) < cfg.forecast_risk_min_change or not is_adverse(direction, s.adverse_direction):
            continue
        last_v = float(hist[np.isfinite(hist)][-1])
        beyond = [
            (p["lo80"] is not None and p["lo80"] > last_v)
            if direction == "up"
            else (p["hi80"] is not None and p["hi80"] < last_v)
            for p in f["points"]
        ]
        if all(p["lo80"] is None for p in f["points"]):
            continue
        imp = _impact(s.entity, rel, f"the next {h}-period forecast mean vs the last {h} periods", cfg)
        if imp is None:
            continue
        skill = clip01(1.0 - bt["mase"] / bt["naive_mase"])
        lik = float(np.mean(beyond))
        out.append(
            make_risk(
                "adverse_forecast",
                s.entity_type,
                s.entity,
                f"{s.entity} {s.metric} forecast to move {direction} {abs(rel):.0%} (adverse)",
                lik,
                imp[0],
                skill,
                dq,
                {
                    "likelihood": (
                        f"share of the {h} forecast steps whose 80 % interval lies entirely "
                        f"{'above' if direction == 'up' else 'below'} the last value {last_v:g}"
                    ),
                    "impact": imp[1],
                    "confidence": f"backtest skill 1 - MASE/naive MASE ({bt['mase']} / {bt['naive_mase']})",
                },
                [
                    evidence(
                        "model",
                        f"forecast mean next {h} periods",
                        fnum(nxt, 4),
                        f"{f['model']}; last {h} periods mean {base:.4g}",
                        f["id"],
                    ),
                    evidence("test", "backtest MASE", bt["mase"], f"naive {bt['naive_mase']}", f["id"]),
                ],
                f"Prepare for a continued adverse move of {s.entity} {s.metric}; re-check next period.",
                cfg,
                as_of_key,
                series_id=s.id,
                id_part=s.id,
            )
        )
    return out


def anomaly_risks(
    anomalies: list[dict[str, Any]],
    series_by_id: dict[str, Series],
    last_complete: str | None,
    dq: float,
    cfg: CoreConfig,
    as_of_key: str,
) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    freq = next(iter(series_by_id.values())).freq if series_by_id else "M"
    since = recent_from(last_complete, cfg.warning_anomaly_recent_months, freq)
    for a in anomalies:
        s = series_by_id.get(a.get("series_id") or "")
        if s is None or a["suppressed"] or a["method"] != "robust_z" or a.get("adverse") is not True:
            continue
        if since and a["detected_at"] < since:
            continue
        cur = latest.get(s.id)
        if cur is None or a["detected_at"] > cur["detected_at"]:
            latest[s.id] = a
    out = []
    for sid, a in latest.items():
        s = series_by_id[sid]
        rel = a.get("relative_deviation")
        what_rel = (
            "the period's change vs the previous level"
            if a.get("anomaly_basis") == "change"
            else "the anomalous value vs its baseline median"
        )
        imp = _impact(s.entity, rel, what_rel, cfg)
        if imp is None:
            continue
        n_base = int(a.get("baseline_points") or 0)
        conf = min(1.0, n_base / cfg.anomaly_baseline_months)
        what = "spike" if a["kind"] == "series_spike" else "drop"
        when = a["detected_at"][:7] if s.freq == "M" else a["detected_at"][:10]  # month, or the full date
        out.append(
            make_risk(
                "adverse_anomaly",
                s.entity_type,
                s.entity,
                f"{s.entity} {s.metric} {what} in {when} (adverse)",
                float(a["anomaly_score"] or 0.0),
                imp[0],
                conf,
                dq,
                {
                    "likelihood": "anomaly score min(1, |robust z| / (2 x threshold))",
                    "impact": imp[1],
                    "confidence": f"baseline periods {n_base} / {cfg.anomaly_baseline_months}",
                },
                [*a["evidence"], evidence("record", "anomaly", a["id"], a["kind"], a["id"])],
                f"Check what caused the {what} of {s.entity} {s.metric} before it becomes a trend.",
                cfg,
                as_of_key,
                series_id=s.id,
                id_part=s.id,
            )
        )
    return out
