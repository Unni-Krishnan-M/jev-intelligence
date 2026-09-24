"""DECIDE (early warning): ``early_warning_level``, one decision per situation (contract section 4).

Question: "Should this situation trigger an early warning?" -> NO_ACTION | MONITOR | WARNING |
URGENT_ACTION (``choice``, confidence kind ``margin``). Policy ``ewl-1.1.0`` (1.1: a trend or change
point whose series is already reversing, ``trend.recent_move.reversing``, earns 0 points).

Situations. Evidence computed upstream is grouped by subject:

* ``series:<series_id>``: the series' trend and change point, its anomalies, its forecast (including
  share forecasts) and every risk that names the series (``risk.series_id``);
* ``entity:<entity_type>:<entity>``: anomalies and risks that are not about one series (e.g. a
  rater, the live app, the audience, a model, a data source).

A situation is considered when it has at least one evidence item: any anomaly (also suppressed ones, which
earn no points), a non-flat trend or a change point, a forecast whose next-horizon mean moves at
least ``forecast_risk_min_change`` with backtest skill, or a risk. A series with an adverse
direction whose trend or forecast stage was *skipped* (too little history) also gets a situation, so
the gap is visible as an abstention rather than silence. A decision is emitted for every situation
with at least one point or a skipped stage; situations whose items all earn 0 points are NO_ACTION by
construction and are only listed (``decision_batches[].situations_without_points``).

Points per evidence item (all monotone in the underlying measure; ``CoreConfig.ewl_*``):

* risk and anomaly (the "components"): piecewise-linear in the measure their severity came from
  (``severity_basis``: risk score, |robust z|, IsolationForest score, feedback lift): 0 at 0, then
  ``ewl_severity_points`` = 1 / 3 / 5 / 7 at the lower edge of the low / medium / high / critical
  band, extrapolated past critical and capped at ``ewl_max_points``;
* context caps: an anomaly outside the last ``warning_anomaly_recent_months`` complete periods, or
  (with ``ewl_anomaly_direction = "adverse"``) deviating in the favourable direction, is capped at
  the upper context bound (2.5 points);
* trend (adverse direction, not flat): lo + (hi - lo) x (1 - q); adverse change point:
  lo + (hi - lo) x (1 - p); adverse forecast: lo + (hi - lo) x min(1, |relative change|) x skill,
  with (lo, hi) = ``ewl_context_points`` = (1, 2.5). Favourable or undirected moves earn 0;
* suppressed items (min-volume guard, inactive rater, operator dismissal) earn 0 and are listed.

Aggregation: P = min(max_points, max(points) + corroboration x (number of independent detection
stages among trend / anomaly / forecast with >= 1 point, minus one)). Risks are derived from those
stages, so they add to the maximum but never count as corroboration. Adding evidence or making it
more adverse can only raise P: max and the stage count are monotone, and so is every points map.

Level: the highest level whose threshold P reaches (``ewl_level_points``: MONITOR 1, WARNING 3,
URGENT_ACTION 5). Option scores: with e_L = clip((P - T_L + r) / 2r) for the three escalated levels
(r = ``ewl_ramp``), s(NO_ACTION) = 1 - e_MONITOR, s(L) = e_L - e_next, s(URGENT_ACTION) = e_URGENT;
they sum to 1. Confidence = (s_answer - s_other_max) / (s_answer + s_other_max): 0 on a threshold,
1 in the middle of a band. It measures distance from a boundary, not a probability.

Abstention: when every evidence item of a situation is a skipped stage, the answer is null with
the skipped stages as ``fallback_reason``.

A severity reached by a component maps to the level it supports (medium -> WARNING, high and
critical -> URGENT_ACTION), so the component rule "risk level / anomaly severity >= medium raises a
warning" is exactly "the situation's level is at least WARNING because of this component". The
movie domain (corroboration 0) therefore reproduces the v1.1 warning set; see ``core.warnings``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from jev_ml.core.anomalies import is_adverse
from jev_ml.core.common import clip01, evidence, fnum, stable_id
from jev_ml.core.config import EWL_LEVELS, EWL_POLICY_VERSION, SEVERITIES, CoreConfig, severity_rank
from jev_ml.core.decisions import DecisionSpec, margin
from jev_ml.core.risk import recent_from
from jev_ml.core.series import Series
from jev_ml.core.trends import is_reversing

KEY = "early_warning_level"
SPEC = DecisionSpec(
    KEY,
    EWL_POLICY_VERSION,
    "choice",
    "margin",
    "Should this situation trigger an early warning?",
    EWL_LEVELS,
)
STAGES = ("trend", "anomaly", "forecast")  # independent detection stages (corroboration)


@dataclass
class Situation:
    key: str
    entity_type: str
    entity: str
    series_id: str | None = None
    adverse_direction: str = "none"
    label: str = ""
    components: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)


def severity_points(value: float, bands: dict[str, float], cfg: CoreConfig) -> float:
    """Piecewise-linear, non-decreasing map of a severity measure onto points (module docstring)."""
    xs = [0.0] + [float(bands[s]) for s in SEVERITIES]
    ys = [0.0] + [float(cfg.ewl_severity_points[s]) for s in SEVERITIES]
    if not all(b > a for a, b in itertools.pairwise(xs)):
        # degenerate bands (a low edge at 0): drop the origin anchor, keep monotone order
        xs, ys = xs[1:], ys[1:]
    v = float(value)
    if v <= xs[-1]:
        return float(np.interp(v, xs, ys))
    slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return float(min(cfg.ewl_max_points, ys[-1] + slope * (v - xs[-1])))


def context_points(strength: float, cfg: CoreConfig) -> float:
    lo, hi = cfg.ewl_context_points
    return float(lo + (hi - lo) * clip01(strength))


def level_of_points(p: float, cfg: CoreConfig) -> str:
    level = "NO_ACTION"
    for name in EWL_LEVELS[1:]:
        if p >= cfg.ewl_level_points[name]:
            level = name
    return level


def option_scores(p: float, cfg: CoreConfig) -> dict[str, float]:
    r = max(cfg.ewl_ramp, 1e-9)
    e = [clip01((p - cfg.ewl_level_points[name] + r) / (2 * r)) for name in EWL_LEVELS[1:]]
    for i in range(1, len(e)):  # keep the cumulative ramps nested when thresholds are close
        e[i] = min(e[i], e[i - 1])
    return {
        "NO_ACTION": 1.0 - e[0],
        "MONITOR": e[0] - e[1],
        "WARNING": e[1] - e[2],
        "URGENT_ACTION": e[2],
    }


def aggregate(components: list[dict[str, Any]], cfg: CoreConfig) -> tuple[float, float, list[str]]:
    """(P, max component points, corroborating stages)."""
    pts = [float(c["points"]) for c in components]
    top = max(pts, default=0.0)
    monitor = cfg.ewl_level_points["MONITOR"]
    stages = sorted(
        {c["stage_group"] for c in components if c["stage_group"] in STAGES and c["points"] >= monitor}
    )
    p = top + cfg.ewl_corroboration * max(0, len(stages) - 1)
    return float(min(cfg.ewl_max_points, p)), top, stages


# --------------------------------------------------------------------------------------------------
# evidence -> components


def _comp(
    stage: str, group: str, ref: str | None, points: float, detail: str, title: str, **extra: Any
) -> dict[str, Any]:
    return {
        "stage": stage,
        "stage_group": group,
        "ref": ref,
        "points": float(points),
        "detail": detail,
        "title": title,
        **extra,
    }


def _risk_component(
    r: dict[str, Any], suppressed_reason: str | None, min_rank: int, cfg: CoreConfig
) -> dict[str, Any]:
    basis = r.get("severity_basis") or {}
    raw = severity_points(float(basis.get("value") or 0.0), basis["bands"], cfg) if basis else 0.0
    pts = 0.0 if suppressed_reason else raw
    return _comp(
        "risk",
        "risk",
        r["id"],
        pts,
        f"risk {r['kind']} score {r['score']} ({r['level']})"
        + (f"; suppressed: {suppressed_reason}" if suppressed_reason else ""),
        r["title"],
        key=f"risk:{r['kind']}:{r['entity']}",
        risk_score=r["score"],
        severity=r["level"],
        suppressed=bool(suppressed_reason),
        warnable=(not suppressed_reason) and severity_rank(r["level"]) >= min_rank,
        raw_points=raw,
    )


def _anomaly_component(
    a: dict[str, Any], recent: bool, min_rank: int, cfg: CoreConfig, adverse_mode: bool
) -> dict[str, Any]:
    basis = a.get("severity_basis") or {}
    raw = severity_points(float(basis.get("value") or 0.0), basis["bands"], cfg) if basis else 0.0
    notes = []
    pts = raw
    cap = cfg.ewl_context_points[1]
    if a["suppressed"]:
        pts = 0.0
        notes.append(f"suppressed: {a['suppression_reason']}")
    else:
        if not recent:
            pts = min(pts, cap)
            notes.append(f"outside the last {cfg.warning_anomaly_recent_months} complete periods (context)")
        if adverse_mode and a.get("adverse") is False:
            pts = min(pts, cap)
            notes.append("favourable direction (context)")
    detail = f"{a['kind']} {a.get('series_id') or a['entity']} ({a['severity']}, score {a['score']})"
    if notes:
        detail += "; " + "; ".join(notes)
    return _comp(
        "anomaly",
        "anomaly",
        a["id"],
        pts,
        detail,
        f"{a['kind'].replace('_', ' ')}: {a.get('series_id') or a['entity']}",
        key=a["dedup_key"],
        anomaly_score=a.get("anomaly_score"),
        severity=a["severity"],
        suppressed=bool(a["suppressed"]),
        warnable=(not a["suppressed"]) and recent and severity_rank(a["severity"]) >= min_rank,
        raw_points=raw,
    )


def build_situations(
    series_by_id: dict[str, Series],
    trends: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    forecasts: list[dict[str, Any]],
    states: dict[str, Any],
    risks: list[dict[str, Any]],
    suppressed: dict[str, str],
    last_complete: str | None,
    cfg: CoreConfig,
) -> list[Situation]:
    from jev_ml.core.anomalies import suppression

    sits: dict[str, Situation] = {}
    min_rank = severity_rank(cfg.warning_min_severity)
    freq = next(iter(series_by_id.values())).freq if series_by_id else "M"
    since = recent_from(last_complete, cfg.warning_anomaly_recent_months, freq)
    adverse_mode = cfg.ewl_anomaly_direction == "adverse"

    def for_series(sid: str) -> Situation:
        key = f"series:{sid}"
        if key not in sits:
            s = series_by_id[sid]
            sits[key] = Situation(
                key,
                s.entity_type,
                s.entity,
                sid,
                s.adverse_direction,
                f"{s.entity} {s.metric}",
            )
        return sits[key]

    def for_entity(entity_type: str, entity: str) -> Situation:
        key = f"entity:{entity_type}:{entity}"
        if key not in sits:
            sits[key] = Situation(key, entity_type, entity, None, "none", f"{entity_type} {entity}")
        return sits[key]

    trend_by = {t["series_id"]: t for t in trends}
    for t in trends:
        s = series_by_id.get(t["series_id"])
        if s is None:
            continue
        cp = t.get("change_point")
        # ewl-1.1.0: a trend whose last periods already reverse it is not a live adverse condition
        reversing = is_reversing(t)
        rev_note = (
            f"; reversing: last {t['recent_move']['periods']} periods moved {t['recent_move']['change']:+g} "
            f"(z {t['recent_move']['z']:+.2f}) against it"
            if reversing
            else ""
        )
        if t["direction"] != "flat":
            adv = is_adverse(t["direction"], s.adverse_direction)
            q = t["q_value"] if t["q_value"] is not None else 1.0
            pts = context_points(1.0 - q, cfg) if adv and not reversing else 0.0
            for_series(s.id).components.append(
                _comp(
                    "trend",
                    "trend",
                    t["id"],
                    pts,
                    f"trend {t['direction']} (p {t['p_value']}, q {q}); "
                    + (
                        "adverse"
                        if adv
                        else "not adverse"
                        if adv is False
                        else "no adverse direction declared"
                    )
                    + rev_note,
                    f"{s.entity} {s.metric} trend {t['direction']}",
                    direction=t["direction"],
                    q_value=q,
                    reversing=reversing,
                )
            )
        if cp:
            up = (cp["after_mean"] or 0) > (cp["before_mean"] or 0)
            adv = is_adverse("up" if up else "down", s.adverse_direction)
            cp_rev = reversing and (t["direction"] == ("up" if up else "down"))
            pts = context_points(1.0 - (cp["p_value"] or 1.0), cfg) if adv and not cp_rev else 0.0
            for_series(s.id).components.append(
                _comp(
                    "change_point",
                    "trend",
                    t["id"],
                    pts,
                    f"mean shift {'up' if up else 'down'} in {cp['date'][:7]} (p {cp['p_value']})"
                    + (rev_note if cp_rev else ""),
                    f"{s.entity} {s.metric} shifted {'up' if up else 'down'}",
                )
            )
    for a in anomalies:
        sid = a.get("series_id")
        recent = True
        if a["method"] == "robust_z" and since and a["detected_at"] < since:
            recent = False
        comp = _anomaly_component(a, recent, min_rank, cfg, adverse_mode)
        sit = for_series(sid) if sid in series_by_id else for_entity(a["entity_type"], a["entity"])
        sit.components.append(comp)
    for f in forecasts:
        s = series_by_id.get(f["series_id"])
        st = states.get(f["series_id"])
        if s is None or st is None:
            continue
        bt = f["backtest"]
        h = len(f["points"])
        hist = np.asarray(st.history, dtype=float)
        last_h = hist[-h:]
        last_h = last_h[np.isfinite(last_h)]
        means = [p["mean"] for p in f["points"]]
        if not len(last_h) or any(m is None for m in means) or float(np.mean(last_h)) == 0:
            continue
        base = float(np.mean(last_h))
        rel = (float(np.mean(means)) - base) / abs(base)
        if bt.get("mase") is None or not bt.get("naive_mase") or bt["mase"] >= bt["naive_mase"]:
            continue
        if abs(rel) < cfg.forecast_risk_min_change:
            continue
        skill = clip01(1.0 - bt["mase"] / bt["naive_mase"])
        direction = "up" if rel > 0 else "down"
        adv = is_adverse(direction, s.adverse_direction)
        pts = context_points(min(1.0, abs(rel)) * skill, cfg) if adv else 0.0
        for_series(s.id).components.append(
            _comp(
                "forecast",
                "forecast",
                f["id"],
                pts,
                f"forecast {direction} {abs(rel):.1%} over {h} periods, skill {skill:.2f}; "
                + ("adverse" if adv else "not adverse" if adv is False else "no adverse direction declared"),
                f"{s.entity} {s.metric} forecast {direction} {abs(rel):.0%}",
                direction=direction,
                relative_change=rel,
            )
        )
    fc_ids = {f["series_id"] for f in forecasts}
    for r in risks:
        reason = suppression(f"risk:{r['kind']}:{r['entity']}", r["level"], suppressed)
        sid = r.get("series_id")
        sit = for_series(sid) if sid in series_by_id else for_entity(r["entity_type"], r["entity"])
        sit.components.append(_risk_component(r, reason, min_rank, cfg))
    # monitored series whose trend/forecast stage was skipped
    for s in series_by_id.values():
        if s.adverse_direction == "none":
            continue
        skipped = []
        if s.trend and s.id not in trend_by:
            skipped.append(
                {"stage": "trend", "reason": f"fewer than {cfg.trend_min_points} complete periods"}
            )
        if s.forecast and s.id not in fc_ids:
            skipped.append(
                {"stage": "forecast", "reason": f"fewer than {cfg.forecast_min_history + 2} complete periods"}
            )
        if skipped:
            for_series(s.id).skipped.extend(skipped)
    return list(sits.values())


# --------------------------------------------------------------------------------------------------
# decisions


def _public(c: dict[str, Any]) -> dict[str, Any]:
    out = dict(c)
    out["points"] = fnum(c["points"], 4)
    if "raw_points" in out:
        out["raw_points"] = fnum(c["raw_points"], 4)
    return out


def situation_state(sit: Situation) -> dict[str, Any]:
    """Plain-data form of a situation (goes into the batch's hashed state snapshot)."""
    return {
        "key": sit.key,
        "entity_type": sit.entity_type,
        "entity": sit.entity,
        "series_id": sit.series_id,
        "adverse_direction": sit.adverse_direction,
        "label": sit.label,
        "components": sit.components,
        "skipped": sit.skipped,
    }


def decide(sit_d: dict[str, Any], cfg: CoreConfig, as_of_key: str) -> dict[str, Any]:
    """The ewl-1.1.0 policy for one situation (``situation_state`` form)."""
    sit = Situation(**sit_d)
    policy = {
        "level_points": dict(cfg.ewl_level_points),
        "severity_points": dict(cfg.ewl_severity_points),
        "context_points": list(cfg.ewl_context_points),
        "corroboration": cfg.ewl_corroboration,
        "ramp": cfg.ewl_ramp,
        "anomaly_direction": cfg.ewl_anomaly_direction,
    }
    comps = sit.components
    strengths = [
        float(x)
        for c in comps
        for x in [
            c.get("anomaly_score") if c["stage"] == "anomaly" and not c["suppressed"] else None,
            (1 - c["q_value"]) if c["stage"] == "trend" else None,
        ]
        if x is not None
    ]
    trend = next((c for c in comps if c["stage"] == "trend"), None)
    anoms = [c for c in comps if c["stage"] == "anomaly"]
    top_anom = max(anoms, key=lambda c: c.get("raw_points", c["points"]), default=None)
    fc = next((c for c in comps if c["stage"] == "forecast"), None)
    risk_c = [c for c in comps if c["stage"] == "risk"]
    top_risk = max(risk_c, key=lambda c: c.get("raw_points", c["points"]), default=None)
    state: dict[str, Any] = {
        "situation": sit.key,
        "series_id": sit.series_id,
        "adverse_direction": sit.adverse_direction,
        "signal": {"strength": fnum(max(strengths), 4) if strengths else None, "n_evidence": len(comps)},
        "trend": {
            "direction": trend["direction"],
            "q": fnum(trend["q_value"], 6),
            "points": fnum(trend["points"], 4),
        }
        if trend
        else None,
        "anomaly": {
            "score": top_anom.get("anomaly_score"),
            "severity": top_anom["severity"],
            "suppressed": top_anom["suppressed"],
            "points": fnum(top_anom["points"], 4),
        }
        if top_anom
        else None,
        "forecast": {
            "direction": fc["direction"],
            "relative_change": fnum(fc["relative_change"], 4),
            "adverse_direction": sit.adverse_direction,
            "points": fnum(fc["points"], 4),
        }
        if fc
        else None,
        "risk": {
            "score": top_risk.get("risk_score"),
            "level": top_risk["severity"],
            "points": fnum(top_risk["points"], 4),
        }
        if top_risk
        else None,
        "components": [_public(c) for c in comps],
        "skipped_stages": sit.skipped,
        "policy": policy,
    }
    q = f"Should {sit.label} trigger an early warning?"
    if not comps:
        reason = "only evidence is skipped stages: " + ", ".join(
            f"{s['stage']} ({s['reason']})" for s in sit.skipped
        )
        d = SPEC.abstain(reason, state, sit.entity_type, sit.entity, as_of_key, question=q)
    else:
        p, top, stages = aggregate(comps, cfg)
        level = level_of_points(p, cfg)
        scores = option_scores(p, cfg)
        conf = margin(scores, level)
        state.update(
            {"points": fnum(p, 4), "points_max_component": fnum(top, 4), "corroborating_stages": stages}
        )
        ranked = sorted(comps, key=lambda c: -c["points"])
        rationale = [
            f"{level}: {p:.2f} points (thresholds MONITOR {cfg.ewl_level_points['MONITOR']:g}, WARNING "
            f"{cfg.ewl_level_points['WARNING']:g}, URGENT_ACTION {cfg.ewl_level_points['URGENT_ACTION']:g})",
            *[f"{c['stage']}: {c['points']:.2f} points ({c['detail']})" for c in ranked[:4]],
        ]
        if len(stages) > 1 and cfg.ewl_corroboration:
            rationale.append(f"+{cfg.ewl_corroboration * (len(stages) - 1):g} for agreeing stages {stages}")
        if sit.skipped:
            rationale.append("skipped: " + ", ".join(s["stage"] for s in sit.skipped))
        ev = [
            evidence(
                "record" if c["stage"] in ("risk", "anomaly") else "test",
                c["title"],
                fnum(c["points"], 3),
                c["detail"],
                c["ref"],
            )
            for c in ranked[:5]
        ]
        d = SPEC.decide(
            level, scores, conf, state, rationale, ev, sit.entity_type, sit.entity, as_of_key, question=q
        )
    d["id"] = stable_id("dec", KEY, sit.key, as_of_key)
    d["situation"] = sit.key
    d["series_id"] = sit.series_id
    return d


def build_early_warning(
    situations: list[Situation], cfg: CoreConfig, as_of_key: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One ``early_warning_level`` decision per situation (sorted by situation key), answered in
    the ``early_warning`` batch against one hashed snapshot of every situation's evidence.
    Returns (decisions, batch record)."""
    from jev_ml.core.batches import Question, run_batch

    ordered = sorted(situations, key=lambda s: s.key)
    # a situation whose every item earns 0 points (suppressed anomalies, favourable or undirected
    # moves) is NO_ACTION by construction; it is listed in the batch record, not emitted
    live = [s for s in ordered if s.skipped or any(c["points"] > 0 for c in s.components)]
    silent = [s.key for s in ordered if s not in live]
    state = {"situations": [situation_state(s) for s in live]}
    q = Question(
        KEY,
        EWL_POLICY_VERSION,
        "choice",
        "margin",
        SPEC.question,
        lambda st: [decide(sd, cfg, as_of_key) for sd in st["situations"]],
        lambda st: [(sd["entity_type"], sd["key"]) for sd in st["situations"]],
    )
    decisions, batch = run_batch(
        "early_warning", "Which situations should trigger an early warning now?", state, [q], as_of_key
    )
    for d in decisions:  # a failed batch records abstentions keyed by entity; re-key by situation
        if d.get("situation") is None and d["abstained"]:
            d["situation"] = d["entity"]
    batch["situations_without_points"] = silent
    return decisions, batch
