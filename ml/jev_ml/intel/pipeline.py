"""The single entry point: ``run_pipeline(inputs, config) -> PipelineResult``.

INGEST -> VALIDATE -> UNDERSTAND -> DETECT -> PREDICT -> ASSESS -> DECIDE -> RECOMMEND/ACT -> EXPLAIN.
Every stage is timed (``run.stage_ms``). The result is deterministic for the same inputs (all
randomness is seeded per object), and ``to_dict()`` is strict-JSON serialisable (no NaN, no numpy).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TypeVar

from jev_ml.intel import anomalies as anom
from jev_ml.intel import decisions as dec
from jev_ml.intel import risk as rk
from jev_ml.intel.actions import build_actions
from jev_ml.intel.common import clean, iso
from jev_ml.intel.config import PIPELINE_VERSION, IntelConfig, severity_rank
from jev_ml.intel.forecast import ForecastState, forecast_all, forecast_shares
from jev_ml.intel.ingest import PipelineInputs, Prepared, prepare
from jev_ml.intel.lapse import run_lapse
from jev_ml.intel.modelstats import bootstrap_models
from jev_ml.intel.series import Series, build_series
from jev_ml.intel.signals import build_signals, series_last_totals
from jev_ml.intel.trends import analyse_trends
from jev_ml.intel.warnings import build_warnings

T = TypeVar("T")


@dataclass
class PipelineResult:
    data: dict[str, Any]
    series_objects: list[Series] = field(default_factory=list)
    forecast_states: dict[str, ForecastState] = field(default_factory=dict)
    prepared: Prepared | None = None
    config: IntelConfig = field(default_factory=IntelConfig)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = clean(self.data)
        return out

    @property
    def summary(self) -> dict[str, Any]:
        s: dict[str, Any] = self.data["summary"]
        return s


class _Timer:
    def __init__(self) -> None:
        self.ms: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.ms[name] = round(self.ms.get(name, 0.0) + 1000 * (time.perf_counter() - t0), 2)

    def run(self, name: str, fn: Callable[[], T]) -> T:
        with self.stage(name):
            return fn()


def _summary(
    signals: list[dict[str, Any]],
    trends: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    risks: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    worst = max((severity_rank(w["severity"]) for w in warnings), default=-1)
    status = "alert" if worst >= severity_rank("high") else "watch" if warnings else "nominal"
    counts = {
        "signals": len(signals),
        "trends_up": sum(t["direction"] == "up" for t in trends),
        "trends_down": sum(t["direction"] == "down" for t in trends),
        "anomalies": sum(not a["suppressed"] for a in anomalies),
        "anomalies_suppressed": sum(a["suppressed"] for a in anomalies),
        "risks_high": sum(severity_rank(r["level"]) >= severity_rank("high") for r in risks),
        "warnings": len(warnings),
        "decisions": len(decisions),
        "decisions_abstained": sum(d["abstained"] for d in decisions),
        "actions": len(actions),
    }
    if warnings:
        by = {s: sum(w["severity"] == s for w in warnings) for s in ("critical", "high", "medium", "low")}
        parts = ", ".join(f"{n} {s}" for s, n in by.items() if n)
        headline = (
            f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''} ({parts}): {warnings[0]['title']}"
        )
        if len(warnings) > 1:
            headline += f"; {warnings[1]['title']}"
    else:
        headline = (
            f"No warnings. {counts['trends_up']} rising / {counts['trends_down']} falling trends, "
            f"{counts['anomalies']} active anomalies."
        )
    return {"status": status, "headline": headline, "counts": counts}


def run_pipeline(inputs: PipelineInputs, config: IntelConfig | None = None) -> PipelineResult:
    cfg = config or IntelConfig()
    tm = _Timer()
    t_start = time.perf_counter()
    # inputs arrive as frames (reading files is load_default_inputs' job), so ingest = cleaning,
    # as_of filtering and validation, timed as one stage
    prep = tm.run("ingest_validate", lambda: prepare(inputs, cfg))
    key = iso(prep.as_of) or ""
    sup = inputs.suppressed_keys or {}
    dq = float(prep.quality["score"])

    series = tm.run("series", lambda: build_series(prep, cfg))
    trends = tm.run("trends", lambda: analyse_trends(series, cfg, key))
    with tm.stage("anomalies"):
        series_anoms = anom.scan_all_series(series, cfg, key, sup)
        rater_anoms, rater_feats = anom.detect_raters(prep, cfg, key, sup)
        live_anoms, live = anom.live_feedback(prep, cfg, key, sup)
        if live.get("status") != "ok":
            prep.diagnostics.append(
                {"stage": "live_feedback", "status": "skipped", "reason": live.get("reason")}
            )
    with tm.stage("forecast"):
        forecasts, states = forecast_all(series, cfg, key, prep.data_version)
        share_forecasts, share_states = forecast_shares(series, cfg, key, prep.data_version)
        states = {**states, **share_states}
    lapse, scored = tm.run("lapse", lambda: run_lapse(prep.ratings, prep.as_of_ts, cfg, prep.data_version))
    if lapse["status"] != "ok":
        prep.diagnostics.append({"stage": "lapse", "status": "insufficient_data", "reason": lapse["detail"]})

    manifest = inputs.model_manifest
    with tm.stage("risk"):
        boot = bootstrap_models(inputs.per_user_ndcg, cfg) if prep.model_replayable else None
        fc_by = {f["series_id"]: f for f in forecasts}
        risks = rk.genre_risks(trends, fc_by, dq, cfg, key)
        risks += rk.lapse_risk(lapse, scored, prep, dq, cfg, key)
        manip, infl = rk.manipulation_risk(rater_anoms, rater_feats, prep, dq, cfg, key)
        risks += manip
        risks += rk.quality_risk(prep, cfg, key)
        stale_risks = rk.staleness_risks(prep, cfg, key)
        risks += stale_risks
        mrisks, stale = rk.model_risks(prep, manifest, boot, dq, cfg, key)
        risks += mrisks
        risks += rk.rejection_risk(live, dq, cfg, key)
        risks.sort(key=lambda r: (-(r["score"] or 0), r["id"]))
    with tm.stage("decide"):
        decisions, decision_batches = dec.build_decision_batches(
            manifest=manifest,
            stale=stale,
            boot=boot,
            data_version=prep.data_version,
            replayable=prep.model_replayable,
            trends=trends,
            forecasts_by_series=fc_by,
            share_windows=dec.share_window_inputs(share_forecasts, share_states, cfg),
            rater_anoms=rater_anoms,
            influence=infl,
            lapse=lapse,
            lapse_scored=scored,
            cfg=cfg,
            as_of_key=key,
        )
    all_anoms = series_anoms + rater_anoms[: cfg.top_n] + live_anoms
    last_complete = None
    if series:
        m, _, _ = series[0].complete()
        last_complete = m[-1].strftime("%Y-%m-01") if len(m) else None
    with tm.stage("recommend"):
        warnings = build_warnings(risks, all_anoms, sup, last_complete, cfg)
        actions = build_actions(decisions, warnings, risks, manifest, key)
    with tm.stage("explain"):
        signals = build_signals(
            trends,
            all_anoms,
            forecasts,
            series_last_totals(states, cfg.forecast_horizon),
            prep,
            live,
            stale,
            boot,
            stale_risks,
            cfg,
            key,
        )
        summary = _summary(signals, trends, all_anoms, risks, warnings, decisions, actions)
        series_out = [s.to_dict(cfg.series_output_months) for s in series]
    tm.ms["total"] = round(1000 * (time.perf_counter() - t_start), 2)
    data = {
        "run": {
            "pipeline_version": PIPELINE_VERSION,
            "as_of": key,
            "now": iso(prep.now),
            "data_version": prep.data_version,
            "model_version": prep.model_version,
            "stage_ms": tm.ms,
            "config": cfg.to_dict(),
            "last_complete_month": last_complete,
        },
        "data": {
            "sources": prep.sources,
            "quality": prep.quality,
            "excluded_after_as_of": prep.excluded_after_as_of,
            "live": {k: v for k, v in live.items()},
        },
        "series": series_out,
        "signals": signals,
        "trends": trends,
        "anomalies": all_anoms,
        "predictions": {"forecasts": forecasts, "share_forecasts": share_forecasts, "lapse": lapse},
        "risks": risks,
        "decisions": decisions,
        "decision_batches": decision_batches,
        "warnings": warnings,
        "actions": actions,
        "summary": summary,
        "diagnostics": prep.diagnostics,
    }
    return PipelineResult(data=data, series_objects=series, forecast_states=states, prepared=prep, config=cfg)
