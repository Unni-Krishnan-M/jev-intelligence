"""The generic entry point: ``run_domain(adapter, as_of, now, suppressed_keys, config) -> PipelineResult``.

INGEST/VALIDATE (adapter.load + core checks) -> UNDERSTAND (series) -> DETECT (trends, change
points, series anomalies) -> PREDICT (forecasts) -> domain extras (adapter hook) -> ASSESS (risks)
-> DECIDE (domain decisions + early-warning decisions) -> RECOMMEND/ACT (warnings downstream of the
early-warning decisions, actions) -> EXPLAIN (signals, summary).

Every stage is timed (``run.stage_ms``). The result is deterministic for the same inputs (all
randomness is seeded per object), ``to_dict()`` is strict-JSON serialisable (no NaN, no numpy) and
every emitted object carries ``"domain": <adapter key>``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, TypeVar

from jev_ml.core import risk as rk
from jev_ml.core.actions import build_actions
from jev_ml.core.adapter import CoreContext, DomainAdapter
from jev_ml.core.anomalies import scan_all_series
from jev_ml.core.common import clean, iso, short_hash
from jev_ml.core.config import CORE_VERSION, EWL_LEVELS, CoreConfig, severity_rank
from jev_ml.core.early_warning import KEY as EWL_KEY
from jev_ml.core.early_warning import build_early_warning, build_situations
from jev_ml.core.forecast import ForecastState, forecast_all, forecast_shares
from jev_ml.core.quality import Checks, validate_observations
from jev_ml.core.series import Series, build_series, last_complete_period
from jev_ml.core.signals import build_signals, series_last_totals
from jev_ml.core.trends import analyse_trends
from jev_ml.core.warnings import build_warnings

T = TypeVar("T")
OBJECT_SECTIONS = (
    "signals",
    "trends",
    "anomalies",
    "risks",
    "decisions",
    "decision_batches",
    "warnings",
    "actions",
)


@dataclass
class PipelineResult:
    data: dict[str, Any]
    series_objects: list[Series] = field(default_factory=list)
    forecast_states: dict[str, ForecastState] = field(default_factory=dict)
    prepared: Any = None  # the adapter's private context (movie: the validated ``Prepared`` frames)
    config: CoreConfig = field(default_factory=CoreConfig)
    domain: str = "movie"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = clean(self.data)
        return out

    @property
    def summary(self) -> dict[str, Any]:
        s: dict[str, Any] = self.data["summary"]
        return s


class Timer:
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


def summarise(
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
    ewl = [d for d in decisions if d["key"] == EWL_KEY]
    counts: dict[str, Any] = {
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
        "early_warning": {lv: sum(d["answer"] == lv for d in ewl) for lv in EWL_LEVELS}
        | {"abstained": sum(d["abstained"] for d in ewl)},
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


def run_domain(
    adapter: DomainAdapter,
    as_of: datetime | str | None = None,
    now: datetime | None = None,
    suppressed_keys: dict[str, str] | None = None,
    config: CoreConfig | None = None,
) -> PipelineResult:
    """Run the core pipeline on one domain adapter (see module docstring)."""
    from jev_ml.core.common import to_utc

    cfg = config or adapter.default_config()
    tm = Timer()
    t_start = time.perf_counter()
    sup = dict(suppressed_keys or {})
    with tm.stage("ingest_validate"):
        data = adapter.load(to_utc(as_of) if as_of is not None else None, now, cfg)
        checks = Checks(cfg.check_weights)
        checks.extend(list(data.quality_checks))
        obs = data.observations
        excluded = data.excluded_after_as_of
        as_of_ts, now_ts = data.as_of.timestamp(), data.now.timestamp()
        if data.core_checks:
            src = data.sources[0]["source"] if data.sources else adapter.key
            obs, n_after = validate_observations(obs, src, as_of_ts, now_ts, checks, data.earliest_ts)
            excluded += n_after
        checks.extend(adapter.quality_checks(replace(data, observations=obs)))
        quality: dict[str, Any] = {"score": round(checks.score(), 4), "checks": checks.items}
        if not len(obs):
            raise ValueError(f"{adapter.key}: no valid observations at or before as_of={iso(data.as_of)}")
    key = iso(data.as_of) or ""
    dq = float(quality["score"])
    diagnostics: list[dict[str, Any]] = list(data.diagnostics)

    series = tm.run(
        "series",
        lambda: build_series(obs, data.specs, data.as_of, data.frequency, data.grid_end, adapter.key),
    )
    by_id = {s.id: s for s in series}
    trends = tm.run("trends", lambda: analyse_trends(series, cfg, key))
    with tm.stage("anomalies"):
        series_anoms = scan_all_series(series, cfg, key, sup)
    with tm.stage("forecast"):
        forecasts, states = forecast_all(series, cfg, key, data.data_version)
        share_forecasts, share_states = forecast_shares(
            series, cfg, key, data.data_version, data.share_forecast_min_share
        )
        states = {**states, **share_states}
    last_complete = last_complete_period(series)
    ctx = CoreContext(
        domain=adapter.key,
        config=cfg,
        data=replace(data, observations=obs),
        as_of_key=key,
        as_of_ts=as_of_ts,
        now_ts=now_ts,
        series=series,
        series_by_id=by_id,
        trends=trends,
        anomalies=series_anoms,
        forecasts=forecasts,
        share_forecasts=share_forecasts,
        forecast_states=states,
        quality=quality,
        suppressed=sup,
        last_complete=last_complete,
        timer=tm,
        diagnostics=diagnostics,
    )
    extras = adapter.extra(ctx)  # noqa: S610 (the adapter hook, not Django)
    diagnostics.extend(extras.diagnostics)
    all_anoms = series_anoms + list(extras.anomalies)
    with tm.stage("risk"):
        risks = list(extras.risks)
        risks += rk.quality_risk(quality, cfg, key)
        stale_risks = rk.staleness_risks(data.sources, cfg, key)
        risks += stale_risks
        if "adverse_trend" in cfg.generic_risks:
            risks += rk.trend_risks(trends, by_id, dq, cfg, key)
        if "adverse_forecast" in cfg.generic_risks:
            risks += rk.forecast_risks(forecasts + share_forecasts, states, by_id, dq, cfg, key)
        if "adverse_anomaly" in cfg.generic_risks:
            risks += rk.anomaly_risks(series_anoms, by_id, last_complete, dq, cfg, key)
        risks.sort(key=lambda r: (-(r["score"] or 0), r["id"]))
    with tm.stage("decide"):
        decisions = list(extras.decisions)
        batches = list(extras.decision_batches)
        situations = build_situations(
            by_id, trends, all_anoms, forecasts + share_forecasts, states, risks, sup, last_complete, cfg
        )
        ewl, ewl_batch = build_early_warning(situations, cfg, key)
        decisions += ewl
        batches.append(ewl_batch)
    with tm.stage("recommend"):
        warnings = build_warnings(ewl, risks, all_anoms, sup, cfg, extras.anomaly_text)
        actions = build_actions(decisions, warnings, risks, key, extras.decision_actions)
    with tm.stage("explain"):
        signals = build_signals(
            trends,
            all_anoms,
            forecasts,
            series_last_totals(states, cfg.forecast_horizon),
            by_id,
            quality,
            stale_risks,
            as_of_ts,
            iso(data.as_of),
            iso(data.now),
            cfg,
            key,
            extra=extras.signals,
        )
        summary = summarise(signals, trends, all_anoms, risks, warnings, decisions, actions)
        series_out = [s.to_dict(cfg.series_output_months) for s in series]
    tm.ms["total"] = round(1000 * (time.perf_counter() - t_start), 2)
    out: dict[str, Any] = {
        "run": {
            "pipeline_version": getattr(adapter, "pipeline_version", f"jev-{CORE_VERSION}"),
            "as_of": key,
            "now": iso(data.now),
            "data_version": data.data_version,
            "model_version": data.model_version,
            "stage_ms": tm.ms,
            "config": cfg.to_dict(),
            "last_complete_month": last_complete,
            "domain": adapter.key,
            "core_version": CORE_VERSION,
            "frequency": {"W": "week", "week": "week", "D": "day", "day": "day"}.get(data.frequency, "month"),
            "validation": "core" if data.core_checks else f"adapter: {data.core_checks_reason}",
            # lineage identifiers (core-1.1.0): the exact config, and the inputs as_of saw
            "config_hash": short_hash(clean(cfg.to_dict()), 16),
            "input_fingerprint": input_fingerprint(data.sources, data.data_version, key),
        },
        "data": {
            "sources": data.sources,
            "quality": quality,
            "excluded_after_as_of": excluded,
            **extras.data,
        },
        "series": series_out,
        "signals": signals,
        "trends": trends,
        "anomalies": all_anoms,
        "predictions": {"forecasts": forecasts, "share_forecasts": share_forecasts, **extras.predictions},
        "risks": risks,
        "decisions": decisions,
        "decision_batches": batches,
        "warnings": warnings,
        "actions": actions,
        "summary": summary,
        "diagnostics": diagnostics,
    }
    stamp_domain(out, adapter.key)
    return PipelineResult(
        data=out,
        series_objects=series,
        forecast_states=states,
        prepared=data.context,
        config=cfg,
        domain=adapter.key,
    )


def input_fingerprint(sources: list[dict[str, Any]], data_version: str, as_of_key: str) -> str:
    """sha1 (16 hex) of what the run read: data version, as_of, and per source its rows and first/last
    event. Two runs with the same fingerprint and config_hash saw the same inputs (by these counts)."""
    parts = [[s.get("source"), s.get("rows"), s.get("first_event"), s.get("last_event")] for s in sources]
    return short_hash(clean([data_version, as_of_key, parts]), 16)


def stamp_domain(out: dict[str, Any], domain: str) -> None:
    """Every emitted object gains ``"domain"`` (platform contract section 2)."""
    for sec in OBJECT_SECTIONS:
        for o in out.get(sec, []):
            o.setdefault("domain", domain)
    for f in out["predictions"].get("forecasts", []) + out["predictions"].get("share_forecasts", []):
        f.setdefault("domain", domain)
    for s in out["data"].get("sources", []):
        s.setdefault("domain", domain)
