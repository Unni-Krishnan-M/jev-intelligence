"""Core types (platform contract section 2).

Each dataclass names the fields of one wire object; ``to_dict()`` returns the wire format of
docs/intelligence.md section 4 plus the platform's additive fields, and ``from_dict()`` reads it
back. Stage-specific fields that a type does not name (for example a trend's ``slope_ci``) travel in
``extra`` unchanged, so ``T.from_dict(d).to_dict() == d`` for every emitted object (the round trip
is tested on real pipeline output).

How they are used: the engines build plain dicts for speed and v1.1 compatibility; these types are
the typed view of the same records (``typed(result)``), and ``SeriesSpec`` (series building),
``DataSource``/``Entity`` (adapters) and ``Evidence`` are used directly. Added fields are derived
from values the stage already computes; none is a constant or a placeholder. A value that does not
exist is ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, TypeVar

from jev_ml.core.series import SeriesSpec

T = TypeVar("T", bound="WireType")

# columns of the generic observation frame (group columns are extra, declared by the adapter)
OBSERVATION_COLUMNS = ("timestamp", "entity_id", "entity_type", "event_type", "value", "source")


@dataclass
class WireType:
    extra: dict[str, Any] = field(default_factory=dict, kw_only=True)
    _order: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def from_dict(cls: type[T], d: dict[str, Any]) -> T:
        names = [f.name for f in fields(cls) if f.name != "extra"]
        known = {k: d[k] for k in names if k in d}
        obj = cls(**known)
        obj.extra = {k: v for k, v in d.items() if k not in names}
        obj._present = [k for k in d]  # type: ignore[attr-defined]
        return obj

    def to_dict(self) -> dict[str, Any]:
        names = [f.name for f in fields(self) if f.name != "extra"]
        out = {k: getattr(self, k) for k in names}
        out.update(self.extra)
        present = getattr(self, "_present", None)
        if present is not None:  # keep the original key order (and only the keys that were there)
            return {k: out[k] for k in present}
        return out


@dataclass
class Evidence(WireType):
    kind: str = "metric"  # metric | series | record | test | model
    label: str = ""
    value: Any = None
    detail: str | None = None
    ref: str | None = None


@dataclass
class DataSource(WireType):
    source: str = ""
    kind: str = "dataset"  # static_snapshot | live | artefact | dataset
    rows: int = 0
    first_event: str | None = None
    last_event: str | None = None
    age_days: float | None = None
    lag_days: float | None = None
    expected_update: str | None = None
    fresh: bool | None = None
    detail: str | None = None
    license: str | None = None
    url: str | None = None
    checksum: str | None = None


@dataclass
class Entity(WireType):
    entity_id: str = ""  # "<type>:<name>"
    entity_type: str = ""
    name: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Signal(WireType):
    id: str = ""
    kind: str = ""
    entity_type: str = ""
    entity: str = ""
    value: Any = None
    unit: str | None = None
    strength: float | None = None
    direction: str | None = None
    source: str | None = None
    observed_at: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    entity_id: str | None = None
    baseline: Any = None
    change: Any = None
    confidence: float | None = None
    confidence_kind: str | None = None


@dataclass
class Trend(WireType):
    id: str = ""
    series_id: str = ""
    direction: str = "flat"
    slope: float | None = None
    p_value: float | None = None
    q_value: float | None = None
    magnitude: float | None = None  # recent_mean - prior_mean
    velocity: float | None = None  # slope per period
    baseline: float | None = None  # prior_mean
    supporting_observations: int = 0  # n_points
    confidence: float | None = None  # 1 - q
    confidence_kind: str = "evidence"


@dataclass
class Anomaly(WireType):
    id: str = ""
    kind: str = ""
    entity_type: str = ""
    entity: str = ""
    severity: str = "low"
    suppressed: bool = False
    observed_value: Any = None
    expected_value: Any = None
    anomaly_score: float | None = None  # 0..1 normalised strength
    confidence: float | None = None
    confidence_kind: str | None = None


@dataclass
class Forecast(WireType):
    id: str = ""
    series_id: str = ""
    model: str = ""
    points: list[dict[str, Any]] = field(default_factory=list)
    domain: str | None = None
    horizon_unit: str | None = None


@dataclass
class Risk(WireType):
    id: str = ""
    kind: str = ""
    entity_type: str = ""
    entity: str = ""
    score: float | None = None
    level: str = "low"
    factors: list[dict[str, Any]] = field(default_factory=list)
    severity: str | None = None  # = level
    contributing_factors: list[dict[str, Any]] = field(default_factory=list)  # = factors
    domain: str | None = None


@dataclass
class Decision(WireType):
    id: str = ""
    key: str = ""
    policy_version: str = ""
    kind: str = ""
    options: list[str] = field(default_factory=list)
    answer: Any = None
    option_scores: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    confidence_kind: str = ""
    abstained: bool = False
    fallback_reason: str | None = None


@dataclass
class Recommendation(WireType):
    item_id: Any = None
    score: float | None = None
    rank: int | None = None
    reason: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float | None = None
    decision_id: str | None = None


@dataclass
class Warning(WireType):
    key: str = ""
    severity: str = "low"
    confidence: float | None = None
    decision_id: str | None = None


@dataclass
class Action(WireType):
    id: str = ""
    title: str = ""
    priority: str = "P3"


@dataclass
class Scenario(WireType):
    series_id: str = ""
    scenarios: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Feedback(WireType):
    target_type: str = ""
    target_id: str = ""
    verdict: str = ""


# result section -> type of its items
SECTIONS: dict[str, type[WireType]] = {
    "signals": Signal,
    "trends": Trend,
    "anomalies": Anomaly,
    "risks": Risk,
    "decisions": Decision,
    "warnings": Warning,
    "actions": Action,
}


def typed(result: dict[str, Any]) -> dict[str, list[WireType]]:
    """Typed views of a result's objects (``PipelineResult.to_dict()``)."""
    out: dict[str, list[WireType]] = {
        k: [t.from_dict(o) for o in result.get(k, [])] for k, t in SECTIONS.items()
    }
    preds = result.get("predictions") or {}
    out["forecasts"] = [
        Forecast.from_dict(f) for f in preds.get("forecasts", []) + preds.get("share_forecasts", [])
    ]
    out["sources"] = [DataSource.from_dict(s) for s in (result.get("data") or {}).get("sources", [])]
    return out


__all__ = [
    "OBSERVATION_COLUMNS",
    "Action",
    "Anomaly",
    "DataSource",
    "Decision",
    "Entity",
    "Evidence",
    "Feedback",
    "Forecast",
    "Recommendation",
    "Risk",
    "Scenario",
    "SeriesSpec",
    "Signal",
    "Trend",
    "Warning",
    "typed",
]
