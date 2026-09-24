"""The domain adapter protocol (platform contract section 3).

An adapter loads its data as generic observations, declares which series to build, and may add
domain-specific stages through ``extra``. ``jev_ml.core.pipeline.run_domain`` drives it:

    data = adapter.load(as_of, now, config)          # DomainData (observations <= as_of)
    checks = adapter.quality_checks(data)             # added to the core checks
    ... core: series -> trends -> anomalies -> forecasts ...
    extras = adapter.extra(ctx)                       # DomainExtras from a CoreContext
    ... core: risks -> early-warning decisions -> warnings -> actions -> signals -> summary

Deviation from the contract sketch: ``load`` also receives the run's config (the adapter's
validation thresholds live there), and ``DomainAdapter.default_config()`` supplies the config class
the domain needs (the movie domain's ``IntelConfig`` subclasses ``CoreConfig``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from jev_ml.core.config import CoreConfig
from jev_ml.core.series import Series, SeriesSpec

Check = dict[str, Any]  # {name, passed, value, threshold, severity, detail, source}


@dataclass
class DomainInfo:
    key: str
    name: str
    description: str
    entity_types: list[str]
    frequency: str = "month"  # month | week
    sources: list[dict[str, Any]] = field(default_factory=list)  # {source, kind, license, url}
    capabilities: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "entity_types": list(self.entity_types),
            "frequency": self.frequency,
            "sources": [dict(s) for s in self.sources],
            "capabilities": dict(self.capabilities),
        }


@dataclass
class DomainData:
    """What an adapter's ``load`` returns. ``observations`` must already exclude rows after
    ``as_of`` when ``core_checks`` is False (the adapter validated its raw data itself); otherwise
    the core validates and filters them (``core.quality.validate_observations``)."""

    observations: pd.DataFrame
    specs: list[SeriesSpec]
    as_of: datetime
    now: datetime
    sources: list[dict[str, Any]]
    data_version: str
    entities: list[dict[str, Any]] = field(default_factory=list)
    quality_checks: list[Check] = field(default_factory=list)  # checks the adapter already ran
    core_checks: bool = True
    core_checks_reason: str | None = None  # why the core does not validate (when core_checks False)
    excluded_after_as_of: int = 0
    frequency: str = "M"
    grid_end: str = "as_of"  # as_of | last_observation
    model_version: str | None = None
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    context: Any = None  # adapter-private (the movie adapter keeps its Prepared frames here)
    earliest_ts: float | None = None  # core validation: timestamps before this are invalid
    share_forecast_min_share: float = 0.0  # share series below this recent mean are not forecast


@dataclass
class CoreContext:
    """Everything the core computed before the adapter's ``extra`` hook runs."""

    domain: str
    config: CoreConfig
    data: DomainData
    as_of_key: str
    as_of_ts: float
    now_ts: float
    series: list[Series]
    series_by_id: dict[str, Series]
    trends: list[dict[str, Any]]
    anomalies: list[dict[str, Any]]  # core series anomalies
    forecasts: list[dict[str, Any]]
    share_forecasts: list[dict[str, Any]]
    forecast_states: dict[str, Any]
    quality: dict[str, Any]
    suppressed: dict[str, str]
    last_complete: str | None
    timer: Any  # stage timer: ``with ctx.timer.stage("lapse"): ...``
    diagnostics: list[dict[str, Any]]


@dataclass
class DomainExtras:
    """Optional domain-specific stage output, merged by the core."""

    anomalies: list[dict[str, Any]] = field(default_factory=list)
    predictions: dict[str, Any] = field(default_factory=dict)  # merged into result.predictions
    risks: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    decision_batches: list[dict[str, Any]] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)  # merged into result.data
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    # hooks: domain decisions -> actions; wording of warnings for adapter-specific anomaly methods
    decision_actions: Callable[..., None] | None = None
    anomaly_text: Callable[..., dict[str, Any] | None] | None = None


@runtime_checkable
class DomainAdapter(Protocol):
    key: str
    info: DomainInfo

    def default_config(self) -> CoreConfig: ...

    def load(self, as_of: datetime | None, now: datetime | None, config: CoreConfig) -> DomainData: ...

    def quality_checks(self, data: DomainData) -> list[Check]: ...

    def extra(self, ctx: CoreContext) -> DomainExtras: ...
