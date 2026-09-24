"""Compatibility: ``run_scenario`` on a ``PipelineResult`` or on movie ``PipelineInputs``.
The engine lives in ``jev_ml.core.scenario``."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from jev_ml.core.common import iso, to_utc
from jev_ml.core.pipeline import PipelineResult
from jev_ml.core.scenario import DEFAULT_MULT, KINDS, MAX_SCENARIOS, scenario_on_series  # noqa: F401
from jev_ml.core.scenario import run_scenario as _core_run_scenario
from jev_ml.domains.movie.config import IntelConfig
from jev_ml.domains.movie.ingest import PipelineInputs, prepare
from jev_ml.domains.movie.series import build_series


def run_scenario(
    source: PipelineResult | PipelineInputs, spec: dict[str, Any], config: IntelConfig | None = None
) -> dict[str, Any]:
    """Run a what-if analysis (docs/intelligence.md, scenario contract). ``source`` is a finished
    PipelineResult (its series and as_of are reused) or PipelineInputs (then ``spec['as_of']``, when
    given, overrides ``inputs.as_of``). Raises ValueError for invalid specs."""
    if isinstance(source, PipelineResult):
        return _core_run_scenario(source, spec, config or source.config)
    cfg = config or IntelConfig()
    if spec.get("as_of"):
        source = replace(source, as_of=to_utc(spec["as_of"]))
    prep = prepare(source, cfg)
    series = build_series(prep, cfg)
    return scenario_on_series(series, {}, iso(prep.as_of) or "", prep.data_version, spec, cfg)
