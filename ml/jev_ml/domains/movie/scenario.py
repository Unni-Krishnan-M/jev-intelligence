"""What-if analysis for the movie domain: ``run_scenario`` on a finished ``PipelineResult`` or on movie
``PipelineInputs`` (prepared at ``spec['as_of']`` when given). The engine is ``jev_ml.core.scenario``;
``jev_ml.intel.scenario`` re-exports this module for v1.1 callers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from jev_ml.core.common import iso, to_utc
from jev_ml.core.pipeline import PipelineResult
from jev_ml.core.scenario import run_scenario as _core_run_scenario
from jev_ml.core.scenario import scenario_on_series
from jev_ml.domains.movie.config import IntelConfig
from jev_ml.domains.movie.ingest import PipelineInputs, prepare
from jev_ml.domains.movie.series import build_series


def run_scenario(
    source: PipelineResult | PipelineInputs, spec: dict[str, Any], config: IntelConfig | None = None
) -> dict[str, Any]:
    """Run a what-if analysis (docs/intelligence.md, scenario contract). Raises ValueError for
    invalid specs."""
    if isinstance(source, PipelineResult):
        return _core_run_scenario(source, spec, config or source.config)
    cfg = config or IntelConfig()
    if spec.get("as_of"):
        source = replace(source, as_of=to_utc(spec["as_of"]))
    prep = prepare(source, cfg)
    series = build_series(prep, cfg)
    return scenario_on_series(series, {}, iso(prep.as_of) or "", prep.data_version, spec, cfg)


def run_movie_pipeline(inputs: PipelineInputs, config: IntelConfig | None = None) -> PipelineResult:
    """``run_domain(MovieAdapter(inputs))``: the movie pipeline on explicit inputs (the API path)."""
    from jev_ml.core.pipeline import run_domain
    from jev_ml.domains.movie.adapter import MovieAdapter

    return run_domain(
        MovieAdapter(inputs),
        as_of=inputs.as_of,
        now=inputs.now,
        suppressed_keys=inputs.suppressed_keys,
        config=config or IntelConfig(),
    )


__all__ = ["run_movie_pipeline", "run_scenario"]
