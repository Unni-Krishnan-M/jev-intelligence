"""Compatibility shim: ``run_pipeline(PipelineInputs, config)`` = ``run_domain(MovieAdapter(inputs))``.

Returns exactly what intel-1.1.0 returned for the movie domain, plus the platform's additive fields
(docs/platform.md section 1). ``PipelineResult`` is the core result type.
"""

from __future__ import annotations

from datetime import datetime

from jev_ml.core.pipeline import PipelineResult, run_domain
from jev_ml.domains.movie.adapter import MovieAdapter
from jev_ml.domains.movie.config import IntelConfig
from jev_ml.domains.movie.ingest import PipelineInputs, load_default_inputs


def run_pipeline(inputs: PipelineInputs, config: IntelConfig | None = None) -> PipelineResult:
    return run_domain(
        MovieAdapter(inputs),
        as_of=inputs.as_of,
        now=inputs.now,
        suppressed_keys=inputs.suppressed_keys,
        config=config or IntelConfig(),
    )


def run_default(
    as_of: datetime | str | None = None, now: datetime | None = None, config: IntelConfig | None = None
) -> PipelineResult:
    """The movie pipeline on the real files (``load_default_inputs``)."""
    return run_pipeline(load_default_inputs(as_of=as_of, now=now), config)


__all__ = ["PipelineResult", "run_default", "run_pipeline"]
