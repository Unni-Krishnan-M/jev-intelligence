"""Compatibility shim: ``run_pipeline(PipelineInputs, config)`` = ``run_domain(MovieAdapter(inputs))``.

Returns exactly what intel-1.1.0 returned for the movie domain, plus the platform's additive fields
(docs/platform.md section 1). ``PipelineResult`` is the core result type.
"""

from __future__ import annotations

from datetime import datetime

from jev_ml.core.pipeline import PipelineResult
from jev_ml.domains.movie.config import IntelConfig
from jev_ml.domains.movie.ingest import PipelineInputs, load_default_inputs
from jev_ml.domains.movie.scenario import run_movie_pipeline


def run_pipeline(inputs: PipelineInputs, config: IntelConfig | None = None) -> PipelineResult:
    return run_movie_pipeline(inputs, config)


def run_default(
    as_of: datetime | str | None = None, now: datetime | None = None, config: IntelConfig | None = None
) -> PipelineResult:
    """The movie pipeline on the real files (``load_default_inputs``)."""
    return run_pipeline(load_default_inputs(as_of=as_of, now=now), config)


__all__ = ["PipelineResult", "run_default", "run_pipeline"]
