"""JEV intelligence layer, v1.1 API (compatibility layer over the platform core).

The engines live in ``jev_ml.core`` and the movie domain in ``jev_ml.domains.movie``; these modules
re-export them. ``run_pipeline(PipelineInputs)`` runs the core pipeline with the movie adapter.
See docs/intelligence.md (result contract) and docs/platform.md (core + adapters).
"""

from jev_ml.intel.config import PIPELINE_VERSION, IntelConfig
from jev_ml.intel.ingest import PipelineInputs, load_default_inputs
from jev_ml.intel.pipeline import PipelineResult, run_default, run_pipeline
from jev_ml.intel.scenario import run_scenario

__all__ = [
    "PIPELINE_VERSION",
    "IntelConfig",
    "PipelineInputs",
    "PipelineResult",
    "load_default_inputs",
    "run_default",
    "run_pipeline",
    "run_scenario",
]
