"""JEV intelligence layer: signals, early warnings, bounded decisions and action plans.

Pure functions over pandas/numpy (no web or DB imports). See docs/intelligence.md for the contract.
"""

from jev_ml.intel.config import PIPELINE_VERSION, IntelConfig
from jev_ml.intel.ingest import PipelineInputs, load_default_inputs
from jev_ml.intel.pipeline import PipelineResult, run_pipeline
from jev_ml.intel.scenario import run_scenario

__all__ = [
    "PIPELINE_VERSION",
    "IntelConfig",
    "PipelineInputs",
    "PipelineResult",
    "load_default_inputs",
    "run_pipeline",
    "run_scenario",
]
