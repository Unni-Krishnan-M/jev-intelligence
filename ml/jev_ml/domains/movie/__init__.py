"""The movie domain adapter (MovieLens + app events + the hybrid recommender's artefacts)."""

from jev_ml.domains.movie.adapter import INFO, MovieAdapter, available
from jev_ml.domains.movie.config import PIPELINE_VERSION, IntelConfig
from jev_ml.domains.movie.ingest import PipelineInputs, Prepared, load_default_inputs

__all__ = [
    "INFO",
    "PIPELINE_VERSION",
    "IntelConfig",
    "MovieAdapter",
    "PipelineInputs",
    "Prepared",
    "available",
    "load_default_inputs",
]
