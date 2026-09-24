"""Compatibility: moved to ``jev_ml.domains.movie.ingest``."""

from jev_ml.core.quality import Checks as _Checks  # noqa: F401
from jev_ml.domains.movie.ingest import (  # noqa: F401
    APP_FEEDBACK_COLUMNS,
    APP_RATING_COLUMNS,
    APP_SERVED_COLUMNS,
    FEEDBACK_VALUES,
    INTERACTION_COLUMNS,
    NEGATIVE_FEEDBACK,
    NO_GENRE,
    PipelineInputs,
    Prepared,
    load_default_inputs,
    model_cutoff,
    prepare,
    quality_evidence,
)
