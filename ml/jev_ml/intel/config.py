"""Compatibility: ``IntelConfig`` moved to ``jev_ml.domains.movie.config`` (a ``CoreConfig`` subclass)."""

from jev_ml.core.config import SEVERITIES, CoreConfig, severity_rank  # noqa: F401
from jev_ml.domains.movie.config import (  # noqa: F401
    FORECAST_VERSION,
    LAPSE_VERSION,
    PIPELINE_VERSION,
    IntelConfig,
)
