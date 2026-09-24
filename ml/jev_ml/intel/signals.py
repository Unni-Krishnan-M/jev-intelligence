"""Compatibility: moved to ``jev_ml.core.signals`` and ``jev_ml.domains.movie.signals``."""

from jev_ml.core.signals import (  # noqa: F401
    FORECAST_SIGNAL_MAX,
    FORECAST_SIGNAL_MIN_CHANGE,
    build_signals,
    series_last_totals,
)
from jev_ml.domains.movie.signals import movie_signals  # noqa: F401
