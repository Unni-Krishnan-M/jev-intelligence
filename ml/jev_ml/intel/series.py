"""Compatibility: ``Series`` moved to ``jev_ml.core.series``; the movie builder to
``jev_ml.domains.movie.series``."""

from jev_ml.core.series import COUNT_METRICS, Series, SeriesSpec  # noqa: F401
from jev_ml.domains.movie.series import build_series, month_range  # noqa: F401
