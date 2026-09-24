"""Compatibility: series anomalies moved to ``jev_ml.core.anomalies``; rater and live-feedback
detectors to ``jev_ml.domains.movie.raters``."""

from jev_ml.core.anomalies import _band, robust_z, scan_all_series, scan_series, suppression  # noqa: F401
from jev_ml.domains.movie.raters import (  # noqa: F401
    RATER_FEATURES,
    detect_raters,
    isolation_scores,
    live_feedback,
    rater_features,
)

SCAN_PREFIXES = ("volume:all", "active_users:all", "share:genre:", "rating:genre:")
