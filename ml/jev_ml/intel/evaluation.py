"""Compatibility: the movie evaluation moved to ``jev_ml.domains.movie.evaluation``."""

from jev_ml.domains.movie.evaluation import (  # noqa: F401
    ATTACK_TYPES,
    evaluate,
    evaluate_change_points,
    evaluate_forecasts,
    evaluate_latency,
    evaluate_series_anomalies,
    evaluate_shilling,
    run_evaluation,
    write_report,
)

__all__ = ["evaluate", "run_evaluation", "write_report"]
