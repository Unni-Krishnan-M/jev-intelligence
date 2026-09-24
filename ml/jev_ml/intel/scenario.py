"""Compatibility: ``run_scenario`` moved to ``jev_ml.domains.movie.scenario`` (engine: core.scenario)."""

from jev_ml.core.scenario import DEFAULT_MULT, KINDS, MAX_SCENARIOS, scenario_on_series  # noqa: F401
from jev_ml.domains.movie.scenario import run_scenario

__all__ = ["run_scenario"]
