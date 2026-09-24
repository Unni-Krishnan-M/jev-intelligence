"""JEV core: the domain-independent decision & early-warning engine (docs/platform.md).

Entry point: ``run_domain(adapter, as_of=None, now=None, suppressed_keys=None, config=None)``.
Core never imports the recommender (``jev_ml.models``, ``engine``, ``training``, ``calibration``) or
anything movie-specific.
"""

from jev_ml.core.adapter import CoreContext, DomainAdapter, DomainData, DomainExtras, DomainInfo
from jev_ml.core.config import CORE_VERSION, EWL_POLICY_VERSION, CoreConfig
from jev_ml.core.pipeline import PipelineResult, run_domain
from jev_ml.core.series import Series, SeriesSpec

__all__ = [
    "CORE_VERSION",
    "EWL_POLICY_VERSION",
    "CoreConfig",
    "CoreContext",
    "DomainAdapter",
    "DomainData",
    "DomainExtras",
    "DomainInfo",
    "PipelineResult",
    "Series",
    "SeriesSpec",
    "run_domain",
]
