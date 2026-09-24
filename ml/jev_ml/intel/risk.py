"""Compatibility: the risk framework moved to ``jev_ml.core.risk``; movie risks to
``jev_ml.domains.movie.risks``."""

from jev_ml.core.risk import level_of, make_risk  # noqa: F401
from jev_ml.domains.movie.risks import (  # noqa: F401
    genre_risks,
    lapse_risk,
    manipulation_risk,
    model_risks,
    rejection_risk,
)
