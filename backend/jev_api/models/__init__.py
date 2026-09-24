"""ORM models, split per workstream (docs/PHASE2_ARCHITECTURE_AUDIT.md, "Enabler — done").

Every name that ``jev_api.models`` exported as a single module is re-exported here, so
``from jev_api.models import X`` keeps working. Importing this package registers every table on
``Base.metadata`` (migrations/env.py relies on it).

Integrator-owned. To add a table: define it in your workstream's module (events.py, governance.py,
experiments.py, security.py, ...), then append ONE import line in the block below and its names to
``__all__``. New CHECK vocabularies go in enums.py (integrator).
"""

from __future__ import annotations

from jev_api.models import events, experiments, governance, security  # noqa: F401 - WS placeholders
from jev_api.models.audit import AuditLog
from jev_api.models.base import Base, TimestampMixin, domain_column, in_, nullable_in, utcnow
from jev_api.models.catalog import Genre, Movie, MovieGenre
from jev_api.models.enums import (
    AUDIT_ACTIONS,
    AUDIT_ACTIONS_PHASE2,
    AUDIT_ACTIONS_V12,
    CHECK_ENUMS,
    CONFIDENCE_KINDS,
    DECISION_KINDS,
    DEFAULT_DOMAIN,
    EVIDENCE_OWNERS,
    FEEDBACK_VERDICTS,
    INTEL_RUN_MODES,
    INTEL_RUN_STATUSES,
    INTEL_RUN_TRIGGERS,
    INTEL_SEVERITIES,
    ME_FEEDBACK_TARGETS,
    ME_FEEDBACK_VERDICTS,
    NON_ENUM_CHECKS,
    REC_CONFIDENCE_KINDS,
    REC_FEEDBACK_KINDS,
    TREND_DIRECTIONS,
    WARNING_OPEN_STATUSES,
    WARNING_STATUSES,
)
from jev_api.models.intel import (
    _OPEN_WARNING,
    _VERDICT_MATCHES_TARGET,
    IntelAnomalyRow,
    IntelDecision,
    IntelEvaluationRun,
    IntelEvidenceRow,
    IntelFeedback,
    IntelForecastRow,
    IntelRiskRow,
    IntelRun,
    IntelScenario,
    IntelSignalRow,
    IntelTrendRow,
    IntelWarning,
    IntelWarningEvent,
    run_mode,
)
from jev_api.models.interactions import Favorite, Rating, WatchHistory
from jev_api.models.model_registry import EvaluationMetric, Experiment, ModelVersion
from jev_api.models.recs import (
    _CLICK,
    _NO_REC,
    _VERDICT,
    _WITH_REC,
    FEEDBACK_UNIQUE_INDEXES,
    Recommendation,
    RecommendationFeedback,
    UserIntelFeedback,
)
from jev_api.models.users import User, UserGenrePreference

# the pre-split private helper names, kept importable
_in = in_
_nullable_in = nullable_in
_domain_column = domain_column

__all__ = [
    "AUDIT_ACTIONS",
    "AUDIT_ACTIONS_PHASE2",
    "AUDIT_ACTIONS_V12",
    "CHECK_ENUMS",
    "CONFIDENCE_KINDS",
    "DECISION_KINDS",
    "DEFAULT_DOMAIN",
    "EVIDENCE_OWNERS",
    "FEEDBACK_UNIQUE_INDEXES",
    "FEEDBACK_VERDICTS",
    "INTEL_RUN_MODES",
    "INTEL_RUN_STATUSES",
    "INTEL_RUN_TRIGGERS",
    "INTEL_SEVERITIES",
    "ME_FEEDBACK_TARGETS",
    "ME_FEEDBACK_VERDICTS",
    "NON_ENUM_CHECKS",
    "REC_CONFIDENCE_KINDS",
    "REC_FEEDBACK_KINDS",
    "TREND_DIRECTIONS",
    "WARNING_OPEN_STATUSES",
    "WARNING_STATUSES",
    "_CLICK",
    "_NO_REC",
    "_OPEN_WARNING",
    "_VERDICT",
    "_VERDICT_MATCHES_TARGET",
    "_WITH_REC",
    "AuditLog",
    "Base",
    "EvaluationMetric",
    "Experiment",
    "Favorite",
    "Genre",
    "IntelAnomalyRow",
    "IntelDecision",
    "IntelEvaluationRun",
    "IntelEvidenceRow",
    "IntelFeedback",
    "IntelForecastRow",
    "IntelRiskRow",
    "IntelRun",
    "IntelScenario",
    "IntelSignalRow",
    "IntelTrendRow",
    "IntelWarning",
    "IntelWarningEvent",
    "ModelVersion",
    "Movie",
    "MovieGenre",
    "Rating",
    "Recommendation",
    "RecommendationFeedback",
    "TimestampMixin",
    "User",
    "UserGenrePreference",
    "UserIntelFeedback",
    "WatchHistory",
    "_domain_column",
    "_in",
    "_nullable_in",
    "domain_column",
    "in_",
    "nullable_in",
    "run_mode",
    "utcnow",
]
