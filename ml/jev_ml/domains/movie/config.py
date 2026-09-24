"""Configuration of the movie domain: ``IntelConfig`` = ``CoreConfig`` + MovieLens/app settings.

Every threshold of the movie adapter lives here (embedded in ``run.config``); each default has a
one-line reason. The core settings keep the values of intel-1.1.0. The platform fields are set so
the movie domain reproduces the v1.1 warning set exactly (warnings per component, no
corroboration bonus, anomalies in either direction count, no generic risk families: the movie
domain has its own genre risk).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from jev_ml.core.config import FORECAST_VERSION, SEVERITIES, CoreConfig, severity_rank

PIPELINE_VERSION = "intel-1.1.0"  # 1.1: score decisions, decision batches (contract section 9.1)
LAPSE_VERSION = "lapse-1.0.0"

__all__ = [
    "FORECAST_VERSION",
    "LAPSE_VERSION",
    "PIPELINE_VERSION",
    "SEVERITIES",
    "IntelConfig",
    "severity_rank",
]


@dataclass
class IntelConfig(CoreConfig):
    # ---- validation ---------------------------------------------------------------------------
    rating_min: float = 0.5
    rating_max: float = 5.0
    earliest_valid_ts: int = 788_918_400  # 1995-01-01: MovieLens started collecting in 1995

    # ---- series -------------------------------------------------------------------------------
    rating_min_count: int = 20  # a genre mean rating from < 20 ratings is too noisy to chart

    # ---- rater anomalies (IsolationForest) ---------------------------------------------------
    rater_min_ratings: int = 5
    rater_contamination: float = 0.02  # ~12 of 610 users: a review queue an operator can handle
    rater_estimators: int = 200
    rater_active_days: int = 365  # flagged raters inactive for longer are reported but suppressed
    rater_longtail_max_count: int = 5  # an item with <= 5 ratings from *other* users is long tail
    # IsolationForest anomaly score s in (0, 1]; Liu et al. (2008): s ~ 0.5 is ordinary, s -> 1
    # clearly anomalous. Bands only apply to raters already in the top `contamination` share.
    rater_severity_bands: dict[str, float] = field(
        default_factory=lambda: {"medium": 0.66, "high": 0.72, "critical": 0.80}
    )
    influence_top_n: int = 50
    influence_half_life_days: float = 365.0  # same half-life as the recommender's trending signal

    # ---- live app feedback --------------------------------------------------------------------
    live_recent_days: int = 7
    live_prior_days: int = 28
    live_min_feedback: int = 30  # per window; a rate from fewer events is not tested
    live_alpha: float = 0.05
    live_fresh_days: float = 2.0  # "app" source is fresh when its last event is < 2 days old

    # ---- lapse model --------------------------------------------------------------------------
    lapse_horizon_days: int = 180
    lapse_lookback_days: int = 365  # a sample/user is "active" with >= 1 rating in this window
    lapse_cutoff_step_months: int = 6
    lapse_first_cutoff_year: int = 1998
    lapse_test_frac: float = 0.3  # latest 30 % of cutoffs form the temporal test set
    lapse_val_frac: float = 0.25  # latest 25 % of the training cutoffs select the calibration
    lapse_min_samples: int = 200
    lapse_min_test: int = 50
    lapse_high_risk: float = 0.7
    lapse_c: float = 1.0

    # ---- risk ---------------------------------------------------------------------------------
    # Declared business-impact weights (0..1) for kinds whose impact cannot be measured from data.
    # These are policy inputs set by the operator, not estimates; kinds with a measurable impact
    # (genre share, lapse volume share, manipulation influence) compute it instead.
    risk_declared_impact: dict[str, float] = field(
        default_factory=lambda: {
            "data_quality": 0.7,
            "data_staleness": 0.5,
            "model_staleness": 0.6,
            "model_quality": 0.8,
            "recommendation_rejection": 0.7,
        }
    )
    generic_risks: tuple[str, ...] = ()  # the movie domain's genre risk replaces adverse_trend
    genre_impact_share_ref: float = 0.25  # a genre with >= 25 % of volume has full impact
    model_min_lead: float = 0.05  # hybrid must beat the best single model by >= 5 % NDCG@10

    # ---- decisions ----------------------------------------------------------------------------
    retrain_new_event_frac: float = 0.05  # retrain once >= 5 % new events since training
    bootstrap_samples: int = 2000
    bootstrap_min_users: int = 30
    genre_min_share: float = 0.01  # genres with < 1 % of recent volume get no programming decision
    rater_decisions_max: int = 5
    rater_influence_ref: float = 3.0  # removing one user changing >= 3 top-50 films = full influence
    reengage_min_auc: float = 0.65
    reengage_min_users: int = 5
    slot_share_window_months: int = 3  # "next quarter": slots are planned per quarter
    # a window interval whose backtested coverage fell below 60 % cannot be reported as an 80 %
    # interval (at the ~20 origins available, 60 % is ~2.2 binomial SDs under 80 %)
    slot_share_min_coverage: float = 0.6

    # ---- early warning (platform) ----------------------------------------------------------------
    ewl_corroboration: float = 0.0  # reproduce v1.1: agreement of stages is not an extra trigger
    ewl_anomaly_direction: str = "any"  # v1.1 warns on spikes and drops alike
    warning_mode: str = "components"  # v1.1 keys: risk:<kind>:<entity> and anomaly dedup keys
