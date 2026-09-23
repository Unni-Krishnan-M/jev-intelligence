"""All thresholds, windows and seeds of the intelligence layer in one place.

Every number that changes what the layer reports lives here, so a run records exactly the
configuration it used (``IntelConfig.to_dict()`` is embedded in ``run.config``). Comments give the
reason for each default; none of them were chosen to make a particular output look good.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PIPELINE_VERSION = "intel-1.0.0"
FORECAST_VERSION = "fc-1.0.0"
LAPSE_VERSION = "lapse-1.0.0"

SEVERITIES = ("low", "medium", "high", "critical")


def severity_rank(sev: str | None) -> int:
    return SEVERITIES.index(sev) if sev in SEVERITIES else -1


@dataclass
class IntelConfig:
    # ---- global -------------------------------------------------------------------------------
    seed: int = 42
    series_output_months: int = 120  # points kept per series in to_dict() (charts need ~10 years)
    top_n: int = 20  # cap for top-N lists (lapse users, rater anomalies in output)

    # ---- validation ---------------------------------------------------------------------------
    rating_min: float = 0.5
    rating_max: float = 5.0
    earliest_valid_ts: int = 788_918_400  # 1995-01-01: MovieLens started collecting in 1995
    check_weights: dict[str, float] = field(
        default_factory=lambda: {"error": 3.0, "warning": 1.0, "info": 0.5}
    )

    # ---- series -------------------------------------------------------------------------------
    rating_min_count: int = 20  # a genre mean rating from < 20 ratings is too noisy to chart

    # ---- trends -------------------------------------------------------------------------------
    trend_window_months: int = 24  # two full years: long enough for Mann–Kendall power, short
    #                                 enough to describe the *current* regime
    trend_min_points: int = 12
    trend_alpha: float = 0.05
    # ~60 series are tested per run, so risks/signals additionally require a Benjamini–Hochberg
    # q-value <= trend_fdr (controls the expected share of false trend claims at 10 %).
    trend_fdr: float = 0.10
    change_point_min_segment: int = 4  # months on each side of a mean shift
    change_point_permutations: int = 499
    change_point_alpha: float = 0.01  # stricter than trend_alpha: monthly counts are autocorrelated,
    #                                   which makes permutation p-values optimistic

    # ---- series anomalies (robust z, Iglewicz & Hoaglin 1993) -------------------------------
    anomaly_baseline_months: int = 24
    anomaly_min_baseline_months: int = 12
    anomaly_scan_months: int = 12  # only the last 12 complete months are scanned for *new* events
    anomaly_z_threshold: float = 3.5  # the Iglewicz–Hoaglin cut-off for modified z-scores
    # Severity bands on |robust z| (count series are scored on log1p scale, so z=5 is roughly a
    # 3-4x jump over a typical month on this data).
    anomaly_severity_bands: dict[str, float] = field(
        default_factory=lambda: {"medium": 5.0, "high": 7.0, "critical": 10.0}
    )
    anomaly_min_volume: int = 50  # ratings in the month (or baseline median) below which a spike
    #                               is suppressed: +40 ratings on a base of 10 is not news

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

    # ---- forecasting --------------------------------------------------------------------------
    forecast_horizon: int = 6
    forecast_backtest_origins: int = 24
    forecast_history_months: int = 60  # training history cap per origin: damped Holt with
    #                                    alpha >= 0.1 forgets older data anyway; keeps runs fast
    forecast_min_history: int = 24
    forecast_ma_window: int = 6
    forecast_interval: float = 0.80
    forecast_min_residuals: int = 8  # per horizon step; fewer -> pool all steps <= h
    holt_alphas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8)
    holt_betas: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
    holt_phis: tuple[float, ...] = (0.8, 0.9, 0.98)

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
    risk_levels: dict[str, float] = field(
        default_factory=lambda: {"medium": 15.0, "high": 30.0, "critical": 50.0}
    )
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

    # ---- warnings / actions -------------------------------------------------------------------
    warning_min_severity: str = "medium"
    # a series anomaly raises a warning only while it is recent (last 3 complete months); older
    # ones stay in `anomalies` for context but would otherwise re-open the same warning for a year
    warning_anomaly_recent_months: int = 3
    max_warnings: int = 25

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}
