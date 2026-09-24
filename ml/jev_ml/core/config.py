"""Domain-independent thresholds, windows and seeds of the JEV core engines.

``CoreConfig`` holds every number the generic engines use (series anomalies, trends, forecasts,
risk scoring, the early-warning decision and warnings). A domain extends it: the movie adapter's
``IntelConfig`` subclasses it with the MovieLens/app-specific fields, and the generic adapter
overrides fields from its YAML ``thresholds`` section. ``to_dict()`` is embedded in ``run.config``
so every run records exactly the configuration it used. Comments give the reason for each default.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

CORE_VERSION = "core-1.1.0"
FORECAST_VERSION = "fc-1.0.0"
EWL_POLICY_VERSION = "ewl-1.1.0"

SEVERITIES = ("low", "medium", "high", "critical")
EWL_LEVELS = ("NO_ACTION", "MONITOR", "WARNING", "URGENT_ACTION")
ADVERSE_DIRECTIONS = ("up", "down", "both", "none")
GENERIC_RISK_KINDS = ("adverse_trend", "adverse_forecast", "adverse_anomaly")


def severity_rank(sev: str | None) -> int:
    return SEVERITIES.index(sev) if sev in SEVERITIES else -1


@dataclass
class CoreConfig:
    # ---- global -------------------------------------------------------------------------------
    seed: int = 42
    series_output_months: int = 120  # points kept per series in to_dict() (charts need ~10 years)
    top_n: int = 20  # cap for top-N lists
    check_weights: dict[str, float] = field(
        default_factory=lambda: {"error": 3.0, "warning": 1.0, "info": 0.5}
    )

    # ---- trends -------------------------------------------------------------------------------
    trend_window_months: int = 24  # two full years: long enough for Mann–Kendall power, short
    #                                 enough to describe the *current* regime (periods for weekly data)
    trend_min_points: int = 12
    trend_alpha: float = 0.05
    # many series are tested per run, so risks/signals additionally require a Benjamini–Hochberg
    # q-value <= trend_fdr (controls the expected share of false trend claims at 10 %).
    trend_fdr: float = 0.10
    change_point_min_segment: int = 4  # periods on each side of a mean shift
    change_point_permutations: int = 499
    change_point_alpha: float = 0.01  # stricter than trend_alpha: monthly series are autocorrelated
    # core-1.1.0: the AR(1) null's phi is estimated on up to 60 periods *before* the trend window
    # (needs >= 24); 0 = the pre-1.1 in-window estimate (anti-conservative: 8-10 % false alarms at
    # a nominal 1 % for phi 0.56; about 2 % with the history estimate, INTELLIGENCE_ENGINE_AUDIT.md)
    change_point_history_min: int = 24
    change_point_history_max: int = 60
    # core-1.1.0 recovery-aware trends: a significant window trend whose latest value has already
    # retreated from the window's extreme (in the trend's direction) by > 2 sigma sqrt(3) (sigma: MAD
    # of period changes, the warning outcome's noise scale) is "reversing": context for the
    # early-warning decision, no adverse-trend risk. Basis "drawdown" (distance from the extreme) was
    # chosen over "recent" (change over the last 3 periods) on the 1990-94 / 1999-2003 tuning
    # windows only (docs/INTELLIGENCE_ENGINE_AUDIT.md). trend_reversal_periods 0 = off (core-1.0.0)
    trend_reversal_periods: int = 3
    trend_reversal_z: float = 2.0
    trend_reversal_basis: str = "drawdown"

    # ---- series anomalies (robust z, Iglewicz & Hoaglin 1993) -------------------------------
    anomaly_baseline_months: int = 24
    anomaly_min_baseline_months: int = 12
    anomaly_scan_months: int = 12  # only the last 12 complete periods are scanned for *new* events
    anomaly_z_threshold: float = 3.5  # the Iglewicz–Hoaglin cut-off for modified z-scores
    anomaly_severity_bands: dict[str, float] = field(
        default_factory=lambda: {"medium": 5.0, "high": 7.0, "critical": 10.0}
    )
    anomaly_min_volume: int = 50  # default min-volume guard for count-backed series specs

    # ---- forecasting --------------------------------------------------------------------------
    forecast_horizon: int = 6
    forecast_backtest_origins: int = 24
    forecast_history_months: int = 60  # training history cap per origin
    forecast_min_history: int = 24
    forecast_ma_window: int = 6
    forecast_interval: float = 0.80
    forecast_min_residuals: int = 8  # per horizon step; fewer -> pool all steps <= h
    holt_alphas: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8)
    holt_betas: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
    holt_phis: tuple[float, ...] = (0.8, 0.9, 0.98)

    # ---- risk ---------------------------------------------------------------------------------
    risk_levels: dict[str, float] = field(
        default_factory=lambda: {"medium": 15.0, "high": 30.0, "critical": 50.0}
    )
    # Declared business-impact weights (0..1) for kinds whose impact cannot be measured from data.
    risk_declared_impact: dict[str, float] = field(
        default_factory=lambda: {"data_quality": 0.7, "data_staleness": 0.5}
    )
    # Generic risk families computed by the core from any series with an adverse direction.
    generic_risks: tuple[str, ...] = GENERIC_RISK_KINDS
    # measured impact when no weight is declared: |relative change| / ref, capped at 1 (a 25 %
    # move of the window mean against the prior window, or of the forecast, is full impact)
    risk_relative_change_ref: float = 0.25
    # an adverse forecast risk needs the next-horizon mean to move >= 10 % against the last horizon
    forecast_risk_min_change: float = 0.10
    # declared impact weight per entity (0..1), from the domain config; labelled "declared"
    impact_weights: dict[str, float] = field(default_factory=dict)

    # ---- early-warning decision (ewl-1.1.0, contract section 4) -------------------------------
    # level thresholds on the aggregated point score
    ewl_level_points: dict[str, float] = field(
        default_factory=lambda: {"MONITOR": 1.0, "WARNING": 3.0, "URGENT_ACTION": 5.0}
    )
    # a component (risk/anomaly) at the lower edge of each severity band earns these points; points
    # are interpolated linearly inside a band, so they are monotone in the underlying measure
    ewl_severity_points: dict[str, float] = field(
        default_factory=lambda: {"low": 1.0, "medium": 3.0, "high": 5.0, "critical": 7.0}
    )
    ewl_max_points: float = 8.0
    # context evidence (an adverse trend, change point or forecast; an anomaly outside the recent
    # window or against the adverse direction) can support MONITOR but never WARNING on its own
    ewl_context_points: tuple[float, float] = (1.0, 2.5)
    # points added per *additional* independent detection stage (trend, anomaly, forecast) with
    # >= 1 point: agreement of independent detectors is itself evidence
    ewl_corroboration: float = 1.0
    ewl_ramp: float = 1.0  # half-width (points) of the option-score ramp around each threshold
    ewl_anomaly_direction: str = "adverse"  # "adverse": favourable anomalies are context only
    # "situation": one warning per situation at WARNING/URGENT_ACTION; "components": one warning per
    # qualifying risk/anomaly (keys risk:<kind>:<entity> / the anomaly dedup key)
    warning_mode: str = "situation"

    # ---- warnings -----------------------------------------------------------------------------
    warning_min_severity: str = "medium"
    # a series anomaly raises a warning only while it is recent (last 3 complete periods)
    warning_anomaly_recent_months: int = 3
    max_warnings: int = 25

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()}
