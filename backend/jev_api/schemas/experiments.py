"""Online experiment schemas (WS5, docs/EXPERIMENTATION.md)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from jev_api.schemas.base import UtcDatetime

KEY_PATTERN = r"^[a-z][a-z0-9_-]{2,63}$"
VARIANT_PATTERN = r"^[a-z][a-z0-9_-]{0,39}$"

PrimaryMetric = Literal[
    "interaction_rate", "positive_rate", "rating_rate", "feedback_rate", "ndcg_at_10", "diversity", "novelty"
]
GuardrailMetric = Literal[
    "negative_rate",
    "latency_p95_ms",
    "interaction_rate",
    "positive_rate",
    "rating_rate",
    "feedback_rate",
    "ndcg_at_10",
    "diversity",
    "novelty",
    "coverage",
]
ExperimentStatus = Literal["draft", "running", "paused", "stopped", "concluded"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VariantConfig(_Strict):
    """What a variant changes in the /recommendations serving path. Empty = the champion as served
    today. Validated against the live HybridConfig fields and the model registry at create time."""

    # HybridConfig field overrides, e.g. {"diversity_lambda": 0.7} or {"cold_stages": [...]}
    hybrid_overrides: dict[str, Any] = Field(default_factory=dict, max_length=20)
    # a registered model version (challenger); null = the active (champion) model
    model_version: str | None = Field(None, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.:-]+$")
    # the member recommendation_strategy decision on (as today) or off (served standard, no decision)
    strategy_decision: bool = True
    # profile recency decay half-life (days) when the strategy did not set one
    recency_half_life_days: float | None = Field(None, gt=0, le=36500)


class VariantIn(_Strict):
    name: str = Field(pattern=VARIANT_PATTERN)
    is_control: bool = False
    weight: float = Field(1.0, gt=0, le=1000)
    description: str = Field("", max_length=300)
    config: VariantConfig = Field(default_factory=VariantConfig)


class GuardrailIn(_Strict):
    """negative_rate / other rates: max_increase or max_decrease (absolute, treatment - control).
    latency_p95_ms: max_ratio (treatment / control), ignored below min_absolute_ms of extra latency.
    ndcg/diversity/novelty/coverage: max_decrease (absolute)."""

    metric: GuardrailMetric
    max_increase: float | None = Field(None, ge=0)
    max_decrease: float | None = Field(None, ge=0)
    max_ratio: float | None = Field(None, gt=1)
    min_absolute_ms: float = Field(5.0, ge=0)


class AnalysisIn(_Strict):
    alpha: float = Field(0.05, gt=0, le=0.2)
    power: float = Field(0.8, ge=0.5, le=0.99)
    mde_relative: float = Field(0.1, gt=0, le=10)  # minimum detectable relative lift (sample-size warning)
    min_users_per_variant: int = Field(100, ge=2, le=10_000_000)


def default_guardrails() -> list[GuardrailIn]:
    return [
        GuardrailIn(metric="negative_rate", max_increase=0.02),
        GuardrailIn(metric="latency_p95_ms", max_ratio=1.5),
    ]


class ExperimentCreate(_Strict):
    key: str = Field(pattern=KEY_PATTERN)
    name: str = Field(min_length=1, max_length=200)
    surface: Literal["recommendations"] = "recommendations"
    hypothesis: str = Field("", max_length=5000)
    primary_metric: PrimaryMetric = "interaction_rate"
    guardrails: list[GuardrailIn] = Field(default_factory=default_guardrails, max_length=10)
    traffic_percent: float = Field(100.0, ge=0, le=100)
    attribution_window_hours: float | None = Field(None, gt=0, le=720)
    analysis: AnalysisIn = Field(default_factory=AnalysisIn)
    variants: list[VariantIn] = Field(min_length=2, max_length=5)


class ExperimentUpdate(_Strict):
    """Draft only: a started experiment's config is immutable (a change means a new experiment)."""

    name: str | None = Field(None, min_length=1, max_length=200)
    hypothesis: str | None = Field(None, max_length=5000)
    primary_metric: PrimaryMetric | None = None
    guardrails: list[GuardrailIn] | None = Field(None, max_length=10)
    traffic_percent: float | None = Field(None, ge=0, le=100)
    attribution_window_hours: float | None = Field(None, gt=0, le=720)
    analysis: AnalysisIn | None = None
    variants: list[VariantIn] | None = Field(None, min_length=2, max_length=5)


class RampIn(_Strict):
    traffic_percent: float = Field(ge=0, le=100)


class VariantOut(BaseModel):
    name: str
    is_control: bool
    weight: float
    description: str
    config: dict[str, Any]
    assigned_users: int = 0


class ExperimentOut(BaseModel):
    id: int
    key: str
    name: str
    surface: str
    status: ExperimentStatus
    hypothesis: str
    primary_metric: str
    guardrails: list[dict[str, Any]]
    traffic_percent: float
    attribution_window_hours: float
    analysis: dict[str, Any]
    data_source: str
    variants: list[VariantOut]
    started_at: UtcDatetime | None
    stopped_at: UtcDatetime | None
    concluded_at: UtcDatetime | None
    created_at: UtcDatetime
    updated_at: UtcDatetime
    created_by: int | None
    allowed_actions: list[str]
    conclusion: dict[str, Any] | None = None


class ExperimentList(BaseModel):
    items: list[ExperimentOut]
    total: int
