"""Request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- auth / users ------------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=80)

    @field_validator("display_name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("display name must not be blank")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"  # noqa: S105 - OAuth token type, not a secret
    expires_in: int
    user: UserOut


class UserOut(ORM):
    id: int
    email: str
    display_name: str
    is_admin: bool
    onboarding_completed: bool
    favorite_genres: list[str] = []
    created_at: datetime


class OnboardingRequest(BaseModel):
    genres: list[str] = Field(min_length=1, max_length=20)
    movie_ids: list[int] = Field(default_factory=list, max_length=50)


class PreferenceUpdate(BaseModel):
    genres: list[str] | None = Field(default=None, max_length=20)
    diversity: Literal["focused", "balanced", "adventurous"] | None = None


# --- movies ------------------------------------------------------------------------------------
class MovieBrief(ORM):
    id: int
    title: str
    year: int | None
    genres: list[str]
    directors: list[str] = []
    n_ratings: int
    mean_rating: float | None

    @field_validator("genres", mode="before")
    @classmethod
    def _genre_names(cls, v: Any) -> Any:
        return [g.name if hasattr(g, "name") else g for g in v]


class MovieDetail(MovieBrief):
    description: str | None
    runtime_min: int | None
    cast: list[str]
    keywords: list[str]
    tags: list[str]
    countries: list[str]
    imdb_id: str | None
    tmdb_id: int | None
    user_rating: float | None = None
    is_favorite: bool = False
    watched: bool = False
    in_model: bool = True


class MoviePage(BaseModel):
    items: list[MovieBrief]
    total: int
    page: int
    page_size: int


class RateRequest(BaseModel):
    rating: float = Field(ge=0.5, le=5.0)

    @field_validator("rating")
    @classmethod
    def _half_steps(cls, v: float) -> float:
        if (v * 2) % 1 != 0:
            raise ValueError("rating must be a multiple of 0.5")
        return v


class FavoriteRequest(BaseModel):
    favorite: bool = True


class InteractionResult(BaseModel):
    movie_id: int
    rating: float | None = None
    is_favorite: bool | None = None
    watched: bool | None = None
    profile_version: int


class MovieCreate(BaseModel):
    """Admin: add a movie that is not in the trained model (item cold start)."""

    id: int = Field(gt=0)
    title: str = Field(min_length=1, max_length=300)
    year: int | None = Field(default=None, ge=1870, le=2100)
    genres: list[str] = Field(default_factory=list, max_length=20)
    directors: list[str] = Field(default_factory=list, max_length=20)
    cast: list[str] = Field(default_factory=list, max_length=50)
    keywords: list[str] = Field(default_factory=list, max_length=50)
    description: str | None = Field(default=None, max_length=2000)


# --- recommendations -------------------------------------------------------------------------------
class SignalValue(BaseModel):
    raw: float
    normalized: float
    weight: float
    contribution: float


class RecommendationItem(BaseModel):
    recommendation_id: int | None = None
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = []
    directors: list[str] = []
    score: float
    rank: int
    reason: str
    reason_code: str
    secondary_reasons: list[str] = []
    anchor_movie_ids: list[int] = []
    signals: dict[str, SignalValue]


class RecommendationResponse(BaseModel):
    items: list[RecommendationItem]
    model_version: str
    generated_at: datetime
    request_id: str
    limit: int
    offset: int
    effective_weights: dict[str, float]
    profile: dict[str, int]
    cached: bool = False


class SimpleRecItem(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = []
    directors: list[str] = []
    score: float
    reason: str | None = None
    signals: dict[str, float] = {}


class SimpleRecResponse(BaseModel):
    items: list[SimpleRecItem]
    model_version: str
    generated_at: datetime
    anchor: MovieBrief | None = None
    anchor_kind: Literal["watched", "rated"] | None = None


class FeedbackRequest(BaseModel):
    movie_id: int
    feedback: Literal["like", "dislike", "not_interested", "clicked"]
    recommendation_id: int | None = None


class FeedbackOut(ORM):
    id: int
    movie_id: int
    feedback: str
    recommendation_id: int | None
    created_at: datetime


class RecommendationHistoryItem(ORM):
    id: int
    movie_id: int
    title: str
    request_id: str
    model_version: str
    context: str
    rank: int
    score: float
    reason: str
    created_at: datetime
    feedback: str | None = None


# --- models / experiments --------------------------------------------------------------------------
class ModelVersionOut(ORM):
    id: int
    version: str
    model_type: str
    dataset_version: str
    training_seed: int
    metrics: dict[str, Any]
    artifact_path: str
    artifact_bytes: int
    trained_at: datetime
    is_active: bool


class ModelVersionDetail(ModelVersionOut):
    training_config: dict[str, Any]
    manifest: dict[str, Any] = {}


class MetricOut(ORM):
    model_name: str
    protocol: str
    metric: str
    k: int
    value: float


class ExperimentOut(ORM):
    id: int
    run_id: str
    experiment_name: str
    dataset_version: str
    model_type: str
    training_seed: int
    created_at: datetime
    model_version: str | None = None
    headline: dict[str, float] = {}

    @field_validator("model_version", mode="before")
    @classmethod
    def _version_name(cls, v: Any) -> Any:
        return getattr(v, "version", v)


class ExperimentDetail(ExperimentOut):
    parameters: dict[str, Any]
    split: dict[str, Any]
    summary: dict[str, Any]
    metrics: list[MetricOut]


# --- intelligence layer (docs/intelligence.md, section 6) ------------------------------------------
# Run-backed objects (signals, trends, ...) are passed through from PipelineResult.to_dict(), whose
# contract lives in ml/jev_ml/intel; DB-backed objects are typed here.
WarningStatus = Literal["new", "acknowledged", "investigating", "resolved", "dismissed"]
Severity = Literal["low", "medium", "high", "critical"]
FeedbackTarget = Literal["decision", "warning", "action", "prediction"]
Verdict = Literal["correct", "incorrect", "useful", "not_useful", "false_positive"]


class IntelPage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int
    run_id: str | None
    as_of: str | None


class RunTrigger(BaseModel):
    # "YYYY-MM-DD" (UTC midnight) or a full ISO timestamp; omitted = the last MovieLens event
    as_of: str | None = Field(default=None, max_length=40)


class IntelRunOut(BaseModel):
    id: int
    run_id: str
    trigger: str
    status: str
    as_of: str | None
    started_at: str
    finished_at: str | None
    duration_ms: float | None
    pipeline_version: str
    data_version: str | None
    model_version: str | None
    summary: dict[str, Any] | None
    stage_ms: dict[str, Any]
    error: str | None


class IntelRunList(BaseModel):
    items: list[IntelRunOut]
    total: int


class WarningEventOut(BaseModel):
    from_status: str | None
    to_status: str
    note: str | None
    actor: str
    at: str


class IntelWarningOut(BaseModel):
    id: int
    key: str
    title: str
    description: str
    severity: str
    confidence: float | None
    confidence_kind: str | None
    status: str
    trigger: dict[str, Any]
    evidence: list[Any]
    recommended_action: str
    source: dict[str, Any]
    entity_type: str | None
    entity: str | None
    detected_at: str
    last_seen_at: str
    updated_at: str
    occurrences: int
    first_seen_run_id: str | None
    last_seen_run_id: str | None
    reopened_from: int | None
    suppressed_until: str | None = None
    history: list[WarningEventOut] | None = None


class IntelWarningList(BaseModel):
    items: list[IntelWarningOut]
    total: int
    limit: int
    offset: int


class WarningUpdate(BaseModel):
    status: WarningStatus
    note: str | None = Field(default=None, max_length=1000)


class DecisionFeedbackCounts(BaseModel):
    correct: int = 0
    incorrect: int = 0


class IntelDecisionOut(BaseModel):
    id: str
    key: str
    spec_id: str
    policy_version: str
    question: str
    kind: str
    options: list[str]
    answer: str | None
    option_scores: dict[str, Any]
    confidence: float | None
    confidence_kind: str
    state: dict[str, Any]
    rationale: list[str]
    evidence: list[Any]
    abstained: bool
    fallback_reason: str | None
    entity_type: str | None
    entity: str | None
    db_id: int
    run_id: str
    as_of: str
    created_at: str
    feedback: DecisionFeedbackCounts


class IntelDecisionList(BaseModel):
    items: list[IntelDecisionOut]
    total: int
    limit: int
    offset: int


class IntelFeedbackIn(BaseModel):
    target_type: FeedbackTarget
    target_id: str = Field(min_length=1, max_length=64)
    verdict: Verdict
    note: str | None = Field(default=None, max_length=1000)
    outcome: str | None = Field(default=None, max_length=500)


class IntelFeedbackOut(BaseModel):
    id: int
    target_type: str
    target_id: str
    verdict: str
    note: str | None
    outcome: str | None
    actor: str
    created_at: str


class IntelFeedbackList(BaseModel):
    items: list[IntelFeedbackOut]
    total: int
    limit: int
    offset: int
    summary: dict[str, dict[str, Any]]


class ScenarioSpecIn(BaseModel):
    # value ranges are validated by run_scenario (its messages come back as 422)
    model_config = ConfigDict(extra="allow")

    name: str | None = Field(default=None, max_length=80)
    kind: str = Field(max_length=16)
    trend_multiplier: float | None = None
    level_shift_pct: float | None = None
    shock_month: int | None = None


class ScenarioRequest(BaseModel):
    series_id: str = Field(min_length=1, max_length=200)
    horizon_months: int = 12
    as_of: str | None = Field(default=None, max_length=40)
    scenarios: list[ScenarioSpecIn] = Field(default_factory=list, max_length=6)
    trend_source: Literal["auto", "model"] | None = None
    save: bool = False
    title: str | None = Field(default=None, max_length=200)


class SavedScenarioOut(BaseModel):
    id: int
    title: str
    series_id: str
    created_at: str
    input: dict[str, Any]
    output: dict[str, Any]


class SavedScenarioList(BaseModel):
    items: list[SavedScenarioOut]
    total: int


TokenResponse.model_rebuild()
