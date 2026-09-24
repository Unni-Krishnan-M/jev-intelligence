"""Recommendation serving and feedback schemas (WS5)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from jev_api.schemas.base import ORM, DbId
from jev_api.schemas.movies import MovieBrief


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
    # calibrated P(rating >= 4) (docs/intelligence.md, section 9.2); null without a calibration
    confidence: float | None = None
    confidence_kind: Literal["probability"] | None = None
    # v1.2: the recommendation_strategy decision the item was served under
    decision_id: str | None = None
    strategy: str | None = None


class RecIntelligence(BaseModel):
    """GET /recommendations -> intelligence (docs/platform.md, section 10)."""

    decision_id: str
    # the decision's answer; null when it abstained (then `served_strategy` is "standard")
    strategy: str | None
    served_strategy: str
    confidence: float | None
    confidence_kind: str
    abstained: bool
    drift_detected: bool | None
    summary: str | None
    policy_version: str | None = None
    evidence: list[dict[str, Any]] = []


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
    intelligence: RecIntelligence | None = None
    # Phase 2: request_id is fresh on every response; a cache hit names the request that generated the
    # list (its recommendation_ids belong to that request). Null when the list was generated now.
    source_request_id: str | None = None


class SimpleRecItem(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = []
    directors: list[str] = []
    score: float
    reason: str | None = None
    signals: dict[str, float] = {}
    confidence: float | None = None
    confidence_kind: Literal["probability"] | None = None


class SimpleRecResponse(BaseModel):
    items: list[SimpleRecItem]
    model_version: str
    generated_at: datetime
    anchor: MovieBrief | None = None
    anchor_kind: Literal["watched", "rated"] | None = None


class FeedbackRequest(BaseModel):
    movie_id: DbId
    feedback: Literal["like", "dislike", "not_interested", "clicked"]
    recommendation_id: DbId | None = None


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
    confidence: float | None = None
    confidence_kind: str | None = None
