"""Event ingestion request/response schemas (WS1, docs/STREAMING_ARCHITECTURE.md).

Structure is validated here (a malformed batch is a 422 as a whole); per-item facts that need the
database (unknown movie, someone else's recommendation, a reused key, a timestamp out of range) come
back per item as ``rejected`` with a reason.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jev_api.models.enums import REC_FEEDBACK_KINDS

# printable ASCII without spaces, like the IETF Idempotency-Key draft's opaque token
IDEMPOTENCY_KEY_PATTERN = r"^[\x21-\x7e]{1,200}$"
MAX_DB_INT = 2**31 - 1

MemberEventType = Literal["rating", "rating_removed", "watch", "favorite", "unfavorite", "rec_feedback"]
ItemStatus = Literal["accepted", "duplicate", "rejected"]


class MemberEventIn(BaseModel):
    """One interaction of the authenticated member. ``user_id`` may be omitted; when given it must be
    the caller (another member's id is a 403 for the whole request)."""

    model_config = ConfigDict(extra="forbid")

    event_type: MemberEventType
    movie_id: int = Field(ge=0, le=MAX_DB_INT)
    idempotency_key: str = Field(min_length=1, max_length=200, pattern=IDEMPOTENCY_KEY_PATTERN)
    event_time: datetime | None = Field(
        None, description="when it happened (ISO-8601; naive = UTC). Default: the time of ingestion."
    )
    value: float | None = Field(None, description="rating: 0.5-5.0 in half steps")
    feedback: Literal["like", "dislike", "not_interested", "clicked"] | None = None
    recommendation_id: int | None = Field(None, ge=0, le=MAX_DB_INT)
    user_id: int | None = Field(None, ge=0, le=MAX_DB_INT)

    @model_validator(mode="after")
    def _shape(self) -> MemberEventIn:
        if self.event_type == "rating":
            if self.value is None:
                raise ValueError("a rating event needs a value")
            if not 0.5 <= self.value <= 5.0 or (self.value * 2) % 1 != 0:
                raise ValueError("rating value must be 0.5-5.0 in steps of 0.5")
        elif self.value is not None:
            raise ValueError(f"a {self.event_type} event takes no value")
        if self.event_type == "rec_feedback":
            if self.feedback is None or self.feedback not in REC_FEEDBACK_KINDS:
                raise ValueError("a rec_feedback event needs feedback")
        elif self.feedback is not None or self.recommendation_id is not None:
            raise ValueError("feedback and recommendation_id belong to rec_feedback events")
        return self


class EventBatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[MemberEventIn] = Field(min_length=1)


class ObservationIn(BaseModel):
    """One value of a generic domain's entity at a period (``event_time`` = the period start, midnight UTC;
    the first of the month for a monthly domain)."""

    model_config = ConfigDict(extra="forbid")

    entity: str = Field(min_length=1, max_length=128)
    value: float
    event_time: datetime
    idempotency_key: str = Field(min_length=1, max_length=200, pattern=IDEMPOTENCY_KEY_PATTERN)
    attributes: dict[str, str | float | int | None] = Field(
        default_factory=dict, description="group / calendar columns of the domain (default: the entity's)"
    )

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return v


class ObservationBatchIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observations: list[ObservationIn] = Field(min_length=1)


class IngestItemOut(BaseModel):
    index: int
    status: ItemStatus
    event_id: str | None = None
    seq: int | None = None  # events.id: the log's sequence number
    reason: str | None = None


class IngestResultOut(BaseModel):
    batch_id: str
    domain: str
    accepted: int
    duplicates: int
    rejected: int
    items: list[IngestItemOut]


class ReplayIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apply: bool = Field(False, description="rewrite the projections from the log (default: verify only)")


class ReplayOut(BaseModel):
    applied: bool
    consistent_before: bool
    consistent_after: bool
    diff: dict[str, Any]


class EventHealthOut(BaseModel):
    generated_at: str
    domains: list[dict[str, Any]]
    refresher: dict[str, Any]
