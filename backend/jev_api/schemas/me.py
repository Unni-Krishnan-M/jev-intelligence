"""Per-member intelligence schemas (docs/platform.md, sections 8 and 10). Owned by WS5."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MeScenarioSpec(BaseModel):
    # kind-specific factor ranges are validated by user_preference_scenarios (ValueError -> 422)
    model_config = ConfigDict(extra="ignore")  # unknown keys are dropped, never stored

    name: str | None = Field(default=None, max_length=80)
    kind: Literal["continue", "accelerate", "reverse"]
    factor: float | None = Field(default=None, allow_inf_nan=False, ge=-100, le=100)


class MeScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # unknown keys are dropped, never stored

    k: int = Field(default=10, ge=1, le=50)
    scenarios: list[MeScenarioSpec] = Field(default_factory=list, max_length=4)


class MeFeedbackIn(BaseModel):
    model_config = ConfigDict(extra="ignore")  # unknown keys are dropped, never stored

    target_type: Literal["strategy", "recommendation"]
    # a strategy decision id ("dec-...") or a recommended item (movie) id
    target_id: str = Field(min_length=1, max_length=64)
    verdict: Literal["accepted", "rejected"]
    note: str | None = Field(default=None, max_length=1000)


class MeFeedbackOut(BaseModel):
    id: int
    target_type: str
    target_id: str
    verdict: str
    note: str | None = None
    decision_id: str | None = None
    created_at: str
    updated_at: str | None = None
