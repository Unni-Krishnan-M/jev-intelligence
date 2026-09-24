"""Auth and account schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from jev_api.schemas.base import ORM, DbId


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
    movie_ids: list[DbId] = Field(default_factory=list, max_length=50)


class PreferenceUpdate(BaseModel):
    genres: list[str] | None = Field(default=None, max_length=20)
    diversity: Literal["focused", "balanced", "adventurous"] | None = None


TokenResponse.model_rebuild()
