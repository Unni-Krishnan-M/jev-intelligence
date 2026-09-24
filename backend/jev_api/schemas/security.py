"""Service token and session revocation schemas (security hardening, docs/SECURITY_AUDIT_PHASE2.md)."""

from __future__ import annotations

import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

# events:ingest:<generic domain key> or events:ingest:* (every domain)
SCOPE_PATTERN = r"^events:ingest:(\*|generic:[a-z0-9][a-z0-9:_-]{0,55})$"


class PasswordChange(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class ServiceTokenCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9 ._-]*$")
    scopes: list[str] = Field(min_length=1, max_length=20)
    expires_in_days: int = Field(90, ge=1, le=3650)

    @field_validator("scopes")
    @classmethod
    def _scopes(cls, v: list[str]) -> list[str]:
        bad = [s for s in v if not re.fullmatch(SCOPE_PATTERN, s)]
        if bad:
            raise ValueError("scopes must be events:ingest:<generic domain key> or events:ingest:*")
        return sorted(set(v))


class ServiceTokenOut(BaseModel):
    id: int
    name: str
    token_prefix: str
    scopes: list[str]
    created_by: str
    created_at: datetime
    expires_at: datetime
    revoked_at: datetime | None
    revoked_by: str | None
    last_used_at: datetime | None
    active: bool


class ServiceTokenCreated(ServiceTokenOut):
    token: str  # the plaintext token: shown in this response only, never again
