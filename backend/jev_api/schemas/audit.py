"""Audit log schemas (GET /admin/audit)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AuditEntryOut(BaseModel):
    id: int
    at: str
    actor_user_id: int | None
    actor: str
    action: str
    target_type: str | None
    target_id: str | None
    detail: dict[str, Any]
    request_id: str | None


class AuditPage(BaseModel):
    items: list[AuditEntryOut]
    total: int
    limit: int
    offset: int
