"""Security administration (admin only; docs/SECURITY_AUDIT_PHASE2.md).

POST   /admin/service-tokens                  create a scoped, expiring token: the plaintext is returned once
GET    /admin/service-tokens                  list tokens (metadata only; never the token or its hash)
DELETE /admin/service-tokens/{token_id}       revoke a token (immediate)
POST   /admin/users/{user_id}/revoke-sessions end every session of an account (incident response)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from sqlalchemy import select

from jev_api.deps import DB, AdminUser, IdPath
from jev_api.models import ServiceToken, User
from jev_api.schemas.security import ServiceTokenCreate, ServiceTokenCreated, ServiceTokenOut
from jev_api.security import new_service_token, service_token_hash
from jev_api.services import audit

router = APIRouter(prefix="/admin", tags=["security"])


def _aware(dt: datetime | None) -> datetime | None:
    return None if dt is None else (dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt)


def token_out(t: ServiceToken) -> dict[str, Any]:
    return {
        "id": t.id,
        "name": t.name,
        "token_prefix": t.token_prefix,
        "scopes": list(t.scopes or []),
        "created_by": t.created_by,
        "created_at": _aware(t.created_at),
        "expires_at": _aware(t.expires_at),
        "revoked_at": _aware(t.revoked_at),
        "revoked_by": t.revoked_by,
        "last_used_at": _aware(t.last_used_at),
        "active": t.active(datetime.now(UTC)),
    }


@router.post("/service-tokens", response_model=ServiceTokenCreated, status_code=status.HTTP_201_CREATED)
def create_service_token(
    body: ServiceTokenCreate, user: AdminUser, db: DB, request: Request
) -> dict[str, Any]:
    max_days = request.app.state.settings.security_service_token_max_days
    if body.expires_in_days > max_days:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"expires_in_days is at most {max_days}")
    token = new_service_token()
    now = datetime.now(UTC)
    row = ServiceToken(
        name=body.name,
        token_prefix=token[:12],
        token_hash=service_token_hash(token),
        scopes=body.scopes,
        created_by_user_id=user.id,
        created_by=user.email,
        created_at=now,
        expires_at=now + timedelta(days=body.expires_in_days),
    )
    db.add(row)
    db.flush()
    # never the token or its hash
    audit.record(
        db,
        "token.create",
        user,
        "service_token",
        row.id,
        {"name": row.name, "scopes": row.scopes, "expires_at": row.expires_at.isoformat()},
    )
    db.commit()
    db.refresh(row)
    return {**token_out(row), "token": token}


@router.get("/service-tokens", response_model=list[ServiceTokenOut])
def list_service_tokens(_: AdminUser, db: DB) -> list[dict[str, Any]]:
    rows = db.scalars(select(ServiceToken).order_by(ServiceToken.id.desc()).limit(500))
    return [token_out(t) for t in rows]


@router.delete("/service-tokens/{token_id}", response_model=ServiceTokenOut)
def revoke_service_token(token_id: IdPath, user: AdminUser, db: DB) -> dict[str, Any]:
    row = db.get(ServiceToken, token_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "service token not found")
    if row.revoked_at is None:
        row.revoked_at = datetime.now(UTC)
        row.revoked_by = user.email
        audit.record(db, "token.revoke", user, "service_token", row.id, {"name": row.name})
        db.commit()
        db.refresh(row)
    return token_out(row)


@router.post("/users/{user_id}/revoke-sessions", status_code=status.HTTP_204_NO_CONTENT)
def revoke_user_sessions(user_id: IdPath, user: AdminUser, db: DB) -> None:
    target = db.get(User, user_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    target.token_version += 1
    audit.record(db, "auth.revoke_all", user, "user", target.id, {"by": "admin"})
    db.commit()
