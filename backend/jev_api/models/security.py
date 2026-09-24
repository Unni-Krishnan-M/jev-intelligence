"""Service tokens and session revocation (security hardening, integrator; migration 0010).

- ``service_tokens``: machine credentials for ingestion. Only the SHA-256 of the token is stored; the
  plaintext is shown once, at creation. Each token has scopes (``events:ingest:<domain>``), an expiry and
  an optional revocation.
- ``revoked_tokens``: the jti denylist for sessions ended by logout. Rows are only needed until the JWT
  itself expires (``expires_at``), after which they can be pruned.
- ``users.token_version`` (models/users.py): bumped by revoke-all, a password change or a role change;
  every JWT carries the version it was issued under.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from jev_api.models.base import Base, utcnow


def _aware(dt: datetime | None) -> datetime | None:
    return None if dt is None else (dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt)


class ServiceToken(Base):
    __tablename__ = "service_tokens"
    __table_args__ = (UniqueConstraint("token_hash", name="uq_service_tokens_token_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    # the first characters of the token, so operators can tell tokens apart (not a secret on its own)
    token_prefix: Mapped[str] = mapped_column(String(16), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    scopes: Mapped[list[Any]] = mapped_column(JSON, default=list, nullable=False)
    created_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"))
    created_by: Mapped[str] = mapped_column(String(320), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_by: Mapped[str | None] = mapped_column(String(320))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def active(self, now: datetime) -> bool:
        exp = _aware(self.expires_at)
        return self.revoked_at is None and exp is not None and exp > now

    def allows(self, scope: str) -> bool:
        """Exact scope, or the family wildcard (``events:ingest:*`` covers every domain)."""
        family = ":".join(scope.split(":")[:2])
        return scope in self.scopes or f"{family}:*" in self.scopes


class RevokedToken(Base):
    __tablename__ = "revoked_tokens"

    jti: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    revoked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True, nullable=False)
