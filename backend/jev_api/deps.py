"""FastAPI dependencies: DB session, auth, admin guard, CSRF check, ML engine, cache."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import Depends, HTTPException, Path, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from jev_api.cache import Cache
from jev_api.config import get_settings
from jev_api.db import get_db
from jev_api.models import RevokedToken, ServiceToken, User
from jev_api.security import (
    CLIENT_HEADER,
    decode_access_token,
    is_service_token,
    service_token_hash,
    verify_client_address,
)
from jev_ml.engine import RecommendationEngine

bearer = HTTPBearer(auto_error=False)
CSRF_HEADER = "x-jev-csrf"

DB = Annotated[Session, Depends(get_db)]

# Integer inputs that reach SQL are bounded. Every primary key is an INTEGER column (32-bit on
# PostgreSQL); a larger value overflows the driver (a 500 on SQLite) instead of matching nothing.
MAX_DB_INT = 2**31 - 1
MAX_OFFSET = 1_000_000
MAX_PAGE = 10_000  # page numbers (page_size <= 200)
MAX_COUNT = 10**9  # rating-count filters
IdPath = Annotated[int, Path(ge=0, le=MAX_DB_INT)]


def client_address(request: Request) -> str:
    """The client address used for rate limiting and audit rows.

    X-Forwarded-For is never parsed here: its leftmost entries are whatever the client sent. Behind a
    reverse proxy, run uvicorn with `--proxy-headers --forwarded-allow-ips <proxy address>`; uvicorn
    then replaces `request.client` with the right-most untrusted hop, and only for connections that
    come from the listed proxy. A direct client's headers are ignored.

    With JEV_SECURITY_PROXY_SECRET set, the web proxy may instead send X-JEV-Client, the client IP signed
    with HMAC-SHA256 and a timestamp; it is used only when the signature verifies (security.py)."""
    signed = signed_client_address(request)
    if signed is not None:
        return signed
    return peer_address(request)


def peer_address(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def signed_client_address(request: Request) -> str | None:
    s = get_settings()
    if s.security_proxy_secret is None or not s.security_proxy_secret.get_secret_value():
        return None
    return verify_client_address(
        request.headers.get(CLIENT_HEADER),
        s.security_proxy_secret.get_secret_value(),
        s.security_proxy_max_skew_seconds,
    )


def parse_db_id(ref: str) -> int | None:
    """A numeric database id from a path/body string, or None. ASCII digits only
    (str.isdigit() also accepts "²" and other digits int() rejects) and within MAX_DB_INT."""
    if not (ref.isascii() and ref.isdigit()) or len(ref) > 10:
        return None
    value = int(ref)
    return value if value <= MAX_DB_INT else None


def _token_from_request(
    request: Request, creds: HTTPAuthorizationCredentials | None
) -> tuple[str | None, str]:
    if creds and creds.scheme.lower() == "bearer":
        return creds.credentials, "bearer"
    cookie = request.cookies.get(get_settings().cookie_name)
    return (cookie, "cookie") if cookie else (None, "none")


def get_optional_user(
    request: Request, db: DB, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
) -> User | None:
    token, source = _token_from_request(request, creds)
    if not token:
        return None
    payload = decode_access_token(token)
    if payload is None:
        return None
    # Cookie-authenticated state changes must carry a custom header. Cross-site forms cannot set
    # one, and the CORS preflight blocks it for other origins, which closes the CSRF hole.
    if (
        source == "cookie"
        and request.method not in ("GET", "HEAD", "OPTIONS")
        and request.headers.get(CSRF_HEADER) != "1"
    ):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "missing CSRF header")
    user = db.get(User, int(payload["sub"]))
    if user is None or payload["ver"] != user.token_version:
        return None  # revoke-all, password change or role change since the token was issued
    if db.get(RevokedToken, payload["jti"]) is not None:
        return None  # this session was logged out
    request.state.token_claims = payload
    return user


def get_current_user(user: Annotated[User | None, Depends(get_optional_user)]) -> User:
    if user is None:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "not authenticated", headers={"WWW-Authenticate": "Bearer"}
        )
    return user


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if not user.is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "admin only")
    return user


# --- service tokens -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Writer:
    """Who wrote something through a machine-or-admin endpoint: the audit actor and an idempotency scope
    that cannot collide between members and tokens."""

    actor: User | str
    scope: str
    user: User | None = None
    token: ServiceToken | None = None


def get_service_token(
    db: DB, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]
) -> ServiceToken | None:
    """A presented service token (Authorization: Bearer jevst_...). None when no service token is
    presented; 401 when one is presented but unknown, expired or revoked."""
    if creds is None or creds.scheme.lower() != "bearer" or not is_service_token(creds.credentials):
        return None
    row = db.scalar(
        select(ServiceToken).where(ServiceToken.token_hash == service_token_hash(creds.credentials))
    )
    now = datetime.now(UTC)
    if row is None or not row.active(now):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "invalid service token", headers={"WWW-Authenticate": "Bearer"}
        )
    row.last_used_at = now  # committed with the request's own writes
    return row


ServiceTokenDep = Annotated[ServiceToken | None, Depends(get_service_token)]


def writer_for(scope: str, user: User | None, token: ServiceToken | None) -> Writer:
    """A service token with `scope`, or an admin user. Members get 403, anonymous 401."""
    if token is not None:
        if not token.allows(scope):
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"service token lacks scope {scope!r}")
        return Writer(
            actor=f"service-token:{token.id}:{token.name}"[:320], scope=f"token:{token.id}", token=token
        )
    admin = require_admin(get_current_user(user))
    return Writer(actor=admin, scope=f"user:{admin.id}", user=admin)


def token_claims(request: Request) -> dict[str, Any] | None:
    claims: dict[str, Any] | None = getattr(request.state, "token_claims", None)
    return claims


def get_engine(request: Request) -> RecommendationEngine:
    engine: RecommendationEngine | None = request.app.state.engines.engine
    if engine is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "recommendation model not loaded; train a model first"
        )
    return engine


def get_cache(request: Request) -> Cache:
    cache: Cache = request.app.state.cache
    return cache


CurrentUser = Annotated[User, Depends(get_current_user)]
OptionalUser = Annotated[User | None, Depends(get_optional_user)]
AdminUser = Annotated[User, Depends(require_admin)]
Engine = Annotated[RecommendationEngine, Depends(get_engine)]
CacheDep = Annotated[Cache, Depends(get_cache)]
