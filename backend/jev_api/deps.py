"""FastAPI dependencies: DB session, auth, admin guard, CSRF check, ML engine, cache."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Path, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from jev_api.cache import Cache
from jev_api.config import get_settings
from jev_api.db import get_db
from jev_api.models import User
from jev_api.security import decode_access_token
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
    come from the listed proxy. A direct client's headers are ignored."""
    return request.client.host if request.client else "unknown"


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
