from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from typing import NoReturn

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from jev_api.config import get_settings
from jev_api.deps import DB, CurrentUser, OptionalUser, client_address, token_claims
from jev_api.models import RevokedToken, User
from jev_api.routers.users import user_out
from jev_api.schemas import LoginRequest, RegisterRequest, TokenResponse
from jev_api.schemas.security import PasswordChange
from jev_api.security import create_access_token, hash_password, needs_rehash, verify_password
from jev_api.services import audit

router = APIRouter(prefix="/auth", tags=["auth"])

# Verified against when the email is unknown, so response time does not reveal which accounts exist.
_DUMMY_HASH = hash_password("jev-timing-equalizer")


def _client(request: Request) -> str:
    return client_address(request)  # the same address the rate limiter keys on


def _issue(response: Response, user: User) -> TokenResponse:
    s = get_settings()
    token, expires = create_access_token(user.id, user.is_admin, user.token_version)
    response.set_cookie(
        s.cookie_name, token, max_age=expires, httponly=True, secure=s.cookie_secure, samesite="lax", path="/"
    )
    return TokenResponse(access_token=token, expires_in=expires, user=user_out(user))


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, request: Request, response: Response, db: DB) -> TokenResponse:
    email = body.email.lower()
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with this email already exists")
    user = User(email=email, password_hash=hash_password(body.password), display_name=body.display_name)
    db.add(user)
    try:
        db.flush()
        # audited with the account it creates; never the password
        audit.record(db, "auth.register", user, "user", user.id, {"client": _client(request)})
        db.commit()
    except IntegrityError as exc:  # concurrent registration with the same email
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with this email already exists") from exc
    db.refresh(user)
    return _issue(response, user)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, request: Request, response: Response, db: DB) -> TokenResponse:
    email = body.email.lower()
    _check_account_throttle(db, request, email)
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        verify_password(body.password, _DUMMY_HASH)
        _count_failure(request, email)
        _login_failed(db, email, "unknown_email", _client(request))
    if not verify_password(body.password, user.password_hash):
        _count_failure(request, email)
        _login_failed(db, email, "wrong_password", _client(request))
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
    audit.record(db, "auth.login.success", user, "user", user.id, {"client": _client(request)})
    db.commit()
    return _issue(response, user)


def _login_failed(db: DB, email: str, reason: str, client: str) -> NoReturn:
    # actor = the email that was attempted (the password is never recorded); the response stays generic
    audit.record(
        db, "auth.login.failure", email, "user", None, {"reason": reason, "client": client}, commit=True
    )
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")


# --- per-account login throttle ----------------------------------------------------------------------
# Failed logins are counted per email (hashed, so the cache holds no addresses) in a fixed window,
# independently of the client address: behind the web proxy every client shares one address, and a
# client-chosen address must not reset the count either. Unknown emails are counted too, so the
# throttle does not reveal which accounts exist.
def _account_key(request: Request, email: str) -> str:
    s = get_settings()
    window = int(time.time() // s.security_login_window_seconds)
    digest = hashlib.sha256(email.encode()).hexdigest()[:32]
    return f"rl:login-acct:{digest}:{window}"


def _window_left() -> int:
    w = get_settings().security_login_window_seconds
    return w - int(time.time()) % w


def _check_account_throttle(db: DB, request: Request, email: str) -> None:
    s = get_settings()
    cache = request.app.state.cache
    failures = cache.get(_account_key(request, email)) or 0
    if int(failures) < s.security_login_max_failures:
        return
    # audited once per account per window, so a flood cannot grow the audit log without bound
    if cache.incr_window(_account_key(request, email) + ":audited", s.security_login_window_seconds) == 1:
        audit.record(
            db,
            "auth.login.throttled",
            email,
            "user",
            None,
            {"client": _client(request), "failures": int(failures)},
            commit=True,
        )
    raise HTTPException(
        status.HTTP_429_TOO_MANY_REQUESTS,
        "too many failed sign-in attempts for this account; try again later",
        headers={"Retry-After": str(_window_left())},
    )


def _count_failure(request: Request, email: str) -> None:
    request.app.state.cache.incr_window(
        _account_key(request, email), get_settings().security_login_window_seconds
    )


# --- session lifecycle -------------------------------------------------------------------------------
def _clear_cookie(response: Response) -> Response:
    response.delete_cookie(get_settings().cookie_name, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, response: Response, db: DB, user: OptionalUser) -> Response:
    """Ends this session: its jti is denylisted until the token expires, and the cookie is cleared.
    Anonymous (or already invalid) callers just get the cookie cleared."""
    claims = token_claims(request)
    if user is not None and claims is not None:
        db.merge(
            RevokedToken(
                jti=claims["jti"],
                user_id=user.id,
                revoked_at=datetime.now(UTC),
                expires_at=datetime.fromtimestamp(int(claims["exp"]), UTC),
            )
        )
        audit.record(db, "auth.logout", user, "user", user.id, {"client": _client(request)})
        db.commit()
    return _clear_cookie(response)


@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
def logout_all(request: Request, response: Response, db: DB, user: CurrentUser) -> Response:
    """Ends every session of the account (all devices) by bumping its token_version."""
    user.token_version += 1
    audit.record(db, "auth.revoke_all", user, "user", user.id, {"client": _client(request), "by": "self"})
    db.commit()
    return _clear_cookie(response)


@router.post("/password", response_model=TokenResponse)
def change_password(
    body: PasswordChange, request: Request, response: Response, db: DB, user: CurrentUser
) -> TokenResponse:
    """Changes the password and ends every other session; the caller gets a fresh token."""
    _check_account_throttle(db, request, user.email)
    if not verify_password(body.current_password, user.password_hash):
        _count_failure(request, user.email)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "current password is incorrect")
    user.password_hash = hash_password(body.new_password)
    user.token_version += 1
    audit.record(db, "auth.password_change", user, "user", user.id, {"client": _client(request)})
    db.commit()
    db.refresh(user)
    return _issue(response, user)
