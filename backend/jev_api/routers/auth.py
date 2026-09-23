from __future__ import annotations

from typing import NoReturn

from fastapi import APIRouter, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from jev_api.config import get_settings
from jev_api.deps import DB, client_address
from jev_api.models import User
from jev_api.routers.users import user_out
from jev_api.schemas import LoginRequest, RegisterRequest, TokenResponse
from jev_api.security import create_access_token, hash_password, needs_rehash, verify_password
from jev_api.services import audit

router = APIRouter(prefix="/auth", tags=["auth"])

# Verified against when the email is unknown, so response time does not reveal which accounts exist.
_DUMMY_HASH = hash_password("jev-timing-equalizer")


def _client(request: Request) -> str:
    return client_address(request)  # the same address the rate limiter keys on


def _issue(response: Response, user: User) -> TokenResponse:
    s = get_settings()
    token, expires = create_access_token(user.id, user.is_admin)
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
    user = db.scalar(select(User).where(User.email == email))
    if user is None:
        verify_password(body.password, _DUMMY_HASH)
        _login_failed(db, email, "unknown_email", _client(request))
    if not verify_password(body.password, user.password_hash):
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


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> Response:
    response.delete_cookie(get_settings().cookie_name, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
