from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from jev_api.config import get_settings
from jev_api.deps import DB
from jev_api.models import User
from jev_api.routers.users import user_out
from jev_api.schemas import LoginRequest, RegisterRequest, TokenResponse
from jev_api.security import create_access_token, hash_password, needs_rehash, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

# Verified against when the email is unknown, so response time does not reveal which accounts exist.
_DUMMY_HASH = hash_password("jev-timing-equalizer")


def _issue(response: Response, user: User) -> TokenResponse:
    s = get_settings()
    token, expires = create_access_token(user.id, user.is_admin)
    response.set_cookie(
        s.cookie_name, token, max_age=expires, httponly=True, secure=s.cookie_secure, samesite="lax", path="/"
    )
    return TokenResponse(access_token=token, expires_in=expires, user=user_out(user))


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, response: Response, db: DB) -> TokenResponse:
    email = body.email.lower()
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with this email already exists")
    user = User(email=email, password_hash=hash_password(body.password), display_name=body.display_name)
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:  # concurrent registration with the same email
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "an account with this email already exists") from exc
    db.refresh(user)
    return _issue(response, user)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, response: Response, db: DB) -> TokenResponse:
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if user is None:
        verify_password(body.password, _DUMMY_HASH)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    if not verify_password(body.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(body.password)
        db.commit()
    return _issue(response, user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> Response:
    response.delete_cookie(get_settings().cookie_name, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response
