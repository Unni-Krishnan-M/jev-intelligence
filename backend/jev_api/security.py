"""Password hashing (Argon2id) and JWT access tokens."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from jev_api.config import get_settings

_hasher = PasswordHasher()  # argon2id with library-recommended parameters


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


def create_access_token(user_id: int, is_admin: bool) -> tuple[str, int]:
    s = get_settings()
    now = datetime.now(UTC)
    expires = now + timedelta(minutes=s.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "adm": is_admin,
        "iat": now,
        "exp": expires,
        "typ": "access",
    }
    token = jwt.encode(payload, s.secret_key(), algorithm=s.jwt_algorithm)
    return token, s.jwt_expire_minutes * 60


def decode_access_token(token: str) -> dict[str, Any] | None:
    s = get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            token, s.secret_key(), algorithms=[s.jwt_algorithm], options={"require": ["exp", "sub", "typ"]}
        )
    except jwt.PyJWTError:
        return None
    if payload.get("typ") != "access":
        return None
    return payload
