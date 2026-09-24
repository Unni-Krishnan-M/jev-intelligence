"""Password hashing (Argon2id) and JWT access tokens."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
import uuid
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


def create_access_token(user_id: int, is_admin: bool, token_version: int = 0) -> tuple[str, int]:
    """A session JWT. `jti` makes one token revocable (logout); `ver` is the account's token_version
    at issue time, so bumping it (revoke-all, password change, role change) invalidates every token."""
    s = get_settings()
    now = datetime.now(UTC)
    expires = now + timedelta(minutes=s.jwt_expire_minutes)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "adm": is_admin,
        "iat": now,
        "exp": expires,
        "typ": "access",
        "jti": uuid.uuid4().hex,
        "ver": int(token_version),
    }
    token = jwt.encode(payload, s.secret_key(), algorithm=s.jwt_algorithm)
    return token, s.jwt_expire_minutes * 60


def decode_access_token(token: str) -> dict[str, Any] | None:
    s = get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            s.secret_key(),
            algorithms=[s.jwt_algorithm],
            options={"require": ["exp", "sub", "typ", "jti", "ver"]},
        )
    except jwt.PyJWTError:
        return None
    # tokens issued before revocation existed (no jti/ver) are rejected by "require" above
    if payload.get("typ") != "access" or not isinstance(payload.get("ver"), int):
        return None
    sub, jti = payload.get("sub"), payload.get("jti")
    if not (isinstance(sub, str) and sub.isascii() and sub.isdigit() and len(sub) <= 10):
        return None
    if not (isinstance(jti, str) and 1 <= len(jti) <= 64):
        return None
    return payload


# --- service tokens (machine ingestion) -------------------------------------------------------------
SERVICE_TOKEN_PREFIX = "jevst_"  # noqa: S105 - a public marker, not a secret


def new_service_token() -> str:
    """A fresh service token: shown once, only its SHA-256 is stored. 256 bits of randomness, so a
    fast hash is enough (no password-style stretching needed for a random secret)."""
    return SERVICE_TOKEN_PREFIX + secrets.token_urlsafe(32)


def service_token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def is_service_token(token: str | None) -> bool:
    return token is not None and token.startswith(SERVICE_TOKEN_PREFIX) and len(token) <= 100


# --- signed client address from the web proxy --------------------------------------------------------
CLIENT_HEADER = "x-jev-client"  # "v1:<ip>:<unix ts>:<hex hmac-sha256>"


def sign_client_address(ip: str, secret: str, ts: int | None = None) -> str:
    """The header value the web proxy sends (mirrors frontend/src/proxy.ts; used by tests)."""
    ts = int(datetime.now(UTC).timestamp()) if ts is None else ts
    mac = hmac.new(secret.encode(), f"v1|{ip}|{ts}".encode(), hashlib.sha256).hexdigest()
    return f"v1:{ip}:{ts}:{mac}"


def verify_client_address(
    value: str | None, secret: str, max_skew: int, now: float | None = None
) -> str | None:
    """The client IP from a signed header, or None when it is absent, malformed, stale or forged.
    The IP may itself contain ':' (IPv6), so the fields are split from both ends."""
    if not value or len(value) > 200 or not value.startswith("v1:"):
        return None
    head, _, mac = value.rpartition(":")
    rest, _, ts_s = head.rpartition(":")
    ip = rest[3:]
    if not (ts_s.isascii() and ts_s.isdigit()) or len(mac) != 64:
        return None
    try:
        ip = str(ipaddress.ip_address(ip))
    except ValueError:
        return None
    now = datetime.now(UTC).timestamp() if now is None else now
    if abs(now - int(ts_s)) > max_skew:
        return None
    expected = hmac.new(secret.encode(), f"v1|{rest[3:]}|{ts_s}".encode(), hashlib.sha256).hexdigest()
    return ip if hmac.compare_digest(expected, mac.lower()) else None
