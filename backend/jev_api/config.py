"""Application settings from environment variables (see .env.example)."""

from __future__ import annotations

import logging
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from jev_ml.paths import EXPERIMENTS_DIR, MODELS_DIR, PROCESSED_DIR, ROOT

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(ROOT / ".env"), env_prefix="JEV_", extra="ignore")

    env: str = Field("development", description="development | test | production")
    database_url: str = f"sqlite:///{ROOT / 'jev.db'}"
    redis_url: str | None = None
    jwt_secret: SecretStr | None = None
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24
    cookie_name: str = "jev_session"
    cookie_secure: bool = False
    # comma-separated in the environment (NoDecode: skip JSON parsing, see _split_origins)
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    rate_limit_per_minute: int = 240
    auth_rate_limit_per_minute: int = 20
    admin_email: str | None = None
    admin_password: SecretStr | None = None
    models_dir: Path = MODELS_DIR
    experiments_dir: Path = EXPERIMENTS_DIR
    processed_dir: Path = PROCESSED_DIR  # processed MovieLens files the intelligence layer reads
    recommendation_cache_seconds: int = 300
    log_level: str = "INFO"
    auto_migrate: bool = True  # run `alembic upgrade head` on startup
    trust_proxy: bool = False  # honour X-Forwarded-For (only behind a trusted reverse proxy)
    # intelligence layer (docs/intelligence.md)
    intel_run_on_startup: bool = True  # refresh a missing/stale run in a background thread at startup
    intel_min_interval_hours: float = Field(24.0, ge=0)  # "stale" = latest successful run older than this
    intel_suppress_days: int = Field(30, ge=0)  # a dismissed warning key stays quiet this long

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip().startswith("["):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def secret_key(self) -> str:
        if self.jwt_secret is not None and self.jwt_secret.get_secret_value():
            return self.jwt_secret.get_secret_value()
        if self.is_production:
            raise RuntimeError("JEV_JWT_SECRET must be set in production")
        return _ephemeral_secret()


@lru_cache
def _ephemeral_secret() -> str:
    log.warning("JEV_JWT_SECRET not set: using an ephemeral random secret (sessions reset on restart)")
    return secrets.token_urlsafe(48)


@lru_cache
def get_settings() -> Settings:
    return Settings()
