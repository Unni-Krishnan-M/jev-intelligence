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

    # Sections, one per owner (docs/PHASE2_ARCHITECTURE_AUDIT.md, "Enabler — done"). A workstream appends
    # settings only inside its own section, prefixed with its name (events_*, governance_*, ...), and never
    # reorders or renames: the field name is the environment variable (JEV_<NAME>), so a rename breaks
    # deployments. Validators and properties stay below the fields.

    # --- core runtime (integrator) -----------------------------------------------------------------
    env: str = Field("development", description="development | test | production")
    log_level: str = "INFO"
    database_url: str = f"sqlite:///{ROOT / 'jev.db'}"
    redis_url: str | None = None
    auto_migrate: bool = True  # run `alembic upgrade head` on startup
    models_dir: Path = MODELS_DIR
    experiments_dir: Path = EXPERIMENTS_DIR
    processed_dir: Path = PROCESSED_DIR  # processed MovieLens files the intelligence layer reads

    # --- security: sessions, CORS, rate limits, admin bootstrap (integrator) --------------------------
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
    # Deployed behind a reverse proxy. The app never parses X-Forwarded-For itself: the client address
    # comes from uvicorn, which rewrites it only for peers in --forwarded-allow-ips /
    # FORWARDED_ALLOW_IPS (see deps.client_address). With this flag set, startup warns when that list
    # trusts every peer ("*").
    trust_proxy: bool = False

    # --- Phase 2 security hardening: service tokens, token revocation (integrator) --------------------
    # append security_* settings here

    # --- recommendation serving (WS5) -----------------------------------------------------------------
    recommendation_cache_seconds: int = 300
    # POST /me/intelligence/scenarios ranks the catalogue once per scenario (about 0.1 s): per member
    me_scenario_rate_limit_per_minute: int = 30

    # --- online experiments (WS5) ---------------------------------------------------------------------
    # append experiments_* settings here

    # --- intelligence layer (WS4, docs/intelligence.md) -----------------------------------------------
    # POST /intel/runs costs about 1 s of CPU and a few hundred rows; one run at a time (lock) and
    # at most this many per client per minute
    intel_run_rate_limit_per_minute: int = 10
    intel_run_on_startup: bool = True  # refresh a missing/stale run in a background thread at startup
    intel_min_interval_hours: float = Field(24.0, ge=0)  # "stale" = latest successful run older than this
    intel_suppress_days: int = Field(30, ge=0)  # a dismissed warning key stays quiet this long

    # --- events and ingestion (WS1) -------------------------------------------------------------------
    # append events_* settings here

    # --- retraining and model governance (WS2) --------------------------------------------------------
    # append governance_* settings here

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
