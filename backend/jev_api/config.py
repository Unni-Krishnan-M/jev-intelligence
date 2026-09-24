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
    # per-account login throttle (docs/SECURITY_AUDIT_PHASE2.md, F4): failed logins per email per window,
    # independent of the client address, so neither a shared proxy address nor a spoofed one bypasses it
    security_login_max_failures: int = Field(10, ge=1)
    security_login_window_seconds: int = Field(900, ge=60)
    # signed client address from the web proxy (X-JEV-Client, HMAC-SHA256). Unset: the header is ignored
    # and request.client is used (uvicorn --proxy-headers still applies)
    security_proxy_secret: SecretStr | None = None
    security_proxy_max_skew_seconds: int = Field(60, ge=5, le=600)
    # backstop: all traffic carried by one proxy peer, per minute, when signed client addresses are used
    security_proxy_rate_limit_per_minute: int = Field(6000, ge=1)
    # service tokens (machine ingestion): longest allowed lifetime
    security_service_token_max_days: int = Field(365, ge=1, le=3650)

    # --- recommendation serving (WS5) -----------------------------------------------------------------
    recommendation_cache_seconds: int = 300
    # POST /me/intelligence/scenarios ranks the catalogue once per scenario (about 0.1 s): per member
    me_scenario_rate_limit_per_minute: int = 30

    # --- online experiments (WS5) ---------------------------------------------------------------------
    # append experiments_* settings here
    # docs/EXPERIMENTATION.md. Off: every member is served the default config (exposures are still logged)
    experiments_enabled: bool = True
    # contamination controls: admins and test accounts are never enrolled
    experiments_exclude_admins: bool = True
    # comma-separated fnmatch patterns on the lower-cased email
    experiments_excluded_email_patterns: str = "*@test.invalid,*+jevtest@*"
    # the exposure log (one row per served list, cache hits included; gap P1 #6)
    experiments_log_exposures: bool = True
    experiments_default_attribution_hours: float = Field(24.0, gt=0, le=24 * 30)
    # challenger model versions a variant may load next to the champion (each costs one engine in memory)
    experiments_max_challenger_models: int = Field(2, ge=0, le=5)
    experiments_bootstrap_resamples: int = Field(2000, ge=200, le=20000)
    experiments_srm_alpha: float = Field(0.001, gt=0, lt=0.5)

    # --- intelligence layer (WS4, docs/intelligence.md) -----------------------------------------------
    # POST /intel/runs costs about 1 s of CPU and a few hundred rows; one run at a time (lock) and
    # at most this many per client per minute
    intel_run_rate_limit_per_minute: int = 10
    intel_run_on_startup: bool = True  # refresh a missing/stale run in a background thread at startup
    intel_min_interval_hours: float = Field(24.0, ge=0)  # "stale" = latest successful run older than this
    intel_suppress_days: int = Field(30, ge=0)  # a dismissed warning key stays quiet this long
    # P2.4 stale-warning auto-resolution: an open warning whose key is absent from this many consecutive
    # live runs is resolved by the system (0 = off); severities above the cap wait for an operator
    intel_auto_resolve_runs: int = Field(3, ge=0)
    intel_auto_resolve_max_severity: str = Field("medium", pattern="^(low|medium|high|critical)$")

    # --- events and ingestion (WS1) -------------------------------------------------------------------
    # append events_* settings here
    # docs/STREAMING_ARCHITECTURE.md: batch caps, clock-skew limits, the keyless watch-retry window and
    # the debounced near-real-time refresher (None = auto: on, except under pytest or JEV_ENV=test)
    events_batch_max: int = Field(500, ge=1, le=10_000)  # POST /events items per request
    events_observations_batch_max: int = Field(5000, ge=1, le=100_000)  # POST .../observations
    events_max_future_skew_seconds: float = Field(300.0, ge=0)  # client clocks may run this far ahead
    events_max_age_days: float = Field(30.0, ge=0)  # member events older than this are rejected
    events_watch_dedupe_seconds: float = Field(60.0, ge=0)  # identical keyless watch = a retry
    events_refresh_enabled: bool | None = None
    events_refresh_debounce_seconds: float = Field(30.0, ge=0)  # quiet time before a live run
    events_refresh_max_delay_seconds: float = Field(300.0, ge=0)  # upper bound under steady traffic

    # --- retraining and model governance (WS2) --------------------------------------------------------
    # append governance_* settings here
    # docs/RETRAINING_AND_MODEL_GOVERNANCE.md. Snapshots: data/snapshots/<id>/ (gitignored)
    governance_snapshots_dir: Path = ROOT / "data" / "snapshots"
    governance_app_user_offset: int = Field(10_000_000, ge=1)  # app user u trains as u + offset
    governance_config_path: Path | None = (
        None  # experiment YAML for retraining (default configs/experiment.yaml)
    )
    governance_quick_default: bool = (
        True  # POST /governance/retrain without "quick": skip tuning (laptop budget)
    )
    governance_calibrate: bool = True  # calibrate a candidate when the incumbent serves calibrated confidence
    governance_auto_promote: bool = False  # promote a passing candidate without an operator (off by default)
    # scheduler loop (scripts/retrain.py schedule, compose service "scheduler"); off in dev and tests
    governance_schedule_enabled: bool = False
    governance_schedule_interval_minutes: float = Field(1440.0, gt=0)  # min spacing of automatic jobs
    governance_schedule_poll_seconds: float = Field(300.0, gt=0)  # how often the loop checks
    governance_schedule_require_decision: bool = True  # retrain only when retrain_model answers yes
    governance_job_timeout_minutes: float = Field(180.0, gt=0)  # SQLite lease expiry of the retrain lock
    governance_engine_poll_seconds: float = Field(5.0, ge=0)  # API workers follow registry.json (0 = off)
    # promotion gate (jev_ml.governance.gates.GateConfig)
    # margins follow the gate's power rule (gates.py docstring): 80 % pass rate for an equal model
    governance_gate_ndcg_margin: float = Field(0.010, ge=0)  # non-inferiority margin, NDCG@10
    governance_gate_recall_margin: float = Field(0.011, ge=0)  # non-inferiority margin, Recall@10
    governance_gate_cold_start: bool = True  # also require cold-start NDCG@10 non-inferiority
    governance_gate_cold_start_margin: float = Field(0.006, ge=0)  # margin, cold-start NDCG@10
    governance_gate_coverage_max_drop: float = Field(0.2, ge=0, le=1)  # relative coverage@10 drop allowed
    governance_gate_ece_margin: float = Field(0.01, ge=0)  # candidate ECE <= incumbent ECE + margin
    governance_gate_auc_margin: float = Field(0.01, ge=0)  # candidate AUC >= incumbent AUC - margin
    governance_gate_latency_p95_ms: float = Field(250.0, gt=0)  # build_profile + recommend(k=10)
    governance_gate_latency_requests: int = Field(50, ge=1)
    governance_gate_bootstrap_b: int = Field(2000, ge=100)
    governance_gate_alpha: float = Field(0.10, gt=0, lt=1)  # 0.10 = one-sided 95 % lower bound
    governance_gate_max_users: int | None = Field(None, ge=10)  # subsample evaluated users (None = all)

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
