"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from jev_api import __version__
from jev_api.cache import build_cache
from jev_api.config import Settings, get_settings
from jev_api.db import SessionLocal
from jev_api.deps import client_address
from jev_api.logging_setup import configure_logging, request_id_var
from jev_api.metrics import metrics
from jev_api.routers import admin, auth, health, intel, movies, recommendations, users
from jev_api.services.intel import IntelService
from jev_api.services.ml import EngineHolder
from jev_api.services.sync import ensure_admin, sync_all

log = logging.getLogger("jev_api")

AUTH_PATHS = ("/auth/login", "/auth/register")
# a client or upstream proxy id is only logged (as upstream_request_id), never used as the request id
UPSTREAM_REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # JSON only: nothing to load, frame or run
    "Content-Security-Policy": (
        "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-site",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
# /docs (development only) loads Swagger UI from a CDN with inline scripts; keep it usable
DOCS_PATHS = ("/docs", "/docs/oauth2-redirect")


def run_migrations(settings: Settings) -> None:
    from alembic import command
    from alembic.config import Config

    from jev_ml.paths import ROOT

    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "backend" / "jev_api" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    command.upgrade(cfg, "head")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    if settings.auto_migrate:
        run_migrations(settings)
    app.state.engines.load()
    with SessionLocal() as db:
        synced = sync_all(db)
        ensure_admin(db)
    # a stale or missing intelligence run is refreshed in a background thread: never blocks startup
    app.state.intel.start_background_refresh()
    if settings.trust_proxy and os.environ.get("FORWARDED_ALLOW_IPS", "").strip() == "*":
        log.warning("FORWARDED_ALLOW_IPS='*' trusts X-Forwarded-For from any peer: list the proxy address")
    log.info("startup complete", extra={"extra_fields": {"synced": synced, "cache": app.state.cache.backend}})
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(
        title="JEV API",
        version=__version__,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings
    app.state.log = log
    app.state.cache = build_cache(settings.redis_url)
    app.state.engines = EngineHolder(settings.models_dir)
    app.state.intel = IntelService(settings, app.state.engines, app.state.cache)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-JEV-CSRF"],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # always a server-generated id: a client-chosen one could collide with, or forge, audit rows
        rid = str(uuid.uuid4())
        upstream = request.headers.get("x-request-id", "")
        upstream_rid = upstream if UPSTREAM_REQUEST_ID.fullmatch(upstream) else None
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            limited = _rate_limited(request, settings)
            if limited is not None:
                metrics.inc("http", "rate_limited")
                metrics.record_request(request.method, "rate_limited", 429, 0.0)
                return limited
            try:
                response = await call_next(request)
            except Exception:
                # log the traceback server-side; clients get a generic message plus the request id
                log.exception("unhandled error on %s %s", request.method, request.url.path)
                metrics.inc("http", "unhandled_exceptions")
                response = JSONResponse(
                    {"detail": "internal server error", "request_id": rid}, status_code=500
                )
        finally:
            request_id_var.reset(token)
        elapsed = (time.perf_counter() - start) * 1000
        # the router stores the matched route in the scope; unmatched paths share one label
        route = getattr(request.scope.get("route"), "path", None) or "unmatched"
        metrics.record_request(request.method, route, response.status_code, elapsed)
        response.headers["X-Request-ID"] = rid
        for name, value in SECURITY_HEADERS.items():
            if name == "Content-Security-Policy" and request.url.path in DOCS_PATHS:
                continue
            response.headers.setdefault(name, value)
        if request.url.scheme == "https":  # uvicorn sets the scheme from a trusted proxy's X-Forwarded-Proto
            response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
        log.info(
            "request",
            extra={
                "extra_fields": {
                    "request_id": rid,
                    "upstream_request_id": upstream_rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(elapsed, 1),
                }
            },
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            {"detail": exc.detail, "request_id": request_id_var.get()},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()
        ]
        return JSONResponse(
            {"detail": "validation error", "errors": errors, "request_id": request_id_var.get()},
            status_code=422,
        )

    for r in (
        health.router,
        auth.router,
        users.router,
        movies.router,
        recommendations.router,
        admin.router,
        intel.router,
    ):
        app.include_router(r)
    return app


def _rate_limited(request: Request, settings: Settings) -> JSONResponse | None:
    if request.method == "OPTIONS" or request.url.path.startswith("/health"):
        return None
    client = client_address(request)  # never the client-controlled X-Forwarded-For
    is_auth = request.url.path in AUTH_PATHS
    limit = settings.auth_rate_limit_per_minute if is_auth else settings.rate_limit_per_minute
    bucket = int(time.time() // 60)
    key = f"rl:{'auth' if is_auth else 'api'}:{client}:{bucket}"
    count = request.app.state.cache.incr_window(key, 60)
    if count > limit:
        return JSONResponse(
            {"detail": "rate limit exceeded", "request_id": request_id_var.get()},
            status_code=429,
            headers={"Retry-After": str(60 - int(time.time()) % 60)},
        )
    return None


app = create_app()
