from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from jev_api import __version__
from jev_api.db import SessionLocal
from jev_api.services.ml import engine_calibration

router = APIRouter(tags=["health"])


@router.get("/health")
def health(request: Request, response: Response) -> dict[str, Any]:
    db_ok = True
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception:
        request.app.state.log.exception("database health check failed")
        db_ok = False
    cache = request.app.state.cache
    cache_ok = cache.ping()
    ok = db_ok and cache_ok
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if ok else "degraded",
        "version": __version__,
        "database": "ok" if db_ok else "error",
        "cache": {"backend": cache.backend, "status": "ok" if cache_ok else "error"},
    }


@router.get("/health/ml")
def health_ml(request: Request, response: Response) -> dict[str, Any]:
    holder = request.app.state.engines
    engine = holder.engine
    if engine is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        # public endpoint: a fixed message only. The load exception (it can contain file paths) stays in
        # the log and in the admin-only GET /admin/metrics (engine.last_error).
        return {"status": "unavailable", "error": "no recommendation model is loaded"}
    # live self-check: a cold-start request must produce finite, valid recommendations
    recs = engine.recommend(engine.build_profile([]), k=5)
    valid = bool(recs) and all(
        engine.item_index.index_of(r.movie_id) is not None and r.score == r.score for r in recs
    )
    if not valid:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if valid else "error",
        "self_check": {"recommendations": len(recs)},
        **engine.health(),
        "calibration": engine_calibration(engine),
    }
