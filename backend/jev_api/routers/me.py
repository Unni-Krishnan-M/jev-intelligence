"""Per-member intelligence (/me/intelligence*, any signed-in member). Contract: docs/platform.md,
sections 8 and 10.

Everything is computed from the caller's own data only (there is no user parameter): the interactions,
genre preferences and excluded movies the recommendation service uses, plus the caller's feedback.
Operator feedback on the intelligence layer stays under /intel/feedback.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import select

from jev_api.deps import DB, CacheDep, CurrentUser, Engine, parse_db_id
from jev_api.metrics import metrics
from jev_api.models import Movie, Recommendation, UserIntelFeedback
from jev_api.schemas import MeFeedbackIn, MeFeedbackOut, MeScenarioRequest
from jev_api.services import audit
from jev_api.services.profile import load_user_state
from jev_api.services.user_intel import (
    DECISION_ID,
    feedback_marker,
    feedback_rows,
    me_feedback_out,
    strategy_choice,
    upsert_me_feedback,
)
from jev_ml.domains.movie.user_intel import user_intelligence
from jev_ml.domains.movie.user_scenario import user_preference_scenarios

router = APIRouter(prefix="/me", tags=["me"])


def _rate_limit(request: Request, user_id: int, name: str, limit: int) -> None:
    """Per member per minute, on top of the global per-client limit (these calls rank the catalogue
    several times)."""
    window = int(time.time() // 60)
    if request.app.state.cache.incr_window(f"rl:{name}:{user_id}:{window}", 60) > limit:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "too many requests; try again in a minute",
            headers={"Retry-After": str(60 - int(time.time()) % 60)},
        )


@router.get("/intelligence")
def my_intelligence(
    user: CurrentUser,
    db: DB,
    engine: Engine,
    cache: CacheDep,
    request: Request,
    k: int = Query(10, ge=1, le=50),
) -> dict[str, Any]:
    """Preference history, the drift report, the recommendation-strategy decision with its evidence,
    the caller's drift signals and the recommendations served under that decision."""
    key = f"meintel:{user.id}:{user.profile_version}:{engine.version}:{feedback_marker(db, user.id)}:{k}"
    hit = cache.get(key)
    if isinstance(hit, dict):
        return hit
    t0 = time.perf_counter()
    state = load_user_state(db, user)
    try:
        out = user_intelligence(
            engine,
            state.interactions,
            feedback=feedback_rows(db, user.id),
            k=k,
            user_id=user.id,
            genre_prefs=state.genres,
            excluded_movie_ids=state.excluded,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    metrics.observe("me_intelligence_ms", "all", 1000 * (time.perf_counter() - t0))
    cache.set(key, out, request.app.state.settings.recommendation_cache_seconds)
    return out


@router.post("/intelligence/scenarios")
def my_scenarios(
    body: MeScenarioRequest, user: CurrentUser, db: DB, engine: Engine, request: Request
) -> dict[str, Any]:
    """What if the caller's preference trend continues, accelerates or reverses? Projected genre
    shares (80 % bootstrap bands) and the hybrid model's ranking under each projection."""
    _rate_limit(
        request, user.id, "me_scenarios", request.app.state.settings.me_scenario_rate_limit_per_minute
    )
    spec: dict[str, Any] = {"k": body.k}
    if body.scenarios:
        spec["scenarios"] = [s.model_dump(exclude_none=True) for s in body.scenarios]
    state = load_user_state(db, user)
    t0 = time.perf_counter()
    try:
        out = user_preference_scenarios(
            engine,
            state.interactions,
            spec,
            genre_prefs=state.genres,
            excluded_movie_ids=state.excluded,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    metrics.observe("me_scenarios_ms", "all", 1000 * (time.perf_counter() - t0))
    return out


@router.post("/intelligence/feedback", response_model=MeFeedbackOut, status_code=status.HTTP_201_CREATED)
def my_feedback(
    body: MeFeedbackIn, user: CurrentUser, db: DB, cache: CacheDep, request: Request
) -> dict[str, Any]:
    """accepted | rejected on the caller's strategy decision ("dec-…": one they were served under, or
    their current one) or on a recommended movie (its id). One verdict per target: repeating it is a
    no-op, a new verdict replaces the old one."""
    tid = body.target_id.strip()
    movie_id: int | None = None
    decision_id: str | None
    if body.target_type == "recommendation":
        movie_id = parse_db_id(tid)
        if movie_id is None:
            raise HTTPException(422, "target_id of a recommendation is the recommended movie id")
        if db.get(Movie, movie_id) is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "movie not found")
        tid = str(movie_id)  # canonical ("007" and "7" are one target)
        decision_id = db.scalar(
            select(Recommendation.decision_id)
            .where(
                Recommendation.user_id == user.id,
                Recommendation.movie_id == movie_id,
                Recommendation.decision_id.is_not(None),
            )
            .order_by(Recommendation.created_at.desc(), Recommendation.id.desc())
            .limit(1)
        )
    else:
        if not DECISION_ID.fullmatch(tid):
            raise HTTPException(422, "target_id of a strategy is a decision id (dec-…)")
        served = db.scalar(
            select(Recommendation.id)
            .where(Recommendation.user_id == user.id, Recommendation.decision_id == tid)
            .limit(1)
        )
        known = served is not None
        engine = request.app.state.engines.engine
        if not known and engine is not None:
            choice, _, _ = strategy_choice(db, cache, engine, user)
            known = choice.block["decision_id"] == tid
        if not known:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "strategy decision not found for this member")
        decision_id = tid
    if decision_id is None and request.app.state.engines.engine is not None:
        # a recommendation never served through /recommendations (e.g. shown on /me/intelligence):
        # attribute it to the member's current strategy decision
        choice, _, _ = strategy_choice(db, cache, request.app.state.engines.engine, user)
        decision_id = choice.block["decision_id"]
    row, changed = upsert_me_feedback(
        db, user, body.target_type, tid, body.verdict, body.note, decision_id, movie_id
    )
    if changed:
        audit.record(
            db,
            "me.feedback",
            user,
            body.target_type,
            tid,
            {
                "feedback_id": row.id,
                "verdict": body.verdict,
                "decision_id": decision_id,
                "has_note": bool(body.note),
            },
        )
    db.commit()
    db.refresh(row)
    metrics.inc("me_feedback", f"{body.target_type}:{body.verdict}" if changed else "unchanged")
    return me_feedback_out(row)


@router.get("/intelligence/feedback")
def my_feedback_list(user: CurrentUser, db: DB, limit: int = Query(100, ge=1, le=200)) -> dict[str, Any]:
    """The caller's own verdicts, newest first."""
    rows = db.scalars(
        select(UserIntelFeedback)
        .where(UserIntelFeedback.user_id == user.id)
        .order_by(UserIntelFeedback.updated_at.desc(), UserIntelFeedback.id.desc())
        .limit(limit)
    ).all()
    return {"items": [me_feedback_out(r) for r in rows], "total": len(rows)}
