"""Serving logic shared by the recommendation endpoints: generate → enrich → persist → cache.

v1.2 (docs/platform.md, section 4): GET /recommendations is downstream of the member's
recommendation_strategy decision. The decision is taken first (cached per member, see
services/user_intel.py), the list is served under the chosen strategy, the cache key carries the
decision, and the response gains an `intelligence` block. A failing strategy step never fails the
request: the list is then served as standard, without the block, and the failure is logged and counted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from jev_api.cache import Cache
from jev_api.config import get_settings
from jev_api.metrics import metrics
from jev_api.models import Movie, Recommendation, User
from jev_api.services.profile import build_ml_profile, load_user_state
from jev_api.services.user_intel import StrategyChoice, serve_engine, serve_profile, strategy_choice
from jev_ml.engine import RecommendationEngine
from jev_ml.models.hybrid import RecommendationFilters

log = logging.getLogger(__name__)

DIVERSITY_LAMBDA = {"focused": 1.0, "adventurous": 0.7}  # "balanced" = the model's tuned value
CONFIDENCE_KINDS = ("probability",)  # recommendations.confidence_kind CHECK


def confidence_of(rec: Any) -> tuple[float | None, str | None]:
    """(confidence, confidence_kind) of an engine Recommendation or a similar/trending row dict.
    The calibrated P(rating >= 4) of section 9.2; (None, None) when the model has no calibration,
    or the value is not a finite probability (it is never guessed)."""
    if isinstance(rec, dict):
        value, kind = rec.get("confidence"), rec.get("confidence_kind")
    else:
        value, kind = getattr(rec, "confidence", None), getattr(rec, "confidence_kind", None)
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= float(value) <= 1.0:
        return None, None
    return round(float(value), 6), kind if kind in CONFIDENCE_KINDS else "probability"


def movie_briefs(db: Session, ids: list[int]) -> dict[int, Movie]:
    if not ids:
        return {}
    return {m.id: m for m in db.scalars(select(Movie).where(Movie.id.in_(ids))).all()}


def _engine_for_user(engine: RecommendationEngine, user: User) -> RecommendationEngine:
    """Per-user ranking config (diversity preference) without mutating the shared engine."""
    pref = (user.recommendation_prefs or {}).get("diversity")
    if pref not in DIVERSITY_LAMBDA:
        return engine
    e = copy.copy(engine)
    e.ranker = copy.copy(engine.ranker)
    e.ranker.config = copy.copy(engine.ranker.config)
    e.ranker.config.diversity_lambda = DIVERSITY_LAMBDA[pref]
    return e


def personalized(
    db: Session,
    cache: Cache,
    engine: RecommendationEngine,
    user: User,
    limit: int,
    offset: int,
    filters: RecommendationFilters,
    context: str = "feed",
) -> dict[str, Any]:
    settings = get_settings()
    fkey = hashlib.sha256(json.dumps(filters.__dict__, sort_keys=True, default=str).encode()).hexdigest()[:12]
    diversity = (user.recommendation_prefs or {}).get("diversity", "balanced")
    t0 = time.perf_counter()
    choice: StrategyChoice | None = None
    state = profile = None
    try:
        choice, state, profile = strategy_choice(db, cache, engine, user)
    except Exception:
        db.rollback()
        metrics.inc("strategy_decisions", "errors")
        log.exception(
            "recommendation strategy failed; serving standard", extra={"extra_fields": {"user": user.id}}
        )
    strategy_ms = 1000 * (time.perf_counter() - t0)
    metrics.observe("strategy_overhead_ms", "all", strategy_ms)
    served = choice.served if choice is not None else "standard"
    decision_id = choice.block["decision_id"] if choice is not None else None
    key = (
        f"rec:{user.id}:{user.profile_version}:{engine.version}:{diversity}:{context}:{limit}:{offset}:{fkey}"
        f":{served}:{decision_id or '-'}"
    )
    cached = cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}

    if state is None:
        state = load_user_state(db, user)
    if profile is None:
        profile = build_ml_profile(engine, state)
    eng = _engine_for_user(engine, user)
    if choice is not None:
        profile = serve_profile(profile, choice)
        eng = serve_engine(eng, choice, user)
    recs = eng.recommend(profile, k=limit, offset=offset, filters=filters)
    movies = movie_briefs(db, [r.movie_id for r in recs])

    request_id = str(uuid.uuid4())
    conf = {r.movie_id: confidence_of(r) for r in recs}
    rows = [
        Recommendation(
            user_id=user.id,
            movie_id=r.movie_id,
            request_id=request_id,
            model_version=engine.version,
            context=context,
            rank=r.rank,
            score=r.score,
            reason=r.reason[:300],
            reason_code=r.reason_code,
            signals=r.signals,
            confidence=conf[r.movie_id][0],
            confidence_kind=conf[r.movie_id][1],
            decision_id=decision_id,
            strategy=served if choice is not None else None,
        )
        for r in recs
        if r.movie_id in movies
    ]
    db.add_all(rows)
    db.commit()
    row_ids = {row.movie_id: row.id for row in rows}

    items = []
    for r in recs:
        m = movies.get(r.movie_id)
        if m is None:
            continue
        items.append(
            {
                "recommendation_id": row_ids.get(r.movie_id),
                "movie_id": r.movie_id,
                "title": m.title,
                "year": m.year,
                "genres": [g.name for g in m.genres],
                "directors": m.directors[:2],
                "score": r.score,
                "rank": r.rank,
                "reason": r.reason,
                "reason_code": r.reason_code,
                "secondary_reasons": r.secondary_reasons,
                "anchor_movie_ids": r.anchor_movie_ids,
                "signals": r.signals,
                "confidence": conf[r.movie_id][0],
                "confidence_kind": conf[r.movie_id][1],
                "decision_id": decision_id,
                "strategy": served if choice is not None else None,
            }
        )
    response = {
        "items": items,
        "model_version": engine.version,
        "generated_at": datetime.now(UTC).isoformat(),
        "request_id": request_id,
        "limit": limit,
        "offset": offset,
        "effective_weights": {k: round(v, 4) for k, v in eng.effective_weights(profile).items()},
        "profile": {
            "interactions": profile.n_interactions,
            "genres": len(state.genres),
            "excluded": len(state.excluded),
        },
        "cached": False,
        "intelligence": choice.block if choice is not None else None,
    }
    cache.set(key, response, settings.recommendation_cache_seconds)
    return response


def simple_items(db: Session, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    movies = movie_briefs(db, [r["movie_id"] for r in rows])
    out = []
    for r in rows:
        m = movies.get(r["movie_id"])
        if m is None:
            continue
        confidence, kind = confidence_of(r)
        out.append(
            {
                "movie_id": m.id,
                "title": m.title,
                "year": m.year,
                "genres": [g.name for g in m.genres],
                "directors": m.directors[:2],
                "score": r["score"],
                "reason": r.get("reason"),
                "signals": r.get("signals", {}),
                "confidence": confidence,
                "confidence_kind": kind,
            }
        )
    return out
