from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select

from jev_api.deps import DB, CacheDep, CurrentUser, Engine, OptionalUser
from jev_api.models import (
    Favorite,
    Movie,
    Rating,
    Recommendation,
    RecommendationFeedback,
    WatchHistory,
)
from jev_api.schemas import (
    FeedbackOut,
    FeedbackRequest,
    MovieBrief,
    RecommendationHistoryItem,
    RecommendationResponse,
    SimpleRecResponse,
)
from jev_api.services.profile import load_user_state
from jev_api.services.recommend import personalized, simple_items
from jev_ml.models.hybrid import RecommendationFilters
from jev_ml.signals import LIKE_THRESHOLD

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("", response_model=RecommendationResponse)
def recommendations(
    user: CurrentUser,
    db: DB,
    engine: Engine,
    cache: CacheDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=500),
    genres: list[str] | None = Query(None, max_length=10),
    year_min: int | None = Query(None, ge=1800, le=2100),
    year_max: int | None = Query(None, ge=1800, le=2100),
    min_ratings: int | None = Query(None, ge=0),
    max_ratings: int | None = Query(None, ge=0),
    context: str = Query("feed", pattern=r"^[a-z_]{1,32}$"),
) -> dict[str, Any]:
    filters = RecommendationFilters(
        genres=genres, year_min=year_min, year_max=year_max, min_ratings=min_ratings, max_ratings=max_ratings
    )
    return personalized(db, cache, engine, user, limit, offset, filters, context)


@router.get("/similar/{movie_id}", response_model=SimpleRecResponse)
def similar(
    movie_id: int,
    db: DB,
    engine: Engine,
    cache: CacheDep,
    user: OptionalUser,
    limit: int = Query(12, ge=1, le=50),
) -> dict[str, Any]:
    movie = db.get(Movie, movie_id)
    if movie is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "movie not found")
    key = f"sim:{engine.version}:{movie_id}:{limit}"
    rows = cache.get(key)
    if rows is None:
        if engine.item_index.index_of(movie_id) is not None:
            rows = engine.similar(movie_id, limit)
        else:
            # item cold start: vectorize the movie's metadata with the trained featurizer
            rows = engine.similar_to_metadata(
                {
                    "title": movie.title,
                    "year": movie.year,
                    "genres": "|".join(g.name for g in movie.genres),
                    "directors": "|".join(movie.directors),
                    "cast": "|".join(movie.cast),
                    "keywords": "|".join(movie.keywords),
                    "tags": "",
                    "wd_genres": "",
                    "description": movie.description or "",
                },
                limit,
            )
        cache.set(key, rows, 3600)
    items = simple_items(db, rows)
    if user is not None:  # don't suggest what the viewer already consumed
        consumed = load_user_state(db, user).consumed
        items = [i for i in items if i["movie_id"] not in consumed] or items
    return {
        "items": items,
        "model_version": engine.version,
        "generated_at": datetime.now(UTC),
        "anchor": MovieBrief.model_validate(movie),
    }


@router.get("/trending", response_model=SimpleRecResponse)
def trending(
    db: DB,
    engine: Engine,
    cache: CacheDep,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=500),
) -> dict[str, Any]:
    """Time-decayed dataset popularity blended with JEV activity from the last 7 days."""
    key = f"trend:{engine.version}:{limit}:{offset}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    since = datetime.now(UTC) - timedelta(days=7)
    live: dict[int, float] = {}
    for model, col in (
        (Rating, Rating.updated_at),
        (WatchHistory, WatchHistory.watched_at),
        (Favorite, Favorite.created_at),
    ):
        for mid, n in db.execute(
            select(model.movie_id, func.count()).where(col >= since).group_by(model.movie_id)
        ).all():
            live[mid] = live.get(mid, 0.0) + float(n)
    rows = engine.trending(k=limit, offset=offset, live_counts=live)
    for r in rows:
        r["reason"] = "Trending on JEV this week" if live.get(r["movie_id"]) else "Trending with viewers"
        r["signals"] = {"trending": r["score"], "live_events": live.get(r["movie_id"], 0.0)}
    out = {
        "items": simple_items(db, rows),
        "model_version": engine.version,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    cache.set(key, out, 120)
    return out


def _anchor_rows(engine: Any, anchors: list[int], exclude: set[int], limit: int) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for a in anchors:
        if engine.item_index.index_of(a) is None:
            continue
        for r in engine.similar(a, limit * 2):
            if r["movie_id"] in exclude:
                continue
            cur = merged.get(r["movie_id"])
            if cur is None or r["score"] > cur["score"]:
                merged[r["movie_id"]] = r
    return sorted(merged.values(), key=lambda r: (-r["score"], r["movie_id"]))[:limit]


@router.get("/because-you-watched", response_model=SimpleRecResponse)
def because_you_watched(
    user: CurrentUser, db: DB, engine: Engine, limit: int = Query(12, ge=1, le=50)
) -> dict[str, Any]:
    """Item-to-item neighbours of the most recent movie the user watched or rated >= 4."""
    state = load_user_state(db, user)
    recent = sorted(
        (
            e
            for e in state.interactions
            if e.kind == "watch" or (e.kind == "rating" and (e.value or 0) >= LIKE_THRESHOLD)
        ),
        key=lambda e: (-e.timestamp, e.movie_id),
    )
    if not recent:
        return {
            "items": [],
            "model_version": engine.version,
            "generated_at": datetime.now(UTC),
            "anchor": None,
        }
    anchor_id = recent[0].movie_id
    rows = _anchor_rows(engine, [anchor_id], state.consumed | set(state.excluded), limit)
    anchor = db.get(Movie, anchor_id)
    return {
        "items": simple_items(db, rows),
        "model_version": engine.version,
        "generated_at": datetime.now(UTC),
        "anchor": MovieBrief.model_validate(anchor) if anchor else None,
    }


@router.get("/similar-to-favorites", response_model=SimpleRecResponse)
def similar_to_favorites(
    user: CurrentUser, db: DB, engine: Engine, limit: int = Query(12, ge=1, le=50)
) -> dict[str, Any]:
    state = load_user_state(db, user)
    favs = [
        e.movie_id
        for e in sorted(state.interactions, key=lambda e: (-e.timestamp, e.movie_id))
        if e.kind in ("favorite", "onboarding")
    ][:5]
    rows = _anchor_rows(engine, favs, state.consumed | set(state.excluded), limit)
    titles = {m.id: m.title for m in db.scalars(select(Movie).where(Movie.id.in_(favs)))} if favs else {}
    for r in rows:  # attribute each item to its best-matching favorite
        best = max(favs, key=lambda f: (_sim(engine, f, r["movie_id"]), -f))
        r["reason"] = f"Similar to your favorite {titles.get(best, '')}".strip()
    return {
        "items": simple_items(db, rows),
        "model_version": engine.version,
        "generated_at": datetime.now(UTC),
    }


def _sim(engine: Any, a: int, b: int) -> float:
    ia, ib = engine.item_index.index_of(a), engine.item_index.index_of(b)
    if ia is None or ib is None:
        return 0.0
    return float(engine.content.similarity_to(ia, [ib])[0])


@router.post("/feedback", response_model=FeedbackOut, status_code=status.HTTP_201_CREATED)
def feedback(body: FeedbackRequest, user: CurrentUser, db: DB) -> RecommendationFeedback:
    if db.get(Movie, body.movie_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "movie not found")
    if body.recommendation_id is not None:
        rec = db.get(Recommendation, body.recommendation_id)
        if rec is None or rec.user_id != user.id or rec.movie_id != body.movie_id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "recommendation not found")
    fb = RecommendationFeedback(
        user_id=user.id,
        movie_id=body.movie_id,
        feedback=body.feedback,
        recommendation_id=body.recommendation_id,
    )
    db.add(fb)
    if body.feedback in ("dislike", "not_interested"):
        user.profile_version += 1  # these exclude the movie from future recommendations
    db.commit()
    db.refresh(fb)
    return fb


@router.get("/history", response_model=list[RecommendationHistoryItem])
def history(user: CurrentUser, db: DB, limit: int = Query(50, ge=1, le=200)) -> list[dict[str, Any]]:
    rows = db.scalars(
        select(Recommendation)
        .where(Recommendation.user_id == user.id)
        .order_by(Recommendation.created_at.desc(), Recommendation.rank)
        .limit(limit)
    ).all()
    fb = dict(
        db.execute(
            select(RecommendationFeedback.recommendation_id, RecommendationFeedback.feedback).where(
                RecommendationFeedback.user_id == user.id,
                RecommendationFeedback.recommendation_id.in_([r.id for r in rows]),
            )
        ).all()
    )
    return [
        {
            "id": r.id,
            "movie_id": r.movie_id,
            "title": r.movie.title,
            "request_id": r.request_id,
            "model_version": r.model_version,
            "context": r.context,
            "rank": r.rank,
            "score": r.score,
            "reason": r.reason,
            "created_at": r.created_at,
            "feedback": fb.get(r.id),
        }
        for r in rows
    ]
