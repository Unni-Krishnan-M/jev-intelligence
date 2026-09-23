from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import delete, func, select

from jev_api.deps import DB, CurrentUser
from jev_api.models import Favorite, Genre, Movie, Rating, User, UserGenrePreference, WatchHistory
from jev_api.schemas import MovieBrief, OnboardingRequest, PreferenceUpdate, UserOut
from jev_api.services.profile import taste_profile

router = APIRouter(prefix="/users", tags=["users"])


def user_out(user: User) -> UserOut:
    return UserOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        is_admin=user.is_admin,
        onboarding_completed=user.onboarding_completed,
        favorite_genres=sorted(p.genre.name for p in user.genre_preferences),
        created_at=user.created_at,
    )


def _set_genres(db: DB, user: User, names: list[str]) -> None:
    genres = db.scalars(select(Genre).where(Genre.name.in_(names))).all()
    unknown = set(names) - {g.name for g in genres}
    if unknown:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"unknown genres: {sorted(unknown)}")
    db.execute(delete(UserGenrePreference).where(UserGenrePreference.user_id == user.id))
    db.add_all(UserGenrePreference(user_id=user.id, genre_id=g.id) for g in genres)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return user_out(user)


@router.post("/me/onboarding", response_model=UserOut)
def onboarding(body: OnboardingRequest, user: CurrentUser, db: DB) -> UserOut:
    _set_genres(db, user, body.genres)
    if body.movie_ids:
        known = set(db.scalars(select(Movie.id).where(Movie.id.in_(body.movie_ids))).all())
        missing = set(body.movie_ids) - known
        if missing:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"unknown movie ids: {sorted(missing)}")
        existing = set(db.scalars(select(Favorite.movie_id).where(Favorite.user_id == user.id)).all())
        db.add_all(
            Favorite(user_id=user.id, movie_id=m, source="onboarding") for m in sorted(known - existing)
        )
    user.onboarding_completed = True
    user.profile_version += 1
    db.commit()
    db.refresh(user)
    return user_out(user)


@router.patch("/me/preferences", response_model=UserOut)
def update_preferences(body: PreferenceUpdate, user: CurrentUser, db: DB) -> UserOut:
    if body.genres is not None:
        _set_genres(db, user, body.genres)
    if body.diversity is not None:
        user.recommendation_prefs = {**(user.recommendation_prefs or {}), "diversity": body.diversity}
    user.profile_version += 1
    db.commit()
    db.refresh(user)
    return user_out(user)


@router.get("/me/profile")
def my_profile(user: CurrentUser, db: DB, request: Request) -> dict[str, Any]:
    engine = request.app.state.engines.engine
    return {
        "user": user_out(user).model_dump(mode="json"),
        "preferences": user.recommendation_prefs or {},
        **taste_profile(db, user, engine),
    }


def _page(q: Any, db: DB, page: int, size: int) -> tuple[list[Any], int]:
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    return list(db.execute(q.limit(size).offset((page - 1) * size)).unique().scalars().all()), total


@router.get("/me/ratings")
def my_ratings(
    user: CurrentUser, db: DB, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)
) -> dict[str, Any]:
    q = select(Rating).where(Rating.user_id == user.id).order_by(Rating.updated_at.desc(), Rating.id.desc())
    rows, total = _page(q, db, page, page_size)
    return {
        "items": [
            {
                "movie": MovieBrief.model_validate(r.movie).model_dump(),
                "rating": r.rating,
                "rated_at": r.updated_at,
            }
            for r in rows
        ],
        "total": total,
        "page": page,
    }


@router.get("/me/history")
def my_history(
    user: CurrentUser, db: DB, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)
) -> dict[str, Any]:
    q = (
        select(WatchHistory)
        .where(WatchHistory.user_id == user.id)
        .order_by(WatchHistory.watched_at.desc(), WatchHistory.id.desc())
    )
    rows, total = _page(q, db, page, page_size)
    ratings = dict(db.execute(select(Rating.movie_id, Rating.rating).where(Rating.user_id == user.id)).all())
    return {
        "items": [
            {
                "id": w.id,
                "movie": MovieBrief.model_validate(w.movie).model_dump(),
                "watched_at": w.watched_at,
                "rating": ratings.get(w.movie_id),
            }
            for w in rows
        ],
        "total": total,
        "page": page,
    }


@router.get("/me/favorites")
def my_favorites(
    user: CurrentUser, db: DB, page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200)
) -> dict[str, Any]:
    q = (
        select(Favorite)
        .where(Favorite.user_id == user.id)
        .order_by(Favorite.created_at.desc(), Favorite.id.desc())
    )
    rows, total = _page(q, db, page, page_size)
    return {
        "items": [
            {
                "movie": MovieBrief.model_validate(f.movie).model_dump(),
                "source": f.source,
                "added_at": f.created_at,
            }
            for f in rows
        ],
        "total": total,
        "page": page,
    }
