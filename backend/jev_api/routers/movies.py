from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from sqlalchemy import func, or_, select

from jev_api.deps import DB, MAX_COUNT, MAX_PAGE, AdminUser, CurrentUser, Engine, IdPath, OptionalUser
from jev_api.models import (
    DEFAULT_DOMAIN,
    Favorite,
    Genre,
    Movie,
    MovieGenre,
    Rating,
    User,
    WatchHistory,
    utcnow,
)
from jev_api.schemas import (
    FavoriteRequest,
    InteractionResult,
    MovieBrief,
    MovieCreate,
    MovieDetail,
    MoviePage,
    RateRequest,
)
from jev_api.schemas.events import IDEMPOTENCY_KEY_PATTERN
from jev_api.services import events

router = APIRouter(tags=["movies"])

SortKey = Literal["popular", "rating", "year", "title"]


def _get_movie(db: DB, movie_id: int) -> Movie:
    movie = db.get(Movie, movie_id)
    if movie is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "movie not found")
    return movie


@router.get("/genres", response_model=list[dict])
def list_genres(db: DB) -> list[dict]:
    rows = db.execute(
        select(Genre.name, func.count(MovieGenre.movie_id))
        .join(MovieGenre, MovieGenre.genre_id == Genre.id, isouter=True)
        .group_by(Genre.name)
        .order_by(Genre.name)
    ).all()
    return [{"name": n, "movie_count": c} for n, c in rows]


@router.get("/movies", response_model=MoviePage)
def list_movies(
    db: DB,
    page: int = Query(1, ge=1, le=MAX_PAGE),
    page_size: int = Query(24, ge=1, le=100),
    genre: str | None = Query(None, max_length=64),
    year_min: int | None = Query(None, ge=1800, le=2100),
    year_max: int | None = Query(None, ge=1800, le=2100),
    min_ratings: int = Query(0, ge=0, le=MAX_COUNT),
    sort: SortKey = "popular",
) -> MoviePage:
    q = select(Movie)
    if genre:
        q = (
            q.join(MovieGenre, MovieGenre.movie_id == Movie.id)
            .join(Genre, Genre.id == MovieGenre.genre_id)
            .where(Genre.name == genre)
        )
    if year_min is not None:
        q = q.where(Movie.year >= year_min)
    if year_max is not None:
        q = q.where(Movie.year <= year_max)
    if min_ratings:
        q = q.where(Movie.n_ratings >= min_ratings)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    order = {
        "popular": (Movie.n_ratings.desc(), Movie.id),
        "rating": (func.coalesce(Movie.mean_rating, 0).desc(), Movie.n_ratings.desc(), Movie.id),
        "year": (func.coalesce(Movie.year, 0).desc(), Movie.id),
        "title": (Movie.title, Movie.id),
    }[sort]
    if sort == "rating" and not min_ratings:
        q = q.where(Movie.n_ratings >= 10)  # a 5.0 average over one vote is not a "top rated" film
    items = db.scalars(q.order_by(*order).limit(page_size).offset((page - 1) * page_size)).all()
    return MoviePage(
        items=[MovieBrief.model_validate(m) for m in items], total=total, page=page, page_size=page_size
    )


@router.get("/movies/search", response_model=list[MovieBrief])
def search_movies(
    db: DB, q: str = Query(..., min_length=1, max_length=100), limit: int = Query(20, ge=1, le=50)
) -> list[MovieBrief]:
    # parameterized LIKE (SQLAlchemy binds the value); escape LIKE wildcards in user input
    term = q.strip().lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    if not term:
        return []
    starts = func.lower(Movie.title).like(f"{term}%", escape="\\")
    rows = db.scalars(
        select(Movie)
        .where(
            or_(
                func.lower(Movie.title).like(f"%{term}%", escape="\\"),
                Movie.search_text.like(f"%{term}%", escape="\\"),
            )
        )
        .order_by(starts.desc(), Movie.n_ratings.desc(), Movie.id)
        .limit(limit)
    ).all()
    return [MovieBrief.model_validate(m) for m in rows]


@router.get("/movies/popular", response_model=list[MovieBrief])
def popular_movies(
    db: DB, engine: Engine, limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=1000)
) -> list[MovieBrief]:
    """Highest Bayesian-average rating (model popularity component), not raw counts."""
    rows = engine.popular(k=limit, offset=offset)
    movies = {m.id: m for m in db.scalars(select(Movie).where(Movie.id.in_([r["movie_id"] for r in rows])))}
    return [MovieBrief.model_validate(movies[r["movie_id"]]) for r in rows if r["movie_id"] in movies]


@router.get("/movies/{movie_id}", response_model=MovieDetail)
def get_movie(movie_id: IdPath, db: DB, user: OptionalUser, engine: Engine) -> MovieDetail:
    m = _get_movie(db, movie_id)
    detail = MovieDetail.model_validate(m)
    detail.in_model = engine.item_index.index_of(movie_id) is not None
    if user is not None:
        detail.user_rating = db.scalar(
            select(Rating.rating).where(Rating.user_id == user.id, Rating.movie_id == movie_id)
        )
        detail.is_favorite = (
            db.scalar(select(Favorite.id).where(Favorite.user_id == user.id, Favorite.movie_id == movie_id))
            is not None
        )
        detail.watched = (
            db.scalar(
                select(WatchHistory.id)
                .where(WatchHistory.user_id == user.id, WatchHistory.movie_id == movie_id)
                .limit(1)
            )
            is not None
        )
    return detail


# --- member writes: an event first, the current-state row is its projection (services/events.py) ---------
IdempotencyKeyHeader = Annotated[
    str | None,
    Header(
        alias="Idempotency-Key",
        min_length=1,
        max_length=200,
        pattern=IDEMPOTENCY_KEY_PATTERN,
        description="optional: a retry with the same key returns the first response and writes nothing",
    ),
]


def _interaction(
    request: Request,
    db: DB,
    user: User,
    key: str | None,
    route: str,
    body: dict[str, Any],
    work: Callable[[datetime], dict[str, Any]],
) -> InteractionResult:
    fp = events.fingerprint(request.method, route, body)

    def run() -> dict[str, Any]:
        now = utcnow()
        out = work(now)
        user.profile_version += 1
        out["profile_version"] = user.profile_version
        return out

    try:
        response, replayed = events.idempotent(db, events.user_scope(user), key, fp, run)
    except events.IdempotencyMismatchError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    if not replayed:
        events.refresher_for(request.app).mark_dirty(DEFAULT_DOMAIN)
    return InteractionResult(**response)


@router.post("/movies/{movie_id}/rate", response_model=InteractionResult)
def rate_movie(
    movie_id: IdPath,
    body: RateRequest,
    user: CurrentUser,
    db: DB,
    request: Request,
    key: IdempotencyKeyHeader = None,
) -> InteractionResult:
    _get_movie(db, movie_id)

    def work(now: datetime) -> dict[str, Any]:
        events.apply_member_event(db, user, "rating", movie_id, value=body.rating, key=key, now=now)
        return {"movie_id": movie_id, "rating": body.rating}

    return _interaction(request, db, user, key, f"/movies/{movie_id}/rate", body.model_dump(), work)


@router.delete("/movies/{movie_id}/rate", response_model=InteractionResult)
def unrate_movie(
    movie_id: IdPath, user: CurrentUser, db: DB, request: Request, key: IdempotencyKeyHeader = None
) -> InteractionResult:
    def work(now: datetime) -> dict[str, Any]:
        if db.get(Movie, movie_id) is not None:  # an unknown movie has no rating: nothing to record
            events.apply_member_event(db, user, "rating_removed", movie_id, key=key, now=now)
        return {"movie_id": movie_id, "rating": None}

    return _interaction(request, db, user, key, f"/movies/{movie_id}/rate", {}, work)


@router.post("/movies/{movie_id}/favorite", response_model=InteractionResult)
def favorite_movie(
    movie_id: IdPath,
    user: CurrentUser,
    db: DB,
    request: Request,
    body: FavoriteRequest | None = None,
    key: IdempotencyKeyHeader = None,
) -> InteractionResult:
    _get_movie(db, movie_id)
    want = True if body is None else body.favorite

    def work(now: datetime) -> dict[str, Any]:
        etype = "favorite" if want else "unfavorite"
        events.apply_member_event(db, user, etype, movie_id, key=key, now=now)
        return {"movie_id": movie_id, "is_favorite": want}

    return _interaction(request, db, user, key, f"/movies/{movie_id}/favorite", {"favorite": want}, work)


@router.post("/movies/{movie_id}/watch", response_model=InteractionResult)
def watch_movie(
    movie_id: IdPath, user: CurrentUser, db: DB, request: Request, key: IdempotencyKeyHeader = None
) -> InteractionResult:
    """Watches are additive. A keyless watch identical to one recorded in the last
    events_watch_dedupe_seconds is a client retry: it returns the current state and appends nothing."""
    _get_movie(db, movie_id)
    if key is None:
        window = request.app.state.settings.events_watch_dedupe_seconds
        if events.recent_watch(db, user.id, movie_id, window, utcnow()):
            events.count_ingest(db, DEFAULT_DOMAIN, utcnow().date(), duplicates=1)
            db.commit()
            return InteractionResult(movie_id=movie_id, watched=True, profile_version=user.profile_version)

    def work(now: datetime) -> dict[str, Any]:
        events.apply_member_event(db, user, "watch", movie_id, key=key, now=now)
        return {"movie_id": movie_id, "watched": True}

    return _interaction(request, db, user, key, f"/movies/{movie_id}/watch", {}, work)


@router.post("/movies", response_model=MovieDetail, status_code=status.HTTP_201_CREATED)
def create_movie(body: MovieCreate, _: AdminUser, db: DB, engine: Engine) -> MovieDetail:
    """Add a catalog entry not seen by the model; it's discoverable via metadata similarity."""
    if db.get(Movie, body.id) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "movie id already exists")
    genres = db.scalars(select(Genre).where(Genre.name.in_(body.genres))).all()
    m = Movie(
        id=body.id,
        title=body.title,
        year=body.year,
        description=body.description,
        directors=body.directors,
        cast=body.cast,
        keywords=body.keywords,
        tags=[],
        countries=[],
        n_ratings=0,
        search_text=" ".join([body.title, *body.directors, *body.cast[:10]]).lower(),
    )
    m.genres = list(genres)
    db.add(m)
    db.commit()
    db.refresh(m)
    detail = MovieDetail.model_validate(m)
    detail.in_model = engine.item_index.index_of(m.id) is not None
    return detail
