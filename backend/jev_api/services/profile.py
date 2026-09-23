"""Build ML profiles and human-facing taste profiles from a user's stored interactions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from jev_api.models import (
    Favorite,
    Genre,
    Movie,
    Rating,
    RecommendationFeedback,
    User,
    UserGenrePreference,
    WatchHistory,
)
from jev_ml.engine import Interaction, RecommendationEngine
from jev_ml.signals import LIKE_THRESHOLD, UserProfile


@dataclass
class UserState:
    interactions: list[Interaction]
    genres: list[str]
    excluded: list[int]  # "not interested" / disliked recommendations
    consumed: set[int]
    counts: dict[str, int]


def load_user_state(db: Session, user: User) -> UserState:
    ratings = db.execute(
        select(Rating.movie_id, Rating.rating, Rating.updated_at).where(Rating.user_id == user.id)
    ).all()
    favs = db.execute(
        select(Favorite.movie_id, Favorite.source, Favorite.created_at).where(Favorite.user_id == user.id)
    ).all()
    watches = db.execute(
        select(WatchHistory.movie_id, WatchHistory.watched_at).where(WatchHistory.user_id == user.id)
    ).all()
    negative = db.scalars(
        select(RecommendationFeedback.movie_id).where(
            RecommendationFeedback.user_id == user.id,
            RecommendationFeedback.feedback.in_(("dislike", "not_interested")),
        )
    ).all()
    genres = db.scalars(
        select(Genre.name)
        .join(UserGenrePreference, UserGenrePreference.genre_id == Genre.id)
        .where(UserGenrePreference.user_id == user.id)
        .order_by(Genre.name)
    ).all()

    events: list[Interaction] = []
    events += [Interaction(m, "rating", float(r), ts.timestamp()) for m, r, ts in ratings]
    events += [
        Interaction(m, "onboarding" if src == "onboarding" else "favorite", None, ts.timestamp())
        for m, src, ts in favs
    ]
    events += [Interaction(m, "watch", None, ts.timestamp()) for m, ts in watches]
    consumed = {e.movie_id for e in events}
    return UserState(
        events,
        list(genres),
        sorted(set(negative)),
        consumed,
        {
            "ratings": len(ratings),
            "favorites": len(favs),
            "watches": len(watches),
            "negative_feedback": len(set(negative)),
        },
    )


def build_ml_profile(engine: RecommendationEngine, state: UserState) -> UserProfile:
    return engine.build_profile(
        state.interactions, genre_prefs=state.genres, excluded_movie_ids=state.excluded
    )


def taste_profile(db: Session, user: User, engine: RecommendationEngine | None) -> dict[str, Any]:
    """Preferences derived only from the user's own interactions (nothing is inferred elsewhere)."""
    state = load_user_state(db, user)
    liked_ids = [
        e.movie_id
        for e in state.interactions
        if (e.kind == "rating" and (e.value or 0) >= LIKE_THRESHOLD) or e.kind in ("favorite", "onboarding")
    ]
    movies = (
        {m.id: m for m in db.scalars(select(Movie).where(Movie.id.in_(liked_ids))).all()} if liked_ids else {}
    )
    genre_c: Counter[str] = Counter()
    director_c: Counter[str] = Counter()
    actor_c: Counter[str] = Counter()
    for mid in liked_ids:
        m = movies.get(mid)
        if m is None:
            continue
        genre_c.update(g.name for g in m.genres)
        director_c.update(m.directors)
        actor_c.update(m.cast)
    ratings = [e.value for e in state.interactions if e.kind == "rating" and e.value is not None]
    hist = Counter(ratings)
    out: dict[str, Any] = {
        "explicit_genres": state.genres,
        "genre_affinity": [{"genre": g, "count": c} for g, c in genre_c.most_common(12)],
        "favorite_directors": [{"name": d, "count": c} for d, c in director_c.most_common(6) if c >= 1],
        "favorite_actors": [{"name": a, "count": c} for a, c in actor_c.most_common(8) if c >= 2],
        "rating_histogram": [{"rating": r / 2, "count": hist.get(r / 2, 0)} for r in range(1, 11)],
        "mean_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "counts": state.counts,
        "liked_count": len(liked_ids),
    }
    if engine is not None:
        profile = build_ml_profile(engine, state)
        weights = engine.effective_weights(profile)
        gvec = engine.ranker.genre_preference_vector(profile)
        out["effective_weights"] = {k: round(v, 4) for k, v in weights.items()}
        out["preference_vector"] = sorted(
            (
                {"genre": g, "weight": round(float(w), 4)}
                for g, w in zip(engine.ranker.genres, gvec, strict=True)
                if abs(w) > 1e-6
            ),
            key=lambda d: -d["weight"],
        )
        n = profile.n_interactions
        out["stage"] = "cold" if n == 0 else "warming" if n < engine.config.behavioral_ramp else "warm"
        out["behavioral_ramp"] = engine.config.behavioral_ramp
        out["model_version"] = engine.version
    return out
