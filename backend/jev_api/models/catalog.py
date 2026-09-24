"""The movie catalogue: movies, genres and the movie-genre link."""

from __future__ import annotations

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from jev_api.models.base import Base, TimestampMixin


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)


class Movie(TimestampMixin, Base):
    __tablename__ = "movies"

    # MovieLens movieId, so ids line up with model artifacts
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    year: Mapped[int | None] = mapped_column(Integer, index=True)
    description: Mapped[str | None] = mapped_column(Text)
    runtime_min: Mapped[int | None] = mapped_column(Integer)
    directors: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    cast: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    countries: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    imdb_id: Mapped[str | None] = mapped_column(String(16))
    tmdb_id: Mapped[int | None] = mapped_column(Integer)
    n_ratings: Mapped[int] = mapped_column(Integer, default=0, nullable=False, index=True)
    mean_rating: Mapped[float | None] = mapped_column(Float)
    # lower-cased title + directors + cast, for simple portable search
    search_text: Mapped[str] = mapped_column(Text, default="", nullable=False)

    genres: Mapped[list[Genre]] = relationship(
        secondary="movie_genres", lazy="selectin", order_by="Genre.name"
    )


class MovieGenre(Base):
    __tablename__ = "movie_genres"

    movie_id: Mapped[int] = mapped_column(ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    genre_id: Mapped[int] = mapped_column(
        ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True, index=True
    )
