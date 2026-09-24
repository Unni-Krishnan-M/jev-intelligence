"""Catalogue and interaction schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from jev_api.schemas.base import ORM


class MovieBrief(ORM):
    id: int
    title: str
    year: int | None
    genres: list[str]
    directors: list[str] = []
    n_ratings: int
    mean_rating: float | None

    @field_validator("genres", mode="before")
    @classmethod
    def _genre_names(cls, v: Any) -> Any:
        return [g.name if hasattr(g, "name") else g for g in v]


class MovieDetail(MovieBrief):
    description: str | None
    runtime_min: int | None
    cast: list[str]
    keywords: list[str]
    tags: list[str]
    countries: list[str]
    imdb_id: str | None
    tmdb_id: int | None
    user_rating: float | None = None
    is_favorite: bool = False
    watched: bool = False
    in_model: bool = True


class MoviePage(BaseModel):
    items: list[MovieBrief]
    total: int
    page: int
    page_size: int


class RateRequest(BaseModel):
    rating: float = Field(ge=0.5, le=5.0)

    @field_validator("rating")
    @classmethod
    def _half_steps(cls, v: float) -> float:
        if (v * 2) % 1 != 0:
            raise ValueError("rating must be a multiple of 0.5")
        return v


class FavoriteRequest(BaseModel):
    favorite: bool = True


class InteractionResult(BaseModel):
    movie_id: int
    rating: float | None = None
    is_favorite: bool | None = None
    watched: bool | None = None
    profile_version: int


class MovieCreate(BaseModel):
    """Admin: add a movie that is not in the trained model (item cold start)."""

    id: int = Field(gt=0, le=2**31 - 1)
    title: str = Field(min_length=1, max_length=300)
    year: int | None = Field(default=None, ge=1870, le=2100)
    genres: list[str] = Field(default_factory=list, max_length=20)
    directors: list[str] = Field(default_factory=list, max_length=20)
    cast: list[str] = Field(default_factory=list, max_length=50)
    keywords: list[str] = Field(default_factory=list, max_length=50)
    description: str | None = Field(default=None, max_length=2000)
