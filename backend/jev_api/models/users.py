"""Accounts and their onboarding genre preferences."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from jev_api.models.base import Base, TimestampMixin, utcnow
from jev_api.models.catalog import Genre


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    onboarding_completed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # bumped on every taste-relevant event; part of the recommendation cache key
    profile_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    recommendation_prefs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # security (migration 0010): every JWT carries the version it was issued under; bumping it (revoke-all,
    # password change, role change) invalidates all of the account's sessions at once
    token_version: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

    genre_preferences: Mapped[list[UserGenrePreference]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UserGenrePreference(Base):
    __tablename__ = "user_genre_preferences"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    genre_id: Mapped[int] = mapped_column(ForeignKey("genres.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="genre_preferences")
    genre: Mapped[Genre] = relationship(lazy="joined")
