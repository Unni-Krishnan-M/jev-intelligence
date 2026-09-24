"""Shared schema building blocks (integrator-owned)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

# ids in request bodies reach SQL: bounded to the INTEGER primary keys (jev_api.deps.MAX_DB_INT)
DbId = Annotated[int, Field(ge=0, le=2**31 - 1)]


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


# SQLite returns naive datetimes (stored as UTC); responses always carry an explicit offset
UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
