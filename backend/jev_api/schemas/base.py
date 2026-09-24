"""Shared schema building blocks (integrator-owned)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

# ids in request bodies reach SQL: bounded to the INTEGER primary keys (jev_api.deps.MAX_DB_INT)
DbId = Annotated[int, Field(ge=0, le=2**31 - 1)]


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)
