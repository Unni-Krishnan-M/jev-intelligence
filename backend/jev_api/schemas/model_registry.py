"""Model versions and offline experiments (admin /models, /experiments)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import field_validator

from jev_api.schemas.base import ORM


class ModelVersionOut(ORM):
    id: int
    version: str
    model_type: str
    dataset_version: str
    training_seed: int
    metrics: dict[str, Any]
    artifact_path: str
    artifact_bytes: int
    trained_at: datetime
    is_active: bool


class ModelVersionDetail(ModelVersionOut):
    training_config: dict[str, Any]
    manifest: dict[str, Any] = {}


class MetricOut(ORM):
    model_name: str
    protocol: str
    metric: str
    k: int
    value: float


class ExperimentOut(ORM):
    id: int
    run_id: str
    experiment_name: str
    dataset_version: str
    model_type: str
    training_seed: int
    created_at: datetime
    model_version: str | None = None
    headline: dict[str, float] = {}

    @field_validator("model_version", mode="before")
    @classmethod
    def _version_name(cls, v: Any) -> Any:
        return getattr(v, "version", v)


class ExperimentDetail(ExperimentOut):
    parameters: dict[str, Any]
    split: dict[str, Any]
    summary: dict[str, Any]
    metrics: list[MetricOut]
