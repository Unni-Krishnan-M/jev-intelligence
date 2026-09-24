"""Trained model versions and offline experiments with their evaluation metrics (synced from models/ and
experiments/). Online experiments live in models/experiments.py, governance state in models/governance.py."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from jev_api.models.base import Base, utcnow


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(120), nullable=False)
    training_config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    training_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    artifact_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    experiment_name: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    dataset_version: Mapped[str] = mapped_column(String(120), nullable=False)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    training_seed: Mapped[int] = mapped_column(Integer, nullable=False)
    split: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    metrics: Mapped[list[EvaluationMetric]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", lazy="selectin"
    )
    model_version: Mapped[ModelVersion | None] = relationship(lazy="joined")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    __table_args__ = (
        UniqueConstraint("experiment_id", "model_name", "protocol", "metric", "k", name="uq_metric"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    model_name: Mapped[str] = mapped_column(String(32), nullable=False)
    protocol: Mapped[str] = mapped_column(String(16), nullable=False)  # test | cold_start
    metric: Mapped[str] = mapped_column(String(32), nullable=False)
    k: Mapped[int] = mapped_column(Integer, default=0, nullable=False)  # 0 = metric without a cut-off
    value: Mapped[float] = mapped_column(Float, nullable=False)

    experiment: Mapped[Experiment] = relationship(back_populates="metrics")
