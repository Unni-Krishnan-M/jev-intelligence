"""Model governance schemas: snapshots, training jobs, governed models, promotion and rollback (WS2)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from jev_api.schemas.base import ORM, UtcDatetime

# a config for POST /governance/retrain is a file name inside configs/, never a path
CONFIG_NAME = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}\.ya?ml$"


class SnapshotOut(ORM):
    id: int
    snapshot_id: str
    content_hash: str
    base_dataset_version: str | None
    cutoff: UtcDatetime | None
    row_counts: dict[str, Any]
    sources: dict[str, Any]
    watermark: dict[str, Any]
    semantics_version: str
    created_by: str
    created_at: UtcDatetime


class TrainingJobOut(ORM):
    id: int
    job_id: str
    kind: str
    trigger: str
    status: str
    quick: bool
    config_path: str | None
    decision_id: str | None
    snapshot_id: str | None
    model_version: str | None
    incumbent_version: str | None
    gate_passed: bool | None
    promoted: bool
    requested_by: str
    created_at: UtcDatetime
    started_at: UtcDatetime | None
    finished_at: UtcDatetime | None
    duration_ms: float | None
    steps: list[Any]
    error: str | None


class GateSummary(BaseModel):
    """One gate: status (pass | fail | skipped), the reason when it failed and its key numbers."""

    status: str
    reason: str | None = None
    numbers: dict[str, Any] = {}


class GovernedModelOut(BaseModel):
    version: str
    state: str
    is_active: bool
    # "job": a training_jobs row of this database produced it (job_id links to GET /governance/jobs);
    # "artifact": lineage and gate were read from models/<version>/ files, job_id is not a row here
    source: Literal["job", "artifact"]
    model_version_id: int | None
    trained_at: UtcDatetime | None
    dataset_version: str | None
    snapshot_id: str | None
    job_id: str | None
    decision_id: str | None
    metrics: dict[str, float]  # training-time test metrics of the hybrid (headline @10)
    gate_passed: bool | None
    gate_reasons: list[str]
    gated_against: str | None
    gated_at: UtcDatetime | None
    gates: dict[str, GateSummary]
    promotable: bool  # promotable now without force
    blockers: list[str]  # why not (empty when promotable)
    promoted_at: UtcDatetime | None
    promoted_by: str | None
    forced: bool
    force_reason: str | None


class GovernedModelList(BaseModel):
    active: str | None
    previous: str | None  # what POST /governance/models/rollback restores
    items: list[GovernedModelOut]


class RetrainRequest(BaseModel):
    quick: bool | None = None  # None = JEV_GOVERNANCE_QUICK_DEFAULT
    config: str | None = Field(None, pattern=CONFIG_NAME)  # a file in configs/ (default experiment.yaml)


class EvaluateRequest(BaseModel):
    quick: bool | None = None


class PromoteRequest(BaseModel):
    force: bool = False
    reason: str | None = Field(None, min_length=3, max_length=500)

    @model_validator(mode="after")
    def _force_needs_reason(self) -> PromoteRequest:
        if self.force and not (self.reason and self.reason.strip()):
            raise ValueError("force requires a reason")
        return self


class RollbackRequest(BaseModel):
    """A rollback changes what every member is served, so the API requires a reason (audited)."""

    reason: str = Field(min_length=3, max_length=500)

    @model_validator(mode="after")
    def _reason_not_blank(self) -> RollbackRequest:
        if len(self.reason.strip()) < 3:
            raise ValueError("reason must have at least 3 non-blank characters")
        self.reason = self.reason.strip()
        return self


class PromotionResult(BaseModel):
    version: str
    previous: str | None
    action: str  # promote | rollback
    forced: bool
    gate_passed: bool | None
