"""Intelligence layer schemas (docs/intelligence.md, sections 6 and 9.3). Owned by WS4."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Run-backed objects (signals, trends, ...) are passed through from PipelineResult.to_dict(), whose
# contract lives in ml/jev_ml/intel; DB-backed objects are typed here.
WarningStatus = Literal["new", "acknowledged", "investigating", "resolved", "dismissed"]
Severity = Literal["low", "medium", "high", "critical"]
FeedbackTarget = Literal["decision", "warning", "action", "prediction"]
Verdict = Literal["correct", "incorrect", "useful", "not_useful", "false_positive"]


class IntelPage(BaseModel):
    items: list[dict[str, Any]]
    total: int
    limit: int
    offset: int
    run_id: str | None
    as_of: str | None
    domain: str = "movie"
    mode: str = "live"  # P2.4: the mode of the run the page was read from


DomainKey = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9:_-]{0,63}$")]


class RunTrigger(BaseModel):
    # v1.2: the domain adapter to run ("movie" when omitted, or ?domain=)
    domain: DomainKey | None = None
    # "YYYY-MM-DD" (UTC midnight) or a full ISO timestamp; omitted = the domain's default (movie: the
    # last MovieLens event; generic: now)
    as_of: str | None = Field(default=None, max_length=40)
    # replays only (P2.4 + WS1): events ingested after this are unknown to the run; default = as_of
    knowledge_time: str | None = Field(default=None, max_length=40)


class IntelRunOut(BaseModel):
    id: int
    run_id: str
    domain: str = "movie"
    trigger: str
    status: str
    as_of: str | None
    started_at: str
    finished_at: str | None
    duration_ms: float | None
    pipeline_version: str
    data_version: str | None
    model_version: str | None
    summary: dict[str, Any] | None
    stage_ms: dict[str, Any]
    error: str | None
    event_watermark: dict[str, Any] | None = None  # WS1: the events the run read (lineage)
    mode: str = "live"  # P2.4: "replay" when the run was asked for an explicit as_of


class IntelRunList(BaseModel):
    items: list[IntelRunOut]
    total: int


class WarningEventOut(BaseModel):
    from_status: str | None
    to_status: str
    note: str | None
    actor: str
    at: str


class IntelWarningOut(BaseModel):
    id: int
    key: str
    title: str
    description: str
    severity: str
    confidence: float | None
    confidence_kind: str | None
    status: str
    trigger: dict[str, Any]
    evidence: list[Any]
    recommended_action: str
    source: dict[str, Any]
    entity_type: str | None
    entity: str | None
    detected_at: str
    last_seen_at: str
    updated_at: str
    occurrences: int
    first_seen_run_id: str | None
    last_seen_run_id: str | None
    reopened_from: int | None
    suppressed_until: str | None = None
    history: list[WarningEventOut] | None = None
    # v1.2 (docs/platform.md, section 4)
    domain: str = "movie"
    decision_id: str | None = None
    early_warning_level: str | None = None


class IntelWarningList(BaseModel):
    items: list[IntelWarningOut]
    total: int
    limit: int
    offset: int


class WarningUpdate(BaseModel):
    status: WarningStatus
    note: str | None = Field(default=None, max_length=1000)


class DecisionFeedbackCounts(BaseModel):
    correct: int = 0
    incorrect: int = 0


class IntelDecisionOut(BaseModel):
    id: str
    key: str
    domain: str = "movie"
    spec_id: str
    policy_version: str
    question: str
    kind: str
    options: list[str]
    answer: str | float | None  # a number for kind "score" (section 9.1)
    option_scores: dict[str, Any]
    confidence: float | None
    confidence_kind: str
    state: dict[str, Any]
    rationale: list[str]
    evidence: list[Any]
    abstained: bool
    fallback_reason: str | None
    entity_type: str | None
    entity: str | None
    batch_id: str | None = None
    answer_value: float | None = None
    answer_interval: list[float | None] | None = None
    scale: dict[str, Any] | None = None
    db_id: int
    run_id: str
    as_of: str
    created_at: str
    feedback: DecisionFeedbackCounts


class IntelDecisionList(BaseModel):
    items: list[IntelDecisionOut]
    total: int
    limit: int
    offset: int


class IntelFeedbackIn(BaseModel):
    target_type: FeedbackTarget
    target_id: str = Field(min_length=1, max_length=64)
    verdict: Verdict
    note: str | None = Field(default=None, max_length=1000)
    outcome: str | None = Field(default=None, max_length=500)


class IntelFeedbackOut(BaseModel):
    id: int
    domain: str = "movie"
    target_type: str
    target_id: str
    verdict: str
    note: str | None
    outcome: str | None
    actor: str
    created_at: str


class IntelFeedbackList(BaseModel):
    items: list[IntelFeedbackOut]
    total: int
    limit: int
    offset: int
    summary: dict[str, dict[str, Any]]


class ScenarioSpecIn(BaseModel):
    # value ranges (and NaN/Infinity) are validated by run_scenario (its messages come back as 422);
    # unknown keys are dropped, so a saved scenario never stores arbitrary client data
    model_config = ConfigDict(extra="ignore")

    name: str | None = Field(default=None, max_length=80)
    kind: str = Field(max_length=16)
    trend_multiplier: float | None = None
    level_shift_pct: float | None = None
    shock_month: int | None = None


class ScenarioRequest(BaseModel):
    series_id: str = Field(min_length=1, max_length=200)
    horizon_months: int = 12
    as_of: str | None = Field(default=None, max_length=40)
    scenarios: list[ScenarioSpecIn] = Field(default_factory=list, max_length=6)
    trend_source: Literal["auto", "model"] | None = None
    save: bool = False
    title: str | None = Field(default=None, max_length=200)


class SavedScenarioOut(BaseModel):
    id: int
    domain: str = "movie"
    title: str
    series_id: str
    created_at: str
    input: dict[str, Any]
    output: dict[str, Any]


class SavedScenarioList(BaseModel):
    items: list[SavedScenarioOut]
    total: int


# --- v1.1 (docs/intelligence.md, section 9.3) -------------------------------------------------------
EvidenceOwner = Literal["signal", "trend", "anomaly", "forecast", "risk", "decision", "warning", "action"]
HistoryEntity = Literal["signals", "risks", "trends", "anomalies"]


class IntelEvidenceOut(BaseModel):
    id: int
    kind: str
    label: str
    value: float | str | None
    detail: str | None
    ref: str | None
    owner_type: str
    owner_id: str
    owner_title: str
    position: int
    run_id: str


class IntelEvidencePage(BaseModel):
    items: list[IntelEvidenceOut]
    total: int
    limit: int
    offset: int
    run_id: str | None
    as_of: str | None
    domain: str = "movie"


class HistoryPoint(BaseModel):
    run_id: str
    as_of: str | None
    created_at: str
    id: str
    observed_at: str | None
    value: float | None
    score: float | None
    level: str | None
    direction: str | None


class IntelHistory(BaseModel):
    entity: str
    key: str
    items: list[HistoryPoint]
    domain: str = "movie"


class DecisionBatchList(BaseModel):
    items: list[dict[str, Any]]
    run_id: str
    as_of: str | None
    domain: str = "movie"
    mode: str = "live"


class IntelLineage(BaseModel):
    """GET /intel/decisions/{id}/lineage and /intel/warnings/{id}/lineage (docs/DECISION_ENGINE.md)."""

    version: str
    root: dict[str, Any]
    run: dict[str, Any]  # mode, versions, config_hash, input_fingerprint, event_watermark
    evidence: list[dict[str, Any]]  # the root's evidence items with `resolves_to`
    nodes: list[dict[str, Any]]  # {id "<type>:<ref>", type, ref, title, ...}
    edges: list[dict[str, Any]]  # {from, to, relation}
    series: list[str]
    sources: list[str]
    node_types: list[str]
    unresolved: list[dict[str, Any]]
    complete: bool
    event_watermark: dict[str, Any] | None = None


class EvaluationRunOut(BaseModel):
    id: int
    run_dir: str
    created_at: str | None
    pipeline_version: str | None
    data_version: str | None
    headline: dict[str, float | None]


class EvaluationRunList(BaseModel):
    items: list[EvaluationRunOut]
    total: int
