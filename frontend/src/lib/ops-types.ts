/**
 * Types for the Phase 2 operations endpoints (admin only): the event log, model governance and
 * online experiments, plus the decision lineage graph. They mirror the backend schemas field for
 * field (backend/jev_api/schemas/{events,governance,experiments,intel}.py and the services that
 * fill their `dict[str, Any]` fields); change them only when the API changes.
 */

// ---- events: GET /events/health, POST /events/replay ----------------------------------------

export interface IngestCounts {
  accepted: number;
  duplicates: number;
  rejected: number;
}

/** intel_runs.event_watermark: the newest event a run could see (services/events.watermark). */
export interface EventWatermark {
  domain: string;
  as_of: string | null;
  knowledge_time: string | null;
  max_event_id: number | null;
  max_ingested_at: string | null;
  max_event_time: string | null;
  n_events: number;
}

export interface LastLiveRun {
  run_id: string;
  trigger: string;
  finished_at: string | null;
}

export interface RefreshState {
  dirty: boolean;
  /** null until the refresher has run this domain in this process */
  last_refresh: { at: string; status: string | null; run_id?: string | null; error?: string } | null;
}

/** One entry of GET /events/health `domains` (services/events.health). */
export interface EventDomainHealth {
  domain: string;
  events: number;
  max_event_id: number | null;
  /** absent for a domain that only has daily counters (no rows in the log) */
  max_event_time?: string | null;
  max_ingested_at?: string | null;
  /** now − newest event_time, seconds */
  event_time_lag_s?: number | null;
  /** now − newest ingested_at, seconds: silence on the feed */
  ingest_lag_s?: number | null;
  today: IngestCounts;
  last_7_days: IngestCounts;
  last_live_run: LastLiveRun | null;
  watermark: EventWatermark | null;
  events_since_last_run: number;
  /** present when the API process has a refresher */
  refresh?: RefreshState;
}

export interface RefresherSummary {
  enabled: boolean;
  debounce_seconds?: number;
  max_delay_seconds?: number;
  dirty_domains?: string[];
  running?: boolean;
}

export interface EventHealth {
  generated_at: string;
  domains: EventDomainHealth[];
  refresher: RefresherSummary;
}

export interface ProjectionDiff {
  missing: number;
  extra: number;
  changed: number;
  sample: string[];
}

export interface ReplayDiff {
  ratings: ProjectionDiff;
  favorites: ProjectionDiff;
  recommendation_feedback: ProjectionDiff;
  watch_history: ProjectionDiff;
  consistent: boolean;
}

export interface ReplayIn {
  apply: boolean;
}

export interface ReplayOut {
  applied: boolean;
  consistent_before: boolean;
  consistent_after: boolean;
  diff: ReplayDiff;
}

// ---- governance -------------------------------------------------------------------------------

export type ModelState = "candidate" | "active" | "retired" | "rejected";
export type GateStatus = "pass" | "fail" | "skipped";
export type JobKind = "retrain" | "evaluate";
export type JobStatus = "queued" | "running" | "succeeded" | "failed";
export type JobTrigger = "manual" | "schedule" | "decision" | "cli";

export interface SnapshotOut {
  id: number;
  snapshot_id: string;
  content_hash: string;
  base_dataset_version: string | null;
  cutoff: string | null;
  row_counts: Record<string, number>;
  sources: Record<string, unknown>;
  watermark: Record<string, unknown>;
  semantics_version: string;
  created_by: string;
  created_at: string;
}

/** One entry of TrainingJob.steps (services/governance._step): the step name, when, its key numbers. */
export interface JobStep {
  step: "snapshot" | "train" | "calibrate" | "gate" | "promote" | string;
  at: string;
  [info: string]: unknown;
}

export interface TrainingJobOut {
  id: number;
  job_id: string;
  kind: JobKind;
  trigger: JobTrigger;
  status: JobStatus;
  quick: boolean;
  config_path: string | null;
  decision_id: string | null;
  snapshot_id: string | null;
  model_version: string | null;
  incumbent_version: string | null;
  gate_passed: boolean | null;
  promoted: boolean;
  requested_by: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_ms: number | null;
  steps: JobStep[];
  error: string | null;
}

/** One gate of GET /governance/models: status, the reason when it failed and its scalar numbers. */
export interface GateSummary {
  status: GateStatus;
  reason: string | null;
  /** e.g. candidate, incumbent, diff, ci_lo, ci_hi, ci_level, margin, p_value, n_users (non-inferiority gates) */
  numbers: Record<string, number | string | boolean | null>;
}

export interface GovernedModel {
  version: string;
  state: ModelState;
  is_active: boolean;
  model_version_id: number | null;
  trained_at: string | null;
  dataset_version: string | null;
  snapshot_id: string | null;
  job_id: string | null;
  /** "job": from a training_jobs row; "artifact": read from the model's files (no job row) */
  source?: "job" | "artifact";
  decision_id: string | null;
  /** training-time test metrics of the hybrid (headline @10) */
  metrics: Record<string, number>;
  gate_passed: boolean | null;
  gate_reasons: string[];
  gated_against: string | null;
  gated_at: string | null;
  gates: Record<string, GateSummary>;
  /** promotable now without force */
  promotable: boolean;
  /** why not (empty when promotable) */
  blockers: string[];
  promoted_at: string | null;
  promoted_by: string | null;
  forced: boolean;
  force_reason: string | null;
}

export interface GovernedModelList {
  active: string | null;
  /** what POST /governance/models/rollback restores */
  previous: string | null;
  items: GovernedModel[];
}

export interface RegistryHistoryEntry {
  action: string;
  at: string;
  version?: string;
  previous?: string | null;
  reason?: string | null;
  [k: string]: unknown;
}

/** The full gate result (jev_ml.governance.gates.evaluate_candidate), stored per version. */
export interface GateResult {
  gate_version: string;
  candidate: string;
  incumbent: string | null;
  snapshot_dir?: string;
  config: Record<string, unknown>;
  evaluated_at: string;
  gates: Record<string, { status: GateStatus; reason: string | null; [k: string]: unknown }>;
  method?: string;
  split?: Record<string, unknown>;
  metrics?: { candidate: Record<string, number>; incumbent: Record<string, number>; n_users: number };
  passed: boolean;
  reasons: string[];
  seconds?: number;
}

/** GET /governance/models/{version}/lineage */
export interface ModelLineage {
  version: string;
  state: ModelState;
  snapshot: (SnapshotOut & { manifest: Record<string, unknown> | null }) | null;
  snapshot_id: string | null;
  config_hash: string | null;
  git_commit: string | null;
  seed: number | null;
  dataset_version: string | null;
  experiment_run: string | null;
  training_metrics: Record<string, number>;
  training_config: Record<string, unknown>;
  /** lineage.json: config path, jev_ml version, split, train seconds, incumbent at training… */
  lineage: Record<string, unknown> | null;
  job: TrainingJobOut | null;
  decision_id: string | null;
  gate: GateResult | null;
  promoted_at: string | null;
  promoted_by: string | null;
  forced: boolean;
  force_reason: string | null;
  history: RegistryHistoryEntry[];
}

export interface RetrainRequest {
  quick?: boolean | null;
  config?: string | null;
}

export interface EvaluateRequest {
  quick?: boolean | null;
}

export interface PromoteRequest {
  force?: boolean;
  reason?: string | null;
}

export interface RollbackRequest {
  reason?: string | null;
}

export interface PromotionResult {
  version: string;
  previous: string | null;
  action: "promote" | "rollback";
  forced: boolean;
  gate_passed: boolean | null;
}

// ---- online experiments: /experiments/online ------------------------------------------------

export type ExperimentStatus = "draft" | "running" | "paused" | "stopped" | "concluded";
export type ExperimentAction = "start" | "pause" | "stop" | "conclude" | "ramp" | "update" | "delete";
export type PrimaryMetric = "interaction_rate" | "positive_rate" | "rating_rate" | "feedback_rate" | "ndcg_at_10" | "diversity" | "novelty";
export type GuardrailMetric = "negative_rate" | "latency_p95_ms" | PrimaryMetric | "coverage";
export type RateMetric = "interaction_rate" | "positive_rate" | "rating_rate" | "feedback_rate" | "negative_rate";
export type MeanMetric = "ndcg_at_10" | "diversity" | "novelty";

export interface VariantConfig {
  hybrid_overrides?: Record<string, unknown>;
  model_version?: string | null;
  strategy_decision?: boolean;
  recency_half_life_days?: number | null;
}

export interface VariantIn {
  name: string;
  is_control: boolean;
  weight: number;
  description: string;
  config: VariantConfig;
}

export interface GuardrailIn {
  metric: GuardrailMetric;
  max_increase?: number | null;
  max_decrease?: number | null;
  max_ratio?: number | null;
  min_absolute_ms?: number;
}

export interface AnalysisIn {
  alpha: number;
  power: number;
  mde_relative: number;
  min_users_per_variant: number;
}

export interface ExperimentCreate {
  key: string;
  name: string;
  surface?: "recommendations";
  hypothesis?: string;
  primary_metric: PrimaryMetric;
  guardrails: GuardrailIn[];
  traffic_percent: number;
  attribution_window_hours?: number | null;
  analysis: AnalysisIn;
  variants: VariantIn[];
}

export type ExperimentUpdate = Partial<Omit<ExperimentCreate, "key" | "surface">>;

export interface RampIn {
  traffic_percent: number;
}

export interface VariantOut {
  name: string;
  is_control: boolean;
  weight: number;
  description: string;
  config: VariantConfig;
  assigned_users: number;
}

export type ConclusionDecision = "ship" | "keep_control" | "inconclusive";

export interface Conclusion {
  decision: ConclusionDecision;
  /** set only for ship (the treatment) and keep_control (the control); never for inconclusive */
  winner: string | null;
  reasons: string[];
}

export interface ExperimentOut {
  id: number;
  key: string;
  name: string;
  surface: string;
  status: ExperimentStatus;
  hypothesis: string;
  primary_metric: PrimaryMetric;
  guardrails: GuardrailIn[];
  traffic_percent: number;
  attribution_window_hours: number;
  analysis: Partial<AnalysisIn> & { data_source?: DataSource };
  data_source: DataSource;
  variants: VariantOut[];
  started_at: string | null;
  stopped_at: string | null;
  concluded_at: string | null;
  created_at: string;
  updated_at: string;
  created_by: number | null;
  allowed_actions: ExperimentAction[];
  conclusion: Conclusion | null;
}

export interface ExperimentList {
  items: ExperimentOut[];
  total: number;
}

export type DataSource = "live" | "offline_replay";

export interface RateCell {
  users_with_event: number;
  users: number;
  rate: number | null;
  per_exposure_rate: number | null;
}

export interface VariantResult {
  name: string;
  is_control: boolean;
  weight: number;
  config: VariantConfig;
  assigned_users: number;
  exposed_users: number;
  exposures: number;
  cache_hit_exposures: number;
  rates: Record<RateMetric, RateCell>;
  means: Record<MeanMetric, { mean: number | null; users: number }>;
  coverage: { distinct_items: number; catalogue: number; value: number | null };
  latency_ms: { p50: number | null; p95: number | null; p95_generated: number | null };
}

/** Treatment vs control for one metric (services/experiment_stats two_proportion / bootstrap_diff). */
export interface Comparison {
  control: { successes?: number; n: number; rate?: number | null; mean?: number | null };
  treatment: { successes?: number; n: number; rate?: number | null; mean?: number | null };
  diff: number | null;
  relative_lift: number | null;
  ci: [number | null, number | null];
  z?: number | null;
  p_value: number;
  significant: boolean;
  test: string;
  alpha: number;
}

export interface SrmTest {
  observed: number[];
  expected: number[] | null;
  chi2: number | null;
  p_value: number | null;
  alpha: number;
  detected: boolean;
  test: string;
}

export interface GuardrailResult {
  metric: GuardrailMetric;
  variant: string;
  control_value: number | null;
  treatment_value: number | null;
  threshold: Record<string, number>;
  breached: boolean;
  reason: string | null;
}

/** GET /experiments/online/{key}/results */
export interface ExperimentResults {
  experiment: string;
  status: ExperimentStatus;
  surface: string;
  data_source: DataSource;
  /** "live traffic", or the offline-replay label; always shown */
  label: string;
  primary_metric: PrimaryMetric;
  alpha: number;
  /** Bonferroni-adjusted alpha for the primary metric */
  alpha_primary: number;
  attribution_window_hours: number;
  unit_of_analysis: string;
  computed_at: string;
  totals: { assigned_users: number; exposures: number; outcomes: number };
  variants: VariantResult[];
  /** treatment name -> metric -> comparison */
  comparisons: Record<string, Record<RateMetric | MeanMetric, Comparison>>;
  srm: { assigned: SrmTest; exposed: SrmTest; detected: boolean };
  sample_size: {
    required_per_variant: number | null;
    smallest_variant_users: number;
    min_users_per_variant: number;
    mde_relative: number;
    power: number;
  };
  warnings: string[];
  guardrails: GuardrailResult[];
  conclusion: Conclusion;
}

// ---- decision lineage: GET /intel/decisions/{id}/lineage ------------------------------------

export type LineageNodeType = "decision" | "warning" | "signal" | "trend" | "anomaly" | "forecast" | "risk" | "action" | "series" | "source" | "run";

export interface LineageNode {
  /** "<type>:<ref>" */
  id: string;
  type: LineageNodeType | string;
  ref?: string;
  title?: string;
  [k: string]: unknown;
}

export interface LineageEdge {
  from: string;
  to: string;
  relation: "evidence" | "about" | "built_from" | "observed_in" | "derived_from_run" | "read_by" | string;
}

export interface LineageRunNode extends LineageNode {
  type: "run";
  run_id: string;
  domain: string;
  mode: "live" | "replay";
  trigger: string;
  as_of: string | null;
  started_at: string | null;
  pipeline_version: string;
  core_version: string | null;
  data_version: string | null;
  model_version: string | null;
  config_hash: string | null;
  input_fingerprint: string | null;
  event_watermark: EventWatermark | null;
}

export interface LineageEvidence {
  owner: string;
  position: number;
  kind: string | null;
  label: string | null;
  value: unknown;
  detail: string | null;
  ref: string | null;
  /** the node id the ref resolved to, null when it did not */
  resolves_to: string | null;
}

export interface IntelLineage {
  version: string;
  root: { type: string; id: string; node: string; db_id?: number; [k: string]: unknown };
  run: LineageRunNode;
  evidence: LineageEvidence[];
  nodes: LineageNode[];
  edges: LineageEdge[];
  series: string[];
  sources: string[];
  node_types: string[];
  unresolved: { from: string; ref: string }[];
  complete: boolean;
  event_watermark: EventWatermark | null;
}
