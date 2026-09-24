/**
 * Types for the intelligence layer (`/intel/*`). They mirror docs/intelligence.md sections 4, 6 and
 * 9 field for field; change them only when the contract changes.
 */

export type Severity = "low" | "medium" | "high" | "critical";
export type Direction = "up" | "down" | "flat";
/**
 * `interval` (v1.1): confidence is the nominal coverage of a score decision's answer_interval.
 * `evidence` (v1.2): 1 − an (adjusted) p-value, e.g. trends and preference drift. Not a probability.
 */
export type ConfidenceKind = "probability" | "margin" | "rule" | "interval" | "evidence";
export type SystemStatus = "nominal" | "watch" | "alert";

export interface Evidence {
  kind: "metric" | "series" | "record" | "test" | "model";
  label: string;
  value: number | string | null;
  detail: string | null;
  /** another object id or a series_id */
  ref: string | null;
}

// ---- PipelineResult pieces -------------------------------------------------------------------

export interface RunInfo {
  pipeline_version: string;
  as_of: string;
  now: string;
  data_version: string;
  model_version: string | null;
  stage_ms: Record<string, number>;
  config: Record<string, unknown>;
}

export interface DataSource {
  source: string;
  kind: string;
  rows: number;
  first_event: string | null;
  last_event: string | null;
  /** now - last_event */
  age_days: number | null;
  /** as_of - last_event */
  lag_days: number | null;
  expected_update: string | null;
  /** null = no freshness SLA (or not assessable) */
  fresh: boolean | null;
  detail?: string | null;
}

export interface QualityCheck {
  name: string;
  passed: boolean;
  value: number | null;
  threshold: number | null;
  severity: string;
  detail: string | null;
}

export interface DataBlock {
  sources: DataSource[];
  quality: { score: number; checks: QualityCheck[] };
}

export interface SeriesPoint {
  t: string;
  v: number;
  partial?: boolean;
}

export interface Series {
  id: string;
  metric: string;
  entity: string;
  entity_type: string;
  unit: string;
  points: SeriesPoint[];
}

export type SignalKind = "trend" | "anomaly" | "change_point" | "forecast" | "quality" | "live" | "model";

export interface Signal {
  id: string;
  dedup_key: string;
  kind: SignalKind;
  entity_type: "genre" | "platform" | "user" | "model" | "source";
  entity: string;
  title: string;
  value: number | null;
  unit: string | null;
  /** 0..1, defined per kind */
  strength: number;
  direction: Direction | null;
  source: string;
  observed_at: string;
  window: string | null;
  freshness_days: number | null;
  evidence: Evidence[];
  /** v1.2 (platform.md §2): the domain adapter that emitted it */
  domain?: string;
}

export interface ChangePoint {
  date: string;
  before_mean: number;
  after_mean: number;
  p_value: number;
}

export interface Trend {
  id: string;
  series_id: string;
  entity: string;
  metric: string;
  window: { start: string; end: string; months: number };
  direction: Direction;
  slope: number;
  /** Theil–Sen, 95 % */
  slope_ci: [number, number];
  change_rate_pct_per_month: number;
  kendall_tau: number;
  p_value: number;
  /** 1 - p. NOT a probability that the trend is real. */
  evidence_strength: number;
  recent_mean: number;
  prior_mean: number;
  ratio: number | null;
  change_point: ChangePoint | null;
  domain?: string;
}

export type AnomalyKind = "series_spike" | "series_drop" | "rater_behaviour" | "live_feedback";

export interface Anomaly {
  id: string;
  kind: AnomalyKind;
  entity_type: "genre" | "platform" | "user";
  entity: string;
  series_id: string | null;
  detected_at: string;
  value: number | null;
  baseline: number | null;
  deviation: number | null;
  /** robust z (MAD) for series; isolation-forest percentile for raters */
  score: number;
  method: "robust_z" | "isolation_forest" | "rate_test";
  severity: Severity;
  suppressed: boolean;
  suppression_reason: string | null;
  features: Record<string, number | string | null>;
  evidence: Evidence[];
  domain?: string;
}

export interface ForecastPoint {
  t: string;
  mean: number;
  lo80: number;
  hi80: number;
}

export interface Backtest {
  origins: number;
  mase: number | null;
  smape: number | null;
  mae: number | null;
  coverage80: number | null;
  naive_mase: number | null;
}

export interface Forecast {
  id: string;
  series_id: string;
  entity: string;
  metric: string;
  model: "holt_damped" | "moving_average" | "naive";
  model_version: string;
  horizon_months: number;
  issued_at: string;
  features_used: string[];
  points: ForecastPoint[];
  backtest: Backtest;
  domain?: string;
}

export interface CalibrationBin {
  bin: string;
  predicted: number;
  observed: number;
  n: number;
}

export interface LapseMetrics {
  auc: number | null;
  brier: number | null;
  ece: number | null;
  base_rate: number | null;
  baseline_auc: number | null;
  n_train: number;
  n_test: number;
  train_cutoffs: string[];
  test_cutoffs: string[];
}

export interface Lapse {
  model_version: string;
  horizon_days: number;
  features: string[];
  metrics: LapseMetrics;
  calibration: CalibrationBin[];
  population: { n_scored: number; expected_lapses: number; high_risk: number; threshold: number };
  top: { user_id: number; p: number; features: Record<string, number | string | null> }[];
  status: "ok" | "insufficient_data";
  detail: string | null;
}

export type RiskKind =
  | "genre_demand_decline"
  | "audience_lapse"
  | "rating_manipulation"
  | "data_quality"
  | "data_staleness"
  | "model_staleness"
  | "model_quality"
  | "recommendation_rejection";

export interface RiskFactor {
  name: string;
  value: number;
  weight: number;
  contribution: number;
  detail: string | null;
}

export interface Risk {
  id: string;
  kind: RiskKind;
  title: string;
  entity_type: string;
  entity: string;
  likelihood: number;
  impact: number;
  exposure: number;
  confidence: number;
  /** not in the contract's Risk; honoured if the backend adds it */
  confidence_kind?: ConfidenceKind;
  data_quality: number;
  /** 0..100 = 100 * likelihood * impact, shrunk by (confidence * data_quality) */
  score: number;
  level: Severity;
  factors: RiskFactor[];
  evidence: Evidence[];
  recommended_response: string;
  domain?: string;
}

export interface Decision {
  id: string;
  key: string;
  spec_id: string;
  policy_version: string;
  question: string;
  kind: DecisionKind;
  /** [] for a score decision */
  options: string[];
  /** a number for a score decision, an option otherwise; null when abstained */
  answer: string | number | null;
  /** {} for a score decision */
  option_scores: Record<string, number>;
  confidence: number | null;
  confidence_kind: ConfidenceKind;
  /** exact inputs the policy used */
  state: Record<string, unknown>;
  rationale: string[];
  evidence: Evidence[];
  abstained: boolean;
  fallback_reason: string | null;
  entity_type: string;
  entity: string;
  /** v1.1: the multi-question call this decision was answered in; null when asked alone */
  batch_id?: string | null;
  /** v1.1, score decisions only */
  scale?: DecisionScale | null;
  /** v1.1, score decisions only: [lo, hi] at `confidence` nominal coverage */
  answer_interval?: [number, number] | null;
  /** v1.2: the domain adapter the decision was made for */
  domain?: string;
}

export type DecisionKind = "boolean" | "choice" | "score";

export interface DecisionScale {
  min: number;
  max: number;
  unit: string;
}

/** A multi-question call: several bounded questions answered against one hashed state (9.1). */
export interface DecisionBatch {
  id: string;
  name: string;
  question: string;
  keys: string[];
  decision_ids: string[];
  /** sha1 of the shared input state */
  state_hash: string;
  policy_versions: Record<string, string>;
  /** backend additions: the shared state's top-level keys and how the call went */
  state_keys?: string[];
  n_decisions?: number;
  n_abstained?: number;
  status?: "ok" | "failed";
  failure_reason?: string | null;
}

/** GET /intel/decisions/batches?run_id= */
export interface DecisionBatchList {
  items: DecisionBatch[];
}

/** Decision as stored in the log (GET /intel/decisions). */
export interface DecisionRecord extends Decision {
  db_id: number;
  run_id: string;
  as_of: string;
  created_at: string;
  feedback: { correct: number; incorrect: number };
}

export interface ObjectSource {
  type: string;
  id: string;
}

export interface Action {
  id: string;
  title: string;
  priority: "P1" | "P2" | "P3";
  priority_score: number;
  reason: string;
  expected_impact: string;
  effort: "low" | "medium" | "high";
  risk: string;
  evidence: Evidence[];
  next_step: string;
  source: ObjectSource;
  domain?: string;
}

export interface SummaryCounts {
  signals: number;
  trends_up: number;
  trends_down: number;
  anomalies: number;
  risks_high: number;
  warnings: number;
  decisions: number;
  actions: number;
}

export interface Summary {
  status: SystemStatus;
  headline: string;
  counts: SummaryCounts;
}

// ---- API response shapes (section 6) ---------------------------------------------------------

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
  run_id: string | null;
  as_of: string | null;
}

export interface Run {
  id: number;
  run_id: string;
  trigger: "startup" | "manual" | "script" | "schedule";
  status: "running" | "succeeded" | "failed";
  as_of: string;
  started_at: string;
  finished_at: string | null;
  duration_ms: number | null;
  pipeline_version: string;
  data_version: string;
  model_version: string | null;
  summary: Summary | null;
  stage_ms: Record<string, number>;
  error: string | null;
  /** v1.2 */
  domain?: string;
}

/** GET /intel/runs. The contract names the endpoint but not its envelope; a list page is assumed. */
export interface RunList {
  items: Run[];
  total: number;
}

export interface IntelHealth {
  database: "ok" | "error";
  cache: "ok" | "error";
  /** v1.2: `not_applicable` for a domain without a recommender */
  model: "ok" | "unavailable" | "not_applicable";
  pipeline: "ok" | "failed" | "never_run";
}

export interface IntelStatus {
  latest_run: Run | null;
  summary: Summary | null;
  data: DataBlock | null;
  model: { version: string; trained_at: string; age_days: number; dataset_version: string } | null;
  warnings_open: { total: number; by_severity: Record<Severity, number> };
  recent_decisions: (Decision & Partial<Pick<DecisionRecord, "db_id" | "run_id" | "as_of" | "created_at">>)[];
  top_signals: Signal[];
  top_risks: Risk[];
  /** over the latest run's decisions */
  confidence_histogram: { bin: string; n: number }[];
  health: IntelHealth;
  /** v1.2 */
  domain?: string;
  domain_info?: { name: string; available: boolean; reason: string | null; capabilities: DomainCapabilities } | null;
}

export type WarningStatus = "new" | "acknowledged" | "investigating" | "resolved" | "dismissed";

export interface WarningTrigger {
  rule?: string;
  condition?: string;
  observed?: number | null;
  threshold?: number | null;
}

export interface WarningEvent {
  from_status: WarningStatus | null;
  to_status: WarningStatus;
  note: string | null;
  actor: string;
  at: string;
}

export interface IntelWarning {
  id: number;
  key: string;
  title: string;
  description: string;
  severity: Severity;
  confidence: number;
  confidence_kind?: ConfidenceKind;
  status: WarningStatus;
  trigger: WarningTrigger;
  evidence: Evidence[];
  recommended_action: string;
  source: Partial<ObjectSource>;
  detected_at: string;
  last_seen_at: string;
  updated_at: string;
  occurrences: number;
  first_seen_run_id: string;
  last_seen_run_id: string;
  reopened_from: number | null;
  /** detail only */
  history?: WarningEvent[];
  /**
   * v1.2 (platform.md §4): the early_warning_level decision this warning is downstream of. Present
   * only on warnings raised after the migration; older rows have none.
   */
  decision_id?: string | null;
  /** v1.2: the level that decision answered */
  early_warning_level?: EarlyWarningLevel | null;
  domain?: string;
}

export type FeedbackTarget = "decision" | "warning" | "action" | "prediction";
export type Verdict = "correct" | "incorrect" | "useful" | "not_useful" | "false_positive";

export interface FeedbackIn {
  target_type: FeedbackTarget;
  target_id: string;
  verdict: Verdict;
  note: string | null;
  outcome: string | null;
}

export interface FeedbackRecord extends FeedbackIn {
  id: number;
  actor: string;
  created_at: string;
}

export interface FeedbackSummary {
  decision: { correct: number; incorrect: number; accuracy: number | null };
  warning: { useful: number; not_useful: number; false_positive: number; precision: number | null };
  action: { useful: number; not_useful: number };
  prediction: { correct: number; incorrect: number };
}

export interface FeedbackList {
  items: FeedbackRecord[];
  total: number;
  summary: FeedbackSummary;
}

export interface PredictionsResponse {
  run_id: string | null;
  as_of: string | null;
  forecasts: Forecast[];
  lapse: Lapse | null;
}

export interface SeriesDetail {
  run_id: string | null;
  as_of: string | null;
  series: Series;
  trend: Trend | null;
  anomalies: Anomaly[];
  forecast: Forecast | null;
}

// ---- scenarios -------------------------------------------------------------------------------

export type ScenarioKind = "continue" | "slow" | "reverse" | "shock";

export interface ScenarioSpec {
  name: string;
  kind: ScenarioKind;
  trend_multiplier?: number;
  level_shift_pct?: number;
  shock_month?: number;
}

export interface ScenarioInput {
  series_id: string;
  horizon_months: number;
  as_of: string | null;
  scenarios: ScenarioSpec[];
}

export interface ScenarioRequest extends ScenarioInput {
  save: boolean;
  title?: string;
}

export interface ScenarioResult {
  name: string;
  kind: ScenarioKind;
  assumptions: string[];
  points: ForecastPoint[];
  total: number;
  delta_vs_baseline: number;
  delta_pct: number;
  end_level: number;
}

export interface ScenarioOutput {
  series_id: string;
  as_of: string;
  model: string;
  baseline: { points: ForecastPoint[]; total: number };
  history: { t: string; v: number }[];
  scenarios: ScenarioResult[];
  comparison: { name: string; total: number; delta_pct: number; end_level: number; rank: number }[];
  uncertainty_note: string;
}

export interface ScenarioResponse extends ScenarioOutput {
  id: number | null;
}

export interface SavedScenario {
  id: number;
  title: string;
  series_id: string;
  created_at: string;
  input: ScenarioInput;
  output: ScenarioOutput;
}

// ---- evaluation ------------------------------------------------------------------------------

export interface EvaluationReport {
  created_at: string;
  pipeline_version: string;
  data_version: string;
  as_of: string;
  forecast: {
    per_series: {
      series_id: string;
      model: string;
      mase: number | null;
      smape: number | null;
      mae: number | null;
      coverage80: number | null;
      naive_mase: number | null;
      origins: number;
      selection_origins?: number;
    }[];
    summary: {
      n_series: number;
      median_mase: number | null;
      share_beating_naive: number | null;
      mean_coverage80: number | null;
      median_naive_mase?: number | null;
    };
  };
  lapse: {
    metrics: LapseMetrics;
    calibration: CalibrationBin[];
    baselines: { name: string; auc: number | null; brier: number | null }[];
  };
  anomaly: {
    injection: {
      protocol: string;
      n_genuine: number;
      n_injected: number;
      attack_types: {
        type: string;
        n: number;
        precision: number | null;
        recall: number | null;
        f1: number | null;
        auc: number | null;
        baseline_precision: number | null;
        baseline_recall: number | null;
      }[];
    };
    series: { protocol: string; detection_rate: number | null; false_alarm_rate: number | null; n_trials: number };
  };
  change_point: {
    protocol: string;
    detection_rate: number | null;
    false_alarm_rate: number | null;
    mean_abs_location_error_months: number | null;
    n_trials: number;
  };
  latency: { pipeline_ms_mean: number | null; pipeline_ms_p95: number | null; n_runs: number; stage_ms: Record<string, number> };
  notes: string[];
}

export interface EvaluationResponse {
  available: boolean;
  run_dir: string | null;
  report: EvaluationReport | null;
  /** v1.2 */
  domain?: string;
  /** v1.2: why `report` is null for this domain */
  reason?: string | null;
  /** v1.2: this domain's section of the newest platform-eval report */
  platform?: PlatformEvaluation | null;
}

// ---- v1.2 platform evaluation (experiments/platform-eval-*/report.json) ----------------------

export interface WarningOutcomeCounts {
  units: number;
  observable: number;
  warned: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number | null;
  false_positive_rate: number | null;
  recall: number | null;
  base_rate: number | null;
}

export interface DecisionConsistency {
  replays?: number;
  situation_pairs: number;
  flips: number;
  flip_rate?: number | null;
  escalations?: number;
  de_escalations?: number;
  warning_boundary_flips: number;
  warning_boundary_flip_rate?: number | null;
}

export interface PlatformForecastSeries {
  series_id: string;
  model: string;
  mae: number | null;
  rmse: number | null;
  mase: number | null;
  naive_mae: number | null;
  naive_mase: number | null;
  coverage80: number | null;
  origins: number;
  selection_origins?: number;
}

export interface PlatformDomainEvaluation {
  domain: string;
  as_of_default?: string | null;
  forecast: {
    protocol: string;
    per_series: PlatformForecastSeries[];
    summary: {
      n_series: number;
      median_mae: number | null;
      median_rmse: number | null;
      median_mase: number | null;
      median_naive_mase: number | null;
      share_beating_naive: number | null;
      mean_coverage80: number | null;
    };
  };
  warnings: {
    per_kind: Record<string, WarningOutcomeCounts>;
    overall: WarningOutcomeCounts;
    excluded_kinds?: Record<string, string>;
    examples?: { as_of: string; kind: string; unit: string; confirmed: boolean; detail: string }[];
  };
  /** movie: one consistency block over all replays */
  consistency?: DecisionConsistency;
  /** generic: replays in separate windows (e.g. around two recessions) */
  windows?: { window: [string, string]; warnings: WarningOutcomeCounts; consistency: DecisionConsistency; warnings_per_replay?: number[] }[];
  consistency_pooled_within_windows?: DecisionConsistency;
  replays: { as_of?: string[]; n: number; ms_mean: number | null };
  warnings_per_replay?: number[];
}

export interface PlatformEvaluation {
  run_dir: string;
  created_at: string | null;
  /** periods ahead a warning is checked against */
  horizon: number | null;
  report: PlatformDomainEvaluation;
}

// ---- v1.2 drift evaluation (experiments/drift-eval-*/report.json) ---------------------------

export interface DetectorMetrics {
  precision: number | null;
  recall: number | null;
  recall_ci95?: [number, number] | null;
  f1: number | null;
  fpr_upper_bound: number | null;
  fpr_ci95?: [number, number] | null;
  tp: number;
  fp: number;
  n_pos: number;
  n_neg: number;
}

export interface DetectorSplice {
  overall_drift_detected: DetectorMetrics;
  preference_drift: DetectorMetrics;
  per_aspect: Record<string, DetectorMetrics>;
}

export interface PairedDelta {
  delta: number | null;
  ci95: [number, number] | null;
  p_better: number | null;
  n_users: number;
  n_improved: number;
  n_worse: number;
}

export interface AdaptationGroup {
  mean_ndcg10: Record<string, number | null>;
  mean_recall10: Record<string, number | null>;
  vs_standard: Record<string, { ndcg10: PairedDelta; recall10: PairedDelta }>;
}

export interface DriftEvalReport {
  run: string;
  model_version: string | null;
  dataset_version: string | null;
  drift_config?: Record<string, unknown>;
  strategy_config?: Record<string, unknown>;
  detector: {
    n_users: number;
    modes: Record<string, { negatives_tested: number; by_splice_size: Record<string, DetectorSplice> }>;
    seconds?: number;
    notes?: string[];
  };
  adaptation: {
    protocol: string;
    n_users: number;
    group_sizes: Record<string, number>;
    settings: Record<string, Record<string, AdaptationGroup>>;
    seconds?: number;
  };
  latency?: Record<string, Record<string, number | string>>;
  notes?: string[];
  seconds?: number;
}

/** GET /intel/evaluation/drift (movie) */
export interface DriftEvalResponse {
  available: boolean;
  run_dir: string | null;
  report: DriftEvalReport | null;
  domain?: string;
}

/** GET /admin/metrics: in-process counters. Only the groups are named by the contract. */
export type AdminMetrics = Record<string, unknown>;

// ---- v1.1 (section 9.3) ----------------------------------------------------------------------

export type EvidenceOwnerType = "signal" | "trend" | "anomaly" | "forecast" | "risk" | "decision" | "warning" | "action";

/** GET /intel/evidence row: an Evidence item with the object it belongs to. */
export interface EvidenceRow extends Evidence {
  id: number;
  owner_type: EvidenceOwnerType;
  owner_id: string;
  owner_title: string;
  run_id: string;
}

export type HistoryEntity = "signals" | "risks" | "trends" | "anomalies";

/** One run's reading of an object, oldest → newest across runs. */
export interface HistoryPoint {
  run_id: string;
  as_of: string;
  created_at: string;
  value: number | null;
  score: number | null;
  level: string | null;
  direction: string | null;
}

/** GET /intel/history/{entity}?key= */
export interface HistoryResponse {
  entity: HistoryEntity;
  key: string;
  items: HistoryPoint[];
}

/**
 * Recommendation calibration (the metrics part of models/<version>/calibration.json). The contract
 * only promises "metrics"; the ML side nests them per split (validation, test.<protocol>), so every
 * field is optional and the console reads it through calibrationView().
 */
export interface CalibrationMetrics {
  n?: number | null;
  ece?: number | null;
  brier?: number | null;
  base_rate_brier?: number | null;
  auc?: number | null;
  observed_rate?: number | null;
  mean_predicted?: number | null;
  reliability?: CalibrationBin[] | null;
  bins?: CalibrationBin[] | null;
  primary?: boolean;
  [k: string]: unknown;
}

export interface RecCalibration extends CalibrationMetrics {
  method?: string | null;
  target?: string | null;
  fitted_on?: string | Record<string, unknown> | null;
  created_at?: string | null;
  validation?: CalibrationMetrics | null;
  test?: Record<string, CalibrationMetrics | string> | CalibrationMetrics | null;
  /** the held-out figures the ML side leads with (split + stratum named inside) */
  headline?: (CalibrationMetrics & { split?: string; stratum?: string }) | null;
  /** one calibrator per profile-size stratum */
  strata?: CalibrationStratum[] | null;
  assumptions?: string[] | null;
}

export interface CalibrationStratum {
  name: string;
  applies_to?: { min: number; max: number | null; unit: string } | null;
  feature?: string | null;
  base_rate?: number | null;
  validation?: CalibrationMetrics | null;
  test?: CalibrationMetrics | null;
}

export interface RecFeedbackTotals {
  like: number;
  dislike: number;
  not_interested: number;
  clicked: number;
}

export interface ReasonCodeStats extends RecFeedbackTotals {
  code: string;
  served: number;
  positive_rate: number | null;
}

export interface ServedRecommendation {
  id: number;
  user_id: number;
  movie_id: number;
  title: string;
  rank: number;
  score: number;
  confidence: number | null;
  confidence_kind: "probability" | null;
  reason: string;
  reason_code: string;
  created_at: string;
  /** v1.2 */
  decision_id?: string | null;
  strategy?: string | null;
}

/** GET /intel/recommendations: recommender monitoring. */
export interface RecommenderMonitoring {
  /** v1.2: served rows per strategy ("unrecorded" for rows from before strategies) */
  strategies?: Record<string, number>;
  domain?: string;
  model_version: string | null;
  calibration: RecCalibration | null;
  served: { total: number; per_day: { date: string; count: number }[] };
  feedback_totals: RecFeedbackTotals;
  reason_codes: ReasonCodeStats[];
  confidence_histogram: { bin: string; n: number }[];
  recent: ServedRecommendation[];
}

/** GET /admin/audit row = the audit_logs columns. */
export interface AuditEntry {
  id: number;
  at: string;
  actor_user_id: number | null;
  actor: string | null;
  action: string;
  target_type: string | null;
  target_id: string | null;
  detail: Record<string, unknown> | null;
  request_id: string | null;
}

/** GET /admin/audit. Page<AuditEntry>; the run fields do not apply. */
export type AuditPage = Pick<Page<AuditEntry>, "items" | "total"> & Partial<Pick<Page<AuditEntry>, "limit" | "offset">>;

export interface EvaluationRunSummary {
  id: number;
  run_dir: string;
  created_at: string;
  pipeline_version: string;
  data_version: string;
  headline: {
    median_mase: number | null;
    share_beating_naive: number | null;
    lapse_auc: number | null;
    lapse_ece: number | null;
    shilling_auc_mean: number | null;
    pipeline_ms_mean: number | null;
  };
}

/** GET /intel/evaluation/runs */
export interface EvaluationRunList {
  items: EvaluationRunSummary[];
  total: number;
}

// ---- v1.2: platform domains and user intelligence (platform.md §8–§10) ------------------------

/** What a domain adapter produces. Movie-only stages are false (or absent) for other domains. */
export interface DomainCapabilities {
  recommendation?: boolean;
  user_intelligence?: boolean;
  lapse?: boolean;
  raters?: boolean;
  model_governance?: boolean;
  scenarios?: boolean;
}

export type DomainCapability = keyof DomainCapabilities;

export interface DomainSource extends DataSource {
  license?: string | null;
  url?: string | null;
  checksum?: string | null;
}

/** One adapter in GET /intel/domains. */
export interface DomainInfo {
  key: string;
  name: string;
  description: string;
  entity_types: string[];
  frequency: string;
  sources: DomainSource[];
  capabilities: DomainCapabilities;
  available: boolean;
  /** why the adapter cannot load on this machine; null when available */
  reason: string | null;
  latest_run: Run | null;
  warnings_open: number;
}

/** GET /intel/domains */
export interface DomainList {
  items: DomainInfo[];
}

/** The early_warning_level decision's answers, least to most severe (platform.md §4). */
export type EarlyWarningLevel = "NO_ACTION" | "MONITOR" | "WARNING" | "URGENT_ACTION";

/** recommendation_strategy decision answers (platform.md §4). */
export type RecommendationStrategy = "standard" | "adapt_to_recent" | "explore";

/** A recommendation downstream of a decision (platform.md §2, §10). */
export interface EngineRecommendation {
  item_id: number | string;
  title?: string | null;
  rank: number;
  score: number;
  reason: string;
  confidence: number | null;
  confidence_kind?: ConfidenceKind | null;
  /** the decision this recommendation is downstream of */
  decision_id: string | null;
  strategy?: string | null;
  evidence: Evidence[];
}

export interface TimeWindow {
  start: string | null;
  end: string | null;
  n: number;
}

export interface PreferenceWindow extends TimeWindow {
  /** "historical", "recent" or a period label */
  label: string;
  /** category → share of the window's events (0..1) */
  shares: Record<string, number>;
}

export interface PreferenceHistory {
  categories: string[];
  windows: PreferenceWindow[];
}

export type DriftAspectName =
  | "genre_distribution"
  | "rating_level"
  | "activity_rate"
  | "release_year"
  | "content_similarity"
  | "acceptance";

export interface DriftAspect {
  aspect: DriftAspectName | string;
  status: "ok" | "insufficient_data";
  test: string;
  statistic: number | null;
  p_value: number | null;
  /** Holm-adjusted across aspects */
  p_adjusted: number | null;
  significant: boolean;
  effect: Record<string, unknown>;
  detail: string | null;
}

export interface DriftReport {
  status: "ok" | "insufficient_data";
  drift_detected: boolean;
  /** 1 − the smallest adjusted p; `confidence_kind` is "evidence", never a probability */
  confidence: number | null;
  confidence_kind: "evidence";
  historical_window: TimeWindow;
  recent_window: TimeWindow;
  aspects: DriftAspect[];
  summary: string;
  evidence: Evidence[];
}

/** GET /me/intelligence */
export interface MeIntelligence {
  user_id: number;
  as_of: string;
  profile: { n_events: number; first_event: string | null; last_event: string | null };
  preference_history: PreferenceHistory;
  drift: DriftReport;
  /** spec_id "recommendation_strategy"; options standard | adapt_to_recent | explore */
  strategy: Decision;
  signals: Signal[];
  recommendations: EngineRecommendation[];
}

export type PreferenceScenarioKind = "continue" | "accelerate" | "reverse";

export interface PreferenceScenarioSpec {
  name: string;
  kind: PreferenceScenarioKind;
  factor: number;
}

/** POST /me/intelligence/scenarios body */
export interface PreferenceScenarioRequest {
  k: number;
  scenarios: PreferenceScenarioSpec[];
}

export interface ProjectedShare {
  mean: number;
  lo80: number;
  hi80: number;
}

export interface PreferenceScenarioResult {
  name: string;
  kind: PreferenceScenarioKind | string;
  assumptions: string[];
  projected_shares: Record<string, ProjectedShare>;
  recommendations: EngineRecommendation[];
  /** share of the baseline list that is also in this scenario's list (0..1) */
  overlap_with_baseline: number;
}

/** POST /me/intelligence/scenarios response */
export interface PreferenceScenarioResponse {
  as_of: string;
  assumptions: string[];
  uncertainty_note: string;
  baseline: { shares: Record<string, number>; recommendations: EngineRecommendation[] };
  scenarios: PreferenceScenarioResult[];
  evidence: Evidence[];
}

export type MeFeedbackTarget = "strategy" | "recommendation";
export type MeVerdict = "accepted" | "rejected";

/** POST /me/intelligence/feedback body */
export interface MeFeedbackIn {
  target_type: MeFeedbackTarget;
  target_id: string;
  verdict: MeVerdict;
  note: string | null;
}

export interface MeFeedbackRecord {
  id: number;
  target_type: MeFeedbackTarget;
  target_id: string;
  verdict: MeVerdict;
  note?: string | null;
  /** the strategy decision the verdict was given under */
  decision_id?: string | null;
  created_at: string;
  updated_at?: string | null;
}

/** GET /me/intelligence/feedback: the caller's verdicts, newest first */
export interface MeFeedbackList {
  items: MeFeedbackRecord[];
  total: number;
}
