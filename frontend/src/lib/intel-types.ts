/**
 * Types for the intelligence layer (`/intel/*`). They mirror docs/intelligence.md sections 4, 6 and
 * 9 field for field; change them only when the contract changes.
 */

export type Severity = "low" | "medium" | "high" | "critical";
export type Direction = "up" | "down" | "flat";
/** `interval` (v1.1): confidence is the nominal coverage of a score decision's answer_interval. */
export type ConfidenceKind = "probability" | "margin" | "rule" | "interval";
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
}

/** GET /intel/runs. The contract names the endpoint but not its envelope; a list page is assumed. */
export interface RunList {
  items: Run[];
  total: number;
}

export interface IntelHealth {
  database: "ok" | "error";
  cache: "ok" | "error";
  model: "ok" | "unavailable";
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
}

/** GET /intel/recommendations: recommender monitoring. */
export interface RecommenderMonitoring {
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
