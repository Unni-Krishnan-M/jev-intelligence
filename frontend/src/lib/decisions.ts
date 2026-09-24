/**
 * Helpers for the two v1.2 decision types (docs/platform.md §4): the early-warning level, which
 * warnings are downstream of, and the per-user recommendation strategy. Pure functions only.
 */

import { fmtValue, humanize } from "./intel";
import type { Decision, EarlyWarningLevel, RecommendationStrategy } from "./intel-types";

export const EARLY_WARNING_SPEC = "early_warning_level";
export const STRATEGY_SPEC = "recommendation_strategy";

/** Least to most severe. A warning is raised only for the last two. */
export const EARLY_WARNING_LEVELS: EarlyWarningLevel[] = ["NO_ACTION", "MONITOR", "WARNING", "URGENT_ACTION"];

export const LEVEL_LABEL: Record<EarlyWarningLevel, string> = {
  NO_ACTION: "No action",
  MONITOR: "Monitor",
  WARNING: "Warning",
  URGENT_ACTION: "Urgent action",
};

export const LEVEL_MEANING: Record<EarlyWarningLevel, string> = {
  NO_ACTION: "Evidence does not call for attention.",
  MONITOR: "Something is moving; watch it, no warning raised.",
  WARNING: "A warning is raised for a person to triage.",
  URGENT_ACTION: "A warning is raised and needs a response now.",
};

export function isEarlyWarningLevel(v: unknown): v is EarlyWarningLevel {
  return typeof v === "string" && (EARLY_WARNING_LEVELS as string[]).includes(v);
}

/** The early-warning decision by spec, key or (for older logs) its exact option set. */
export function isEarlyWarningDecision(d: Pick<Decision, "spec_id" | "key" | "options">): boolean {
  if (d.spec_id === EARLY_WARNING_SPEC || d.key === EARLY_WARNING_SPEC || d.key.startsWith(`${EARLY_WARNING_SPEC}:`)) return true;
  return d.options.length === EARLY_WARNING_LEVELS.length && EARLY_WARNING_LEVELS.every((l) => d.options.includes(l));
}

/** 0..3 for a level, -1 for anything else (an abstention, a foreign answer). */
export function levelIndex(answer: unknown): number {
  return isEarlyWarningLevel(answer) ? EARLY_WARNING_LEVELS.indexOf(answer) : -1;
}

export interface LevelStep {
  level: EarlyWarningLevel;
  label: string;
  /** 1..4 */
  step: number;
  chosen: boolean;
  /** at or below the chosen level (the filled part of the scale) */
  reached: boolean;
  /** the policy's point score for this level, when it reported one */
  score: number | null;
  raisesWarning: boolean;
}

/** The four-step scale with the chosen level marked; nothing is chosen when the decision abstained. */
export function levelScale(d: Pick<Decision, "answer" | "option_scores" | "abstained">): LevelStep[] {
  const at = d.abstained ? -1 : levelIndex(d.answer);
  return EARLY_WARNING_LEVELS.map((level, i) => {
    const s = d.option_scores?.[level];
    return {
      level,
      label: LEVEL_LABEL[level],
      step: i + 1,
      chosen: i === at,
      reached: at >= 0 && i <= at,
      score: typeof s === "number" && Number.isFinite(s) ? s : null,
      raisesWarning: i >= 2,
    };
  });
}

// ---- the early-warning state snapshot --------------------------------------------------------

export const EVIDENCE_STAGES = [
  { key: "signal", label: "Signal", what: "strength" },
  { key: "trend", label: "Trend", what: "direction and q-value" },
  { key: "anomaly", label: "Anomaly", what: "score and severity" },
  { key: "forecast", label: "Forecast", what: "direction against the adverse side" },
  { key: "risk", label: "Risk", what: "score and level" },
] as const;

export type EvidenceStage = (typeof EVIDENCE_STAGES)[number]["key"];

export interface StateRow {
  key: string;
  label: string;
  value: string;
}

export interface StageGroup {
  stage: EvidenceStage;
  label: string;
  what: string;
  rows: StateRow[];
  /** false when the state carries nothing for this stage (skipped or not observed) */
  observed: boolean;
}

function stageOf(key: string): EvidenceStage | null {
  const k = key.toLowerCase();
  for (const s of EVIDENCE_STAGES) {
    if (k === s.key || k === `${s.key}s` || k.startsWith(`${s.key}_`) || k.startsWith(`${s.key}s_`)) return s.key;
  }
  return null;
}

function rowsFor(key: string, value: unknown, stage: string | null): StateRow[] {
  const strip = (k: string) => (stage ? k.replace(new RegExp(`^${stage}s?_`, "i"), "") : k);
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return Object.entries(value as Record<string, unknown>).map(([k2, v2]) => ({ key: `${key}.${k2}`, label: humanize(k2), value: fmtValue(v2, k2) }));
  }
  const label = key.toLowerCase() === stage || key.toLowerCase() === `${stage}s` ? "value" : humanize(strip(key));
  return [{ key, label, value: fmtValue(value, key) }];
}

/**
 * The policy's input state grouped by pipeline stage (signal → risk), in pipeline order, so the
 * snapshot reads like the evidence chain. Nested ({trend: {direction, q}}) and flat (trend_q)
 * state shapes both work; keys that belong to no stage are returned as `other`.
 */
export function ewlStateGroups(state: Record<string, unknown> | null | undefined): { groups: StageGroup[]; other: StateRow[] } {
  const byStage = new Map<EvidenceStage, StateRow[]>();
  const other: StateRow[] = [];
  for (const [k, v] of Object.entries(state ?? {})) {
    const stage = stageOf(k);
    if (stage) byStage.set(stage, [...(byStage.get(stage) ?? []), ...rowsFor(k, v, stage)]);
    else other.push(...rowsFor(k, v, null));
  }
  const groups = EVIDENCE_STAGES.map((s) => {
    const rows = byStage.get(s.key) ?? [];
    const observed = rows.some((r) => r.value !== "—");
    return { stage: s.key, label: s.label, what: s.what, rows, observed };
  });
  return { groups, other };
}

// ---- recommendation strategy -----------------------------------------------------------------

export const STRATEGY_LABEL: Record<RecommendationStrategy, string> = {
  standard: "Standard",
  adapt_to_recent: "Adapt to recent taste",
  explore: "Explore",
};

export const STRATEGY_MEANING: Record<RecommendationStrategy, string> = {
  standard: "Your whole history counts as usual.",
  adapt_to_recent: "Recent activity counts for more (a shorter recency half-life).",
  explore: "More variety across the list (a higher diversity weight).",
};

export function strategyLabel(v: unknown): string {
  if (v === null || v === undefined) return "—";
  const s = String(v);
  return s in STRATEGY_LABEL ? STRATEGY_LABEL[s as RecommendationStrategy] : humanize(s);
}
