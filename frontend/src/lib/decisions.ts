/**
 * Helpers for the two v1.2 decision types (docs/platform.md §4): the early-warning level, which
 * warnings are downstream of, and the per-user recommendation strategy. Pure functions only.
 */

import { fmtP, fmtValue, humanize } from "./intel";
import type { Decision, EarlyWarningLevel, RecommendationStrategy } from "./intel-types";
import { EARLY_WARNING_LEVELS, isEarlyWarningLevel, LEVEL_LABEL } from "./levels";

export { EARLY_WARNING_LEVELS, EARLY_WARNING_SPEC, isEarlyWarningDecision, isEarlyWarningLevel, LEVEL_LABEL } from "./levels";

export const STRATEGY_SPEC = "recommendation_strategy";

export const LEVEL_MEANING: Record<EarlyWarningLevel, string> = {
  NO_ACTION: "Evidence does not call for attention.",
  MONITOR: "Something is moving; watch it, no warning raised.",
  WARNING: "A warning is raised for a person to triage.",
  URGENT_ACTION: "A warning is raised and needs a response now.",
};

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
  /** the strongest component's title or detail, when the stage contributed points */
  summary: string | null;
  /** true when a component (or a stage block in the state) carries evidence for this stage */
  observed: boolean;
  /** the reason from `skipped` when the pipeline skipped this stage; null otherwise */
  skippedReason: string | null;
}

function stageOf(key: string): EvidenceStage | null {
  const k = key.toLowerCase();
  for (const s of EVIDENCE_STAGES) {
    if (k === s.key || k === `${s.key}s` || k.startsWith(`${s.key}_`) || k.startsWith(`${s.key}s_`)) return s.key;
  }
  return null;
}

/** A stage name from a component or a skipped entry ("change_point" belongs to trend). */
function stageFromName(name: string | null | undefined): EvidenceStage | null {
  if (!name) return null;
  if (name === "change_point" || name === "change_points") return "trend";
  return stageOf(name);
}

// ---- value formatting (shared by the snapshot and flattenState) -------------------------------

/** null, undefined, string, number or boolean. */
export function isScalar(x: unknown): boolean {
  return x === null || x === undefined || (typeof x !== "object" && typeof x !== "function");
}

export function isScalarList(x: unknown): x is unknown[] {
  return Array.isArray(x) && x.every(isScalar);
}

/** A plain object whose values are all scalars or lists of scalars (printable on one line). */
export function isScalarMap(x: unknown): x is Record<string, unknown> {
  return Boolean(x) && typeof x === "object" && !Array.isArray(x) && Object.values(x as object).every((v) => isScalar(v) || isScalarList(v));
}

const P_KEY = /(^|_)(p|q|p_value|q_value|p_adjusted)$/i;
/** keys that name a list item ("stage: reason") */
const ITEM_LABEL_KEYS = ["stage", "name", "aspect", "kind", "key", "label"];

function itemLabelKey(o: Record<string, unknown>): string | null {
  return ITEM_LABEL_KEYS.find((k) => typeof o[k] === "string") ?? null;
}

/**
 * Any state value as readable text, never JSON: p/q-values as p-values, scalar lists joined,
 * maps as "k v · k v" (nested maps in parentheses), lists of labelled objects as "stage: reason".
 */
export function fmtStateValue(v: unknown, key = ""): string {
  if (typeof v === "number" && P_KEY.test(key)) return fmtP(v);
  if (isScalar(v)) return fmtValue(v, key);
  if (Array.isArray(v)) {
    if (!v.length) return "none";
    if (isScalarList(v)) return v.map((x) => fmtValue(x)).join(", ");
    return v.map((x) => fmtItem(x)).join("; ");
  }
  const entries = Object.entries(v as Record<string, unknown>);
  if (!entries.length) return "none";
  return entries
    .map(([k, x]) => {
      const inner = fmtStateValue(x, k);
      return `${humanize(k)} ${isScalar(x) || isScalarList(x) ? (isScalarList(x) ? `[${inner}]` : inner) : `(${inner})`}`;
    })
    .join(" · ");
}

function fmtItem(x: unknown): string {
  if (!x || typeof x !== "object" || Array.isArray(x)) return fmtStateValue(x);
  const o = x as Record<string, unknown>;
  const lk = itemLabelKey(o);
  if (!lk) return fmtStateValue(o);
  const rest = Object.fromEntries(Object.entries(o).filter(([k]) => k !== lk));
  const keys = Object.keys(rest);
  const body = keys.length === 1 ? fmtStateValue(rest[keys[0]], keys[0]) : fmtStateValue(rest);
  return keys.length ? `${humanize(String(o[lk]))}: ${body}` : humanize(String(o[lk]));
}

function rowsFor(key: string, value: unknown, stage: string | null): StateRow[] {
  const strip = (k: string) => (stage ? k.replace(new RegExp(`^${stage}s?_`, "i"), "") : k);
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return Object.entries(value as Record<string, unknown>).map(([k2, v2]) => ({ key: `${key}.${k2}`, label: humanize(k2), value: fmtStateValue(v2, k2) }));
  }
  const label = key.toLowerCase() === stage || key.toLowerCase() === `${stage}s` ? "value" : humanize(strip(key));
  return [{ key, label, value: fmtStateValue(value, key) }];
}

/**
 * A decision's input state as label/value rows: nested objects are flattened (labels joined by
 * " · ", up to `depth` levels); small maps from the second level down, scalar lists and lists of
 * labelled objects ("stage: reason", one row each) stay readable. Never JSON.
 */
export function flattenState(state: Record<string, unknown> | null | undefined, depth = 3): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [];
  const walk = (v: unknown, key: string, label: string, level: number) => {
    const isObj = Boolean(v) && typeof v === "object" && !Array.isArray(v);
    const inline = isScalarMap(v) && Object.keys(v as object).length <= 8 && level > 1;
    if (isObj && level < depth && !inline) {
      for (const [k, x] of Object.entries(v as Record<string, unknown>)) walk(x, k, label ? `${label} · ${humanize(k)}` : humanize(k), level + 1);
      return;
    }
    if (Array.isArray(v) && v.length && v.every((x) => isScalarMap(x) && itemLabelKey(x) !== null)) {
      for (const x of v as Record<string, unknown>[]) {
        const lk = itemLabelKey(x) as string;
        const rest = Object.fromEntries(Object.entries(x).filter(([k]) => k !== lk));
        const keys = Object.keys(rest);
        rows.push({ label: `${label} · ${humanize(String(x[lk]))}`, value: keys.length === 1 ? fmtStateValue(rest[keys[0]], keys[0]) : fmtStateValue(rest) });
      }
      return;
    }
    rows.push({ label, value: fmtStateValue(v, key) });
  };
  walk(state ?? {}, "", "", 0);
  return rows;
}

export interface PointComponent {
  stage: string;
  /** stage_group when the component carries one (change_point → trend) */
  group: string | null;
  points: number | null;
  title: string | null;
  detail: string | null;
  ref: string | null;
}

function asComponents(v: unknown): PointComponent[] | null {
  if (!Array.isArray(v) || !v.length || !v.every((x) => x && typeof x === "object" && "stage" in x)) return null;
  return (v as Record<string, unknown>[])
    .map((c) => ({
      stage: String(c.stage),
      group: typeof c.stage_group === "string" ? c.stage_group : null,
      points: typeof c.points === "number" ? c.points : null,
      title: typeof c.title === "string" ? c.title : null,
      detail: typeof c.detail === "string" ? c.detail : null,
      ref: typeof c.ref === "string" ? c.ref : null,
    }))
    .sort((a, b) => (b.points ?? -Infinity) - (a.points ?? -Infinity));
}

/** `skipped` as stage → reason ([{stage, reason}] or plain stage names). */
function skippedStages(v: unknown): Map<EvidenceStage, string> {
  const out = new Map<EvidenceStage, string>();
  if (!Array.isArray(v)) return out;
  for (const x of v) {
    if (typeof x === "string") {
      const st = stageFromName(x);
      if (st) out.set(st, "skipped");
    } else if (x && typeof x === "object") {
      const o = x as Record<string, unknown>;
      const st = stageFromName(typeof o.stage === "string" ? o.stage : null);
      if (st) out.set(st, typeof o.reason === "string" ? o.reason : typeof o.detail === "string" ? o.detail : "skipped");
    }
  }
  return out;
}

const STRUCTURAL_KEYS = new Set(["components", "skipped", "skipped_stages"]);

/**
 * The early-warning state grouped by pipeline stage (signal → risk), in pipeline order, so the
 * snapshot reads like the evidence chain. A stage is observed when a point component belongs to
 * it (its points and strongest summary fill the cell); legacy stage blocks ({trend: {q}} or
 * trend_q) are shown too. A stage listed in `skipped` carries its reason; every other key is
 * returned as `other`.
 */
export function ewlStateGroups(state: Record<string, unknown> | null | undefined): { groups: StageGroup[]; other: StateRow[]; components: PointComponent[] } {
  const byStage = new Map<EvidenceStage, StateRow[]>();
  const other: StateRow[] = [];
  const st = state ?? {};
  const components = asComponents(st.components) ?? [];
  const skipped = skippedStages(st.skipped ?? st.skipped_stages);
  for (const [k, v] of Object.entries(st)) {
    if (STRUCTURAL_KEYS.has(k)) continue;
    const stage = stageOf(k);
    if (stage) byStage.set(stage, [...(byStage.get(stage) ?? []), ...rowsFor(k, v, stage)]);
    else other.push(...rowsFor(k, v, null));
  }
  const groups = EVIDENCE_STAGES.map((s) => {
    const comps = components.filter((c) => (stageFromName(c.group) ?? stageFromName(c.stage)) === s.key);
    const rows: StateRow[] = [];
    if (comps.length) {
      const max = Math.max(...comps.map((c) => c.points ?? -Infinity));
      rows.push({ key: `${s.key}.points`, label: "points", value: Number.isFinite(max) ? fmtValue(max) : "—" });
      if (comps.length > 1) rows.push({ key: `${s.key}.n`, label: "components", value: String(comps.length) });
    }
    const legacy = (byStage.get(s.key) ?? []).filter((r) => r.value !== "—");
    rows.push(...legacy);
    const top = comps[0];
    return {
      stage: s.key,
      label: s.label,
      what: s.what,
      rows,
      summary: top ? (top.title ?? top.detail) : null,
      observed: comps.length > 0 || legacy.length > 0,
      skippedReason: comps.length ? null : (skipped.get(s.key) ?? null),
    };
  });
  return { groups, other, components };
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
