/**
 * Pure helpers for the Operations pages (events, model lifecycle, online experiments). Nothing
 * here fetches; every rule that decides what a button allows or what a sentence says lives here so
 * it is unit-tested (src/__tests__/ops.test.ts).
 */

import type {
  Conclusion,
  DataSource,
  EventDomainHealth,
  ExperimentAction,
  ExperimentStatus,
  GateStatus,
  GateSummary,
  GovernedModel,
  JobStatus,
  JobStep,
  ModelState,
  SrmTest,
  TrainingJobOut,
} from "./ops-types";

// ---- durations -------------------------------------------------------------------------------

/** 5 s · 12 min · 3.2 h · 4.0 d · 61 d */
export function fmtSeconds(s: number | null | undefined): string {
  if (s === null || s === undefined || Number.isNaN(s)) return "—";
  const a = Math.max(0, s);
  if (a < 60) return `${Math.round(a)} s`;
  if (a < 3600) return `${Math.round(a / 60)} min`;
  if (a < 86400) return `${(a / 3600).toFixed(1)} h`;
  const d = a / 86400;
  return d < 10 ? `${d.toFixed(1)} d` : `${Math.round(d).toLocaleString("en")} d`;
}

// ---- events: lag thresholds (docs/STREAMING_ARCHITECTURE.md §6) --------------------------------

export type LagLevel = "ok" | "watch" | "stale" | "none";

/** Silence on a feed that should be live: amber after about an hour, stale after a day. */
export const INGEST_WATCH_S = 3600;
export const INGEST_STALE_S = 86400;

/**
 * How old the newest event_time may be before it is unusual, per domain. The movie domain's app
 * events are live (so the same hour as ingest); a generic domain's observations are period starts,
 * so a monthly statistic is weeks old by nature.
 */
export function expectedEventLagS(domain: string, frequency?: string | null): number {
  if (domain === "movie") return INGEST_WATCH_S;
  switch (frequency) {
    case "day":
      return 2 * 86400;
    case "week":
      return 10 * 86400;
    case "month":
      return 45 * 86400;
    case "quarter":
      return 120 * 86400;
    case "year":
      return 400 * 86400;
    default:
      return 45 * 86400;
  }
}

export function ingestLagLevel(seconds: number | null | undefined): LagLevel {
  if (seconds === null || seconds === undefined) return "none";
  if (seconds <= INGEST_WATCH_S) return "ok";
  if (seconds <= INGEST_STALE_S) return "watch";
  return "stale";
}

/** ok within the expectation, watch up to twice it, stale beyond. */
export function eventLagLevel(seconds: number | null | undefined, expectedS: number): LagLevel {
  if (seconds === null || seconds === undefined) return "none";
  if (seconds <= expectedS) return "ok";
  if (seconds <= 2 * expectedS) return "watch";
  return "stale";
}

export const LAG_LABEL: Record<LagLevel, string> = { ok: "On time", watch: "Late", stale: "Stale", none: "No events" };

/** Rejections are the loudest events alert: a producer is sending bad data. */
export function eventsAlert(d: Pick<EventDomainHealth, "today" | "last_7_days">): "rejected_today" | "rejected_week" | null {
  if (d.today.rejected > 0) return "rejected_today";
  if (d.last_7_days.rejected > 0) return "rejected_week";
  return null;
}

/** "N events not yet in intelligence" */
export function pendingText(n: number): string {
  if (n <= 0) return "Every event is in intelligence";
  return `${n.toLocaleString("en")} event${n === 1 ? "" : "s"} not yet in intelligence`;
}

// ---- governance --------------------------------------------------------------------------------

/** Gate display order: the order the gate runs them in (jev_ml/governance/gates.py). */
export const GATE_ORDER = ["split", "artifact", "ndcg@10", "recall@10", "cold_start", "coverage@10", "calibration", "latency"];

export const GATE_LABEL: Record<string, string> = {
  split: "Frozen split",
  artifact: "Artifact loads",
  "ndcg@10": "NDCG@10 non-inferior",
  "recall@10": "Recall@10 non-inferior",
  cold_start: "Cold-start NDCG@10",
  "coverage@10": "Catalogue coverage",
  calibration: "Calibration",
  latency: "Latency p95",
};

export type GateTone = "good" | "bad" | "neutral";

/** The one mapping from a gate status to its word and tone (icon chosen by the badge). */
export function gateStatusMeta(status: GateStatus | string | null | undefined): { label: string; tone: GateTone } {
  switch (status) {
    case "pass":
      return { label: "Pass", tone: "good" };
    case "fail":
      return { label: "Fail", tone: "bad" };
    case "skipped":
      return { label: "Skipped", tone: "neutral" };
    default:
      return { label: "Not run", tone: "neutral" };
  }
}

export function orderedGates(gates: Record<string, GateSummary>): [string, GateSummary][] {
  const rank = (k: string) => {
    const i = GATE_ORDER.indexOf(k);
    return i < 0 ? GATE_ORDER.length : i;
  };
  return Object.entries(gates).sort((a, b) => rank(a[0]) - rank(b[0]) || a[0].localeCompare(b[0]));
}

export function gateCounts(gates: Record<string, GateSummary>): { pass: number; fail: number; skipped: number } {
  const out = { pass: 0, fail: 0, skipped: 0 };
  for (const g of Object.values(gates)) {
    if (g.status === "pass") out.pass++;
    else if (g.status === "fail") out.fail++;
    else out.skipped++;
  }
  return out;
}

/** The overall gate state of a version, including "stale": gated against a model that no longer serves. */
export type GateVerdict = "pass" | "fail" | "not_evaluated" | "stale_pass" | "stale_fail";

export function gateVerdict(m: Pick<GovernedModel, "gate_passed" | "gated_against" | "is_active">, active: string | null): GateVerdict {
  if (m.gate_passed === null) return "not_evaluated";
  const stale = !m.is_active && (m.gated_against ?? null) !== (active ?? null);
  if (m.gate_passed) return stale ? "stale_pass" : "pass";
  return stale ? "stale_fail" : "fail";
}

export const GATE_VERDICT_LABEL: Record<GateVerdict, string> = {
  pass: "Gate passed",
  fail: "Gate failed",
  not_evaluated: "Not evaluated",
  stale_pass: "Passed · stale",
  stale_fail: "Failed · stale",
};

export function isGateStale(m: Pick<GovernedModel, "gate_passed" | "gated_against" | "is_active">, active: string | null): boolean {
  const v = gateVerdict(m, active);
  return v === "stale_pass" || v === "stale_fail";
}

export const MODEL_STATE_LABEL: Record<ModelState, string> = {
  active: "Active",
  candidate: "Candidate",
  rejected: "Rejected",
  retired: "Retired",
};

/**
 * What the promote controls allow for a version. Promote needs `promotable` (the API's verdict:
 * gate passed against the model active now). Force is offered only when promote is blocked, and
 * always needs a reason and a confirmation; the active version offers neither.
 */
export function promoteControls(m: Pick<GovernedModel, "is_active" | "state" | "promotable" | "blockers">): {
  canPromote: boolean;
  canForce: boolean;
  why: string[];
} {
  if (m.is_active || m.state === "active") return { canPromote: false, canForce: false, why: ["This version is serving now."] };
  if (m.promotable) return { canPromote: true, canForce: false, why: [] };
  return { canPromote: false, canForce: true, why: m.blockers.length ? m.blockers : ["The gate has not passed against the active model."] };
}

/** A required reason (force, rollback): 3–500 characters after trimming. Returns the problem or null. */
export function reasonProblem(reason: string, min = 3, max = 500): string | null {
  const r = reason.trim();
  if (!r) return "A reason is required.";
  if (r.length < min) return `Give at least ${min} characters.`;
  if (r.length > max) return `Keep it under ${max} characters.`;
  return null;
}

/** Numbers of a non-inferiority gate (ndcg@10, recall@10, cold_start), or null for other gates. */
export function nonInferiority(g: GateSummary): { diff: number; lo: number; hi: number; margin: number; level: number | null; p: number | null; n: number | null } | null {
  const n = g.numbers;
  const num = (k: string) => (typeof n[k] === "number" ? (n[k] as number) : null);
  const diff = num("diff");
  const lo = num("ci_lo");
  const hi = num("ci_hi");
  const margin = num("margin");
  if (diff === null || lo === null || hi === null || margin === null) return null;
  return { diff, lo, hi, margin, level: num("ci_level"), p: num("p_value"), n: num("n_users") };
}

export function jobIsActive(j: Pick<TrainingJobOut, "status">): boolean {
  return j.status === "queued" || j.status === "running";
}

export function anyJobActive(jobs: Pick<TrainingJobOut, "status">[] | undefined | null): boolean {
  return Boolean(jobs?.some(jobIsActive));
}

export const JOB_STATUS_LABEL: Record<JobStatus, string> = { queued: "Queued", running: "Running", succeeded: "Succeeded", failed: "Failed" };

/** One line per job step: "snapshot snap-1a2b… (reused) · 100,836 rows", "gate passed vs jev-…". */
export function stepSummary(s: JobStep): string {
  const bits: string[] = [];
  const v = (k: string) => s[k];
  switch (s.step) {
    case "snapshot": {
      bits.push(String(v("snapshot_id") ?? ""));
      if (v("reused")) bits.push("reused");
      const rc = v("row_counts") as Record<string, number> | undefined;
      if (rc && typeof rc.interactions === "number") bits.push(`${rc.interactions.toLocaleString("en")} interactions`);
      break;
    }
    case "train":
      bits.push(String(v("version") ?? ""));
      if (typeof v("seconds") === "number") bits.push(`${v("seconds")} s`);
      break;
    case "calibrate":
      if (v("error")) bits.push(`error: ${String(v("error"))}`);
      if (typeof v("ece") === "number") bits.push(`ECE ${(v("ece") as number).toFixed(4)}`);
      if (typeof v("seconds") === "number") bits.push(`${v("seconds")} s`);
      break;
    case "gate":
      bits.push(v("passed") ? "passed" : "failed");
      if (v("incumbent")) bits.push(`vs ${String(v("incumbent"))}`);
      if (typeof v("seconds") === "number") bits.push(`${v("seconds")} s`);
      break;
    case "promote":
      bits.push(String(v("version") ?? ""));
      if (v("auto")) bits.push("automatic");
      break;
    default:
      for (const [k, x] of Object.entries(s)) if (k !== "step" && k !== "at" && (typeof x === "string" || typeof x === "number")) bits.push(`${k} ${x}`);
  }
  return bits.filter(Boolean).join(" · ");
}

// ---- online experiments --------------------------------------------------------------------------

export const EXPERIMENT_STATUS_LABEL: Record<ExperimentStatus, string> = {
  draft: "Draft",
  running: "Running",
  paused: "Paused",
  stopped: "Stopped",
  concluded: "Concluded",
};

/** Lifecycle buttons, in the order they are shown, with the confirm copy. */
export const EXPERIMENT_ACTIONS: Record<Exclude<ExperimentAction, "update" | "ramp">, { label: string; confirm: string; destructive?: boolean }> = {
  start: { label: "Start", confirm: "Members on this surface start being assigned and served their variant." },
  pause: { label: "Pause", confirm: "Everyone is served the default while paused; assignments are kept and the surface stays taken." },
  stop: { label: "Stop", confirm: "Serving returns to the default for everyone. A stopped experiment cannot be restarted, only concluded.", destructive: true },
  conclude: { label: "Conclude", confirm: "The results are computed one last time and frozen with the decision. This cannot be undone." },
  delete: { label: "Delete draft", confirm: "The draft and its configuration are removed.", destructive: true },
};

export const LIFECYCLE_ORDER: (keyof typeof EXPERIMENT_ACTIONS)[] = ["start", "pause", "stop", "conclude", "delete"];

export function lifecycleActions(allowed: ExperimentAction[]): (keyof typeof EXPERIMENT_ACTIONS)[] {
  return LIFECYCLE_ORDER.filter((a) => allowed.includes(a));
}

export const METRIC_LABEL: Record<string, string> = {
  interaction_rate: "Interaction rate",
  positive_rate: "Positive rate",
  rating_rate: "Rating rate",
  feedback_rate: "Feedback rate",
  negative_rate: "Negative rate",
  ndcg_at_10: "NDCG@10",
  diversity: "Diversity",
  novelty: "Novelty (bits)",
  coverage: "Coverage",
  latency_p95_ms: "Latency p95",
};

export const RATE_METRICS = ["interaction_rate", "positive_rate", "rating_rate", "feedback_rate", "negative_rate"] as const;
export const MEAN_METRICS = ["ndcg_at_10", "diversity", "novelty"] as const;

export function isRateMetric(m: string): boolean {
  return (RATE_METRICS as readonly string[]).includes(m);
}

/** A metric value in its unit: rates as %, latency in ms, the rest to 4 decimals. */
export function fmtMetric(metric: string, v: number | null | undefined, signed = false): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const sign = signed ? (v > 0 ? "+" : v < 0 ? "−" : "±") : v < 0 ? "−" : "";
  const a = signed ? Math.abs(v) : Math.abs(v);
  if (isRateMetric(metric) || metric === "coverage") return `${sign}${(a * 100).toFixed(2)}${signed ? " pp" : " %"}`;
  if (metric === "latency_p95_ms") return `${sign}${a.toFixed(1)} ms`;
  if (metric === "novelty") return `${sign}${a.toFixed(2)}`;
  return `${sign}${a.toFixed(4)}`;
}

/** Live traffic, or an offline replay that must never be read as an online result. */
export function labelKind(dataSource: DataSource | string | null | undefined, label: string | null | undefined): "live" | "replay" {
  if (dataSource && dataSource !== "live") return "replay";
  if (label && /replay|offline|not live/i.test(label)) return "replay";
  return "live";
}

export function labelBanner(dataSource: DataSource | string | null | undefined, label: string | null | undefined): { kind: "live" | "replay"; title: string; body: string } {
  const kind = labelKind(dataSource, label);
  const text = (label ?? "").trim();
  if (kind === "replay") {
    return {
      kind,
      title: text ? text.toUpperCase() : "OFFLINE REPLAY, NOT LIVE TRAFFIC",
      body: "Nobody saw these lists. The replay compares rankings through the online pipeline; it cannot measure the effect of showing a list.",
    };
  }
  return { kind, title: text ? text.charAt(0).toUpperCase() + text.slice(1) : "Live traffic", body: "Members were assigned by hash and served their variant on the recommendations surface." };
}

/**
 * The conclusion as words. Only "ship" names a treatment; "keep_control" names the control. An
 * inconclusive result never names a variant (even if a winner field were set) and never uses
 * "winner", "wins" or "better".
 */
export function conclusionWording(c: Pick<Conclusion, "decision" | "winner">): { title: string; body: string; tone: "good" | "bad" | "neutral" } {
  switch (c.decision) {
    case "ship":
      return {
        title: c.winner ? `Ship ${c.winner}` : "Ship the treatment",
        body: "Its primary metric improved significantly at the adjusted α and none of its guardrails was breached.",
        tone: "good",
      };
    case "keep_control":
      return { title: "Keep the control", body: "Every treatment was significantly worse on the primary metric.", tone: "bad" };
    default:
      return {
        title: "Inconclusive",
        body: "No variant is declared ahead. The evidence does not support shipping any treatment; keep serving the control.",
        tone: "neutral",
      };
  }
}

/** "χ² 1.387 · p = 0.500 · no mismatch" */
export function fmtSrm(t: Pick<SrmTest, "chi2" | "p_value" | "detected">): string {
  if (t.chi2 === null || t.p_value === null) return "not computed (too few members)";
  const p = t.p_value < 0.001 ? "< 0.001" : t.p_value.toFixed(3);
  return `χ² ${t.chi2.toFixed(3)} · p = ${p} · ${t.detected ? "mismatch detected" : "no mismatch"}`;
}

export function srmRows(names: string[], t: Pick<SrmTest, "observed" | "expected">): { name: string; observed: number; expected: number | null; deltaPct: number | null }[] {
  return names.map((name, i) => {
    const o = t.observed[i] ?? 0;
    const e = t.expected ? (t.expected[i] ?? null) : null;
    return { name, observed: o, expected: e, deltaPct: e ? (o - e) / e : null };
  });
}

/** Weights → the share of enrolled traffic each variant gets. */
export function weightShares(weights: number[]): number[] {
  const total = weights.reduce((s, w) => s + (w > 0 ? w : 0), 0);
  return weights.map((w) => (total > 0 && w > 0 ? w / total : 0));
}

export const KEY_RE = /^[a-z][a-z0-9_-]{2,63}$/;
export const VARIANT_RE = /^[a-z][a-z0-9_-]{0,39}$/;

/** Client-side checks that mirror ExperimentCreate, so the form says what is wrong before the API does. */
export function experimentFormProblems(f: { key: string; name: string; variants: { name: string; is_control: boolean; weight: number }[] }): string[] {
  const out: string[] = [];
  if (!KEY_RE.test(f.key)) out.push("Key: 3–64 characters, lower-case letters, digits, - or _, starting with a letter.");
  if (f.key === "list") out.push("Key: “list” is reserved.");
  if (!f.name.trim()) out.push("Name is required.");
  if (f.variants.length < 2 || f.variants.length > 5) out.push("Between 2 and 5 variants.");
  const controls = f.variants.filter((v) => v.is_control).length;
  if (controls !== 1) out.push("Exactly one variant must be the control.");
  const names = f.variants.map((v) => v.name);
  if (new Set(names).size !== names.length) out.push("Variant names must be unique.");
  for (const v of f.variants) {
    if (!VARIANT_RE.test(v.name)) out.push(`Variant “${v.name || "(empty)"}”: lower-case letters, digits, - or _, starting with a letter.`);
    if (!(v.weight > 0 && v.weight <= 1000)) out.push(`Variant “${v.name || "(empty)"}”: weight must be above 0 and at most 1000.`);
  }
  return out;
}

// ---- decision lineage and warning auto-resolve (WS4) ---------------------------------------------

/** Lineage layers from the root down to the run that read the data. */
export const LINEAGE_LAYERS = [
  { key: "root", label: "Root" },
  { key: "objects", label: "Run objects (evidence)" },
  { key: "series", label: "Series" },
  { key: "sources", label: "Data sources" },
] as const;

export function lineageLayers(g: { root: { node: string }; nodes: import("./ops-types").LineageNode[] }): Record<(typeof LINEAGE_LAYERS)[number]["key"], import("./ops-types").LineageNode[]> {
  const out = { root: [], objects: [], series: [], sources: [] } as Record<(typeof LINEAGE_LAYERS)[number]["key"], import("./ops-types").LineageNode[]>;
  for (const n of g.nodes) {
    if (n.type === "run") continue;
    if (n.id === g.root.node) out.root.push(n);
    else if (n.type === "series") out.series.push(n);
    else if (n.type === "source") out.sources.push(n);
    else out.objects.push(n);
  }
  return out;
}

/** A warning history entry written by the system when a stale warning was resolved automatically. */
export function isAutoResolve(h: { actor: string; note: string | null }): boolean {
  return h.actor === "system" && (h.note ?? "").toLowerCase().startsWith("auto-resolved");
}
