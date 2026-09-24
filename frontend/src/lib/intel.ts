/**
 * Small helpers shared by the Intelligence console: formatting, series naming, evidence links and
 * the warning lifecycle. Pure functions only; nothing here fetches.
 */

import { useSWRConfig } from "swr";

import { ApiError } from "./api";
import type { CalibrationBin, CalibrationMetrics, ConfidenceKind, Decision, DecisionRecord, DecisionScale, EvidenceOwnerType, HistoryPoint, RecCalibration, Risk, Severity, WarningStatus } from "./intel-types";

export const SEVERITIES: Severity[] = ["critical", "high", "medium", "low"];
export const SEVERITY_RANK: Record<Severity, number> = { critical: 3, high: 2, medium: 1, low: 0 };

/** Allowed PATCH /intel/warnings/{id} transitions; resolved and dismissed are terminal. */
export const TRANSITIONS: Record<WarningStatus, WarningStatus[]> = {
  new: ["acknowledged", "investigating", "resolved", "dismissed"],
  acknowledged: ["investigating", "resolved", "dismissed"],
  investigating: ["resolved", "dismissed"],
  resolved: [],
  dismissed: [],
};
export const OPEN_STATUSES: WarningStatus[] = ["new", "acknowledged", "investigating"];

export function canTransition(from: WarningStatus, to: WarningStatus): boolean {
  return TRANSITIONS[from]?.includes(to) ?? false;
}

/** 404/503 from a run-backed endpoint means no pipeline run exists yet. */
export function isNoRun(err: unknown): boolean {
  return err instanceof ApiError && (err.status === 404 || err.status === 503);
}

/**
 * 404 "Not Found" (FastAPI's route-miss detail), 405 or 501: the endpoint does not exist on this
 * API build. Newer views use it to say "not available yet" instead of failing.
 */
export function isNotDeployed(err: unknown): boolean {
  return err instanceof ApiError && ((err.status === 404 && err.message === "Not Found") || err.status === 405 || err.status === 501);
}

/** 409 on a console read: the chosen domain cannot load here and has no stored runs (v1.2). */
export function isDomainUnavailable(err: unknown): boolean {
  return err instanceof ApiError && err.status === 409;
}

/** Revalidate every cached /intel/* response (after a run, a status change, feedback…). */
export function useIntelRevalidate() {
  const { mutate } = useSWRConfig();
  return () => mutate((k) => typeof k === "string" && (k.startsWith("/intel") || k.startsWith("intel:")));
}

// ---- formatting ------------------------------------------------------------------------------

const COMPACT = new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 });

export function fmtCompact(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Math.abs(v) >= 10000 ? COMPACT.format(v) : Math.round(v).toLocaleString("en");
}

export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)} %`;
}

export function fmtSigned(v: number | null | undefined, digits = 1, suffix = ""): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v).toFixed(digits)}${suffix}`;
}

export function fmtP(p: number | null | undefined): string {
  if (p === null || p === undefined || Number.isNaN(p)) return "—";
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

export function fmtDays(d: number | null | undefined): string {
  if (d === null || d === undefined || Number.isNaN(d)) return "—";
  if (d < 1) return `${(d * 24).toFixed(1)} h`;
  return `${d >= 100 ? Math.round(d).toLocaleString("en") : d.toFixed(1)} d`;
}

export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || Number.isNaN(ms)) return "—";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(ms < 10 ? 1 : 0)} ms`;
}

/** YYYY-MM-DD from any ISO string. */
export function fmtDay(iso: string | null | undefined): string {
  if (!iso) return "—";
  return iso.slice(0, 10);
}

/** "2017-04-01" → "Apr 2017" */
export function fmtMonth(t: string): string {
  const d = new Date(`${t.slice(0, 10)}T00:00:00Z`);
  return Number.isNaN(d.getTime()) ? t : d.toLocaleDateString("en-GB", { month: "short", year: "numeric", timeZone: "UTC" });
}

/** Days between two ISO timestamps (b - a). */
export function daysBetween(a: string, b: string): number | null {
  const x = new Date(a).getTime();
  const y = new Date(b).getTime();
  return Number.isNaN(x) || Number.isNaN(y) ? null : (y - x) / 86_400_000;
}

/** Values on a share series are fractions; ratings have two decimals; volumes are counts. */
export function fmtSeriesValue(v: number | null | undefined, metricOrUnit: string): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (metricOrUnit.startsWith("share")) return `${(v * 100).toFixed(1)} %`;
  if (metricOrUnit.startsWith("rating")) return v.toFixed(2);
  return fmtCompact(v);
}

export function humanize(s: string): string {
  return s.replaceAll("_", " ");
}

// ---- series & references ---------------------------------------------------------------------

const METRIC_LABEL: Record<string, string> = {
  volume: "Ratings",
  share: "Share of ratings",
  rating: "Mean rating",
  active_users: "Active raters",
};

const SERIES_RE = /^(volume|share|rating|active_users):/;

export function isSeriesId(ref: string): boolean {
  return SERIES_RE.test(ref);
}

/** "share:genre:Drama" → "Drama · share of ratings"; "volume:all" → "All films · ratings". */
export function seriesLabel(id: string): string {
  const [metric, ...rest] = id.split(":");
  const entity = rest.length >= 2 ? rest.slice(1).join(":") : rest[0] === "all" || !rest.length ? "All films" : rest.join(":");
  const m = METRIC_LABEL[metric] ?? humanize(metric);
  return `${entity} · ${m.toLowerCase()}`;
}

export function seriesHref(id: string): string {
  return `/intel/series/${encodeURIComponent(id)}`;
}

const REF_PREFIX: [string, string][] = [
  ["sig-", "/intel/signals"],
  ["trend-", "/intel/trends"],
  ["anom-", "/intel/anomalies"],
  ["fc-", "/intel/predictions"],
  ["risk-", "/intel/risks"],
  ["dec-", "/intel/decisions"],
  ["act-", "/intel/actions"],
];

/** Where an Evidence.ref (or a source {type, id}) points in the console; null if nowhere. */
export function refHref(ref: string | null | undefined): string | null {
  if (!ref) return null;
  if (isSeriesId(ref)) return seriesHref(ref);
  for (const [prefix, path] of REF_PREFIX) if (ref.startsWith(prefix)) return `${path}#${ref}`;
  return null;
}

/** Where the owner of an evidence row lives in the console (GET /intel/evidence). */
export function ownerHref(ownerType: EvidenceOwnerType | string, ownerId: string): string | null {
  if (!ownerId) return null;
  switch (ownerType) {
    case "decision":
      // the detail route accepts the db id and the contract id dec-…
      return `/intel/decisions/${encodeURIComponent(ownerId)}`;
    case "warning":
      return /^\d+$/.test(ownerId) ? `/intel/warnings/${ownerId}` : "/intel/warnings";
    case "forecast":
      return refHref(ownerId) ?? "/intel/predictions";
    default:
      return refHref(ownerId) ?? sourceHref({ type: ownerType, id: ownerId });
  }
}

export function sourceHref(src: { type?: string; id?: string } | null | undefined): string | null {
  if (!src?.id) return null;
  const fromRef = refHref(src.id);
  if (fromRef) return fromRef;
  switch (src.type) {
    case "warning":
      return /^\d+$/.test(src.id) ? `/intel/warnings/${src.id}` : "/intel/warnings";
    case "risk":
      return `/intel/risks#${src.id}`;
    case "anomaly":
      return `/intel/anomalies#${src.id}`;
    case "decision":
      return "/intel/decisions";
    default:
      return null;
  }
}

const SIG3 = new Intl.NumberFormat("en", { maximumSignificantDigits: 3 });
const INT = new Intl.NumberFormat("en", { maximumFractionDigits: 0 });
const DAYS_RE = /(^|_)(days?|d)($|_)|_days$/i;

/**
 * The one value formatter for spec rows, evidence, triggers, feature tables and state snapshots:
 * integers without decimals, day counts to whole days, other floats to 3 significant digits,
 * thousands separators throughout. `key` (a field name) lets day counts be recognised.
 */
export function fmtValue(v: unknown, key?: string): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v) || (key && DAYS_RE.test(key)) || Math.abs(v) >= 1000) return INT.format(v);
    return SIG3.format(v);
  }
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "string") return v;
  if (Array.isArray(v)) return v.map((x) => fmtValue(x)).join(", ");
  return JSON.stringify(v);
}

/** Pipeline order of stage timings; unknown stages go last, `total` is reported separately. */
export const STAGE_ORDER = [
  "load_inputs",
  "ingest_validate",
  "series",
  "trends",
  "anomalies",
  "forecast",
  "lapse",
  "risk",
  "decide",
  "recommend",
  "explain",
  "persist",
];

export function orderStages(stageMs: Record<string, number> | null | undefined): { rows: { name: string; value: number }[]; total: number | null } {
  const entries = Object.entries(stageMs ?? {});
  const total = entries.find(([k]) => k === "total")?.[1] ?? null;
  const rank = (k: string) => {
    const i = STAGE_ORDER.indexOf(k);
    return i < 0 ? STAGE_ORDER.length : i;
  };
  const rows = entries
    .filter(([k]) => k !== "total")
    .map(([name, value], i) => ({ name, value, i }))
    .sort((a, b) => rank(a.name) - rank(b.name) || a.i - b.i)
    .map(({ name, value }) => ({ name: humanize(name), value }));
  return { rows, total };
}

// ---- decisions, confidence and history (v1.1) ------------------------------------------------

/** A number on a decision's scale: "12.4 %", "3.2 slots". */
export function fmtScaleValue(v: number | null | undefined, scale?: DecisionScale | null): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const unit = scale?.unit ?? "";
  if (!unit) return fmtValue(v);
  return unit === "%" ? `${fmtValue(v)} %` : `${fmtValue(v)} ${unit}`;
}

/** The answer as text: score decisions on their scale, the chosen option otherwise. */
export function fmtAnswer(d: Pick<Decision, "kind" | "answer" | "scale">): string {
  if (d.answer === null || d.answer === undefined) return "—";
  if (d.kind === "score" && typeof d.answer === "number") return fmtScaleValue(d.answer, d.scale);
  return String(d.answer);
}

/**
 * The confidence as text. Only a probability is a percentage of being right; an interval's
 * percentage is the nominal coverage of its range; margin and rule stay 0–1 scores.
 */
export function fmtConfidence(
  value: number | null | undefined,
  kind?: ConfidenceKind | null,
  interval?: [number, number] | null,
  fmt: (v: number) => string = fmtValue,
): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  switch (kind) {
    case "probability":
      return `${(value * 100).toFixed(0)} %`;
    case "interval":
      return `${(value * 100).toFixed(0)} % interval${interval ? ` [${fmt(interval[0])}, ${fmt(interval[1])}]` : ""}`;
    case "rule":
      return value >= 1 ? "fired" : value.toFixed(2);
    default:
      return value.toFixed(2);
  }
}

/** History key for a risk across runs: risk ids carry the as_of, so the stable key is risk:kind:entity. */
export function riskHistoryKey(r: Pick<Risk, "kind" | "entity">): string {
  return `risk:${r.kind}:${r.entity}`;
}

/** Which history field to plot: score when every point has one, else value; null if neither. */
export function historyField(points: HistoryPoint[]): "score" | "value" | null {
  if (points.length && points.every((p) => typeof p.score === "number")) return "score";
  if (points.length && points.every((p) => typeof p.value === "number")) return "value";
  return null;
}

export type DecisionGroup = { key: string; batchId: string | null; items: DecisionRecord[] };

/** Decisions answered in one call stay together, at the position of the first one on the page. */
export function groupByBatch(items: DecisionRecord[]): DecisionGroup[] {
  const out: DecisionGroup[] = [];
  const at = new Map<string, DecisionGroup>();
  for (const d of items) {
    if (!d.batch_id) {
      out.push({ key: `d-${d.db_id}`, batchId: null, items: [d] });
      continue;
    }
    const k = `${d.run_id}|${d.batch_id}`;
    let g = at.get(k);
    if (!g) {
      g = { key: `b-${k}`, batchId: d.batch_id, items: [] };
      at.set(k, g);
      out.push(g);
    }
    g.items.push(d);
  }
  return out;
}

export interface CalibrationView {
  /** which split the numbers come from: "test · next_10", "validation (in-sample)", … */
  split: string;
  ece: number | null;
  brier: number | null;
  baseRateBrier: number | null;
  n: number | null;
  bins: CalibrationBin[];
  method: string | null;
  fittedOn: string | null;
}

function isMetrics(v: unknown): v is CalibrationMetrics {
  return Boolean(v) && typeof v === "object" && ("ece" in (v as object) || "brier" in (v as object));
}

/**
 * The held-out numbers to show for a recommendation calibration: its declared headline, else the
 * primary test protocol, else any test block, else validation (flagged in-sample), else top-level
 * metrics.
 */
export function calibrationView(cal: RecCalibration | null | undefined): CalibrationView | null {
  if (!cal) return null;
  let split = "reported";
  let m: CalibrationMetrics | null = null;
  const test = cal.test;
  if (isMetrics(cal.headline)) {
    m = cal.headline;
    split = [cal.headline.split, cal.headline.stratum ? humanize(cal.headline.stratum) : null].filter(Boolean).join(" · ") || "headline";
  } else if (isMetrics(test)) {
    m = test;
    split = "test";
  } else if (test && typeof test === "object") {
    const blocks = Object.entries(test).filter((e): e is [string, CalibrationMetrics] => isMetrics(e[1]));
    const pick = blocks.find(([, b]) => b.primary) ?? blocks[0];
    if (pick) {
      m = pick[1];
      split = `test · ${humanize(pick[0])}`;
    }
  }
  if (!m && isMetrics(cal.validation)) {
    m = cal.validation;
    split = "validation (in-sample)";
  }
  if (!m && isMetrics(cal)) m = cal;
  const num = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
  const fitted = cal.fitted_on;
  return {
    split,
    ece: num(m?.ece),
    brier: num(m?.brier),
    baseRateBrier: num(m?.base_rate_brier),
    n: num(m?.n),
    bins: (m?.reliability ?? m?.bins ?? []).filter((b) => typeof b?.predicted === "number" && typeof b?.observed === "number"),
    method: cal.method ? humanize(cal.method) : null,
    fittedOn: typeof fitted === "string" ? fitted : fitted && typeof fitted === "object" && typeof fitted.split === "string" ? `${fitted.split} split` : null,
  };
}
