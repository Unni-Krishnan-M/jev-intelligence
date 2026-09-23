/**
 * Small helpers shared by the Intelligence console: formatting, series naming, evidence links and
 * the warning lifecycle. Pure functions only; nothing here fetches.
 */

import { useSWRConfig } from "swr";

import { ApiError } from "./api";
import type { Severity, WarningStatus } from "./intel-types";

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

/** 404/503 from a run-backed endpoint means no pipeline run exists yet. */
export function isNoRun(err: unknown): boolean {
  return err instanceof ApiError && (err.status === 404 || err.status === 503);
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
