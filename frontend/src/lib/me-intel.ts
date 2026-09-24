/**
 * Helpers for "My intelligence" (GET /me/intelligence, docs/platform.md §5 and §10): drift aspects,
 * preference-share segments and scenario rows. Pure functions only.
 */

import { fmtP, fmtValue, humanize } from "./intel";
import type { DriftAspect, DriftReport, PreferenceHistory, PreferenceScenarioResult, ProjectedShare } from "./intel-types";

export const ASPECT_LABEL: Record<string, string> = {
  genre_distribution: "Genre mix",
  rating_level: "Rating level",
  activity_rate: "Viewing frequency",
  release_year: "Release years",
  content_similarity: "Content similarity",
  acceptance: "Acceptance of recommendations",
};

export type AspectVerdict = "significant" | "not_significant" | "insufficient_data";

export interface DriftAspectView {
  aspect: string;
  label: string;
  test: string;
  statistic: string;
  p: string;
  pAdjusted: string;
  verdict: AspectVerdict;
  verdictLabel: string;
  detail: string | null;
}

export const VERDICT_LABEL: Record<AspectVerdict, string> = {
  significant: "Significant",
  not_significant: "Not significant",
  insufficient_data: "Insufficient data",
};

/** One table row per aspect. Below the minimum sample the numbers are dashes, never zeros. */
export function driftAspectView(a: DriftAspect): DriftAspectView {
  const insufficient = a.status === "insufficient_data";
  const verdict: AspectVerdict = insufficient ? "insufficient_data" : a.significant ? "significant" : "not_significant";
  return {
    aspect: a.aspect,
    label: ASPECT_LABEL[a.aspect] ?? humanize(a.aspect),
    test: a.test || "—",
    statistic: insufficient ? "—" : fmtValue(a.statistic),
    p: insufficient ? "—" : fmtP(a.p_value),
    pAdjusted: insufficient ? "—" : fmtP(a.p_adjusted),
    verdict,
    verdictLabel: VERDICT_LABEL[verdict],
    detail: a.detail,
  };
}

export type DriftVerdict = "detected" | "not_detected" | "insufficient_data";

export const EVIDENCE_NOTE = "evidence (1 − adjusted p), not a probability";

/** The report's headline: detected / not detected / not enough data, and its evidence strength. */
export function driftHeadline(d: Pick<DriftReport, "status" | "drift_detected" | "confidence">): { verdict: DriftVerdict; label: string; confidence: string } {
  if (d.status === "insufficient_data") return { verdict: "insufficient_data", label: "Not enough history to test", confidence: "—" };
  const c = d.confidence === null || d.confidence === undefined || Number.isNaN(d.confidence) ? "—" : d.confidence.toFixed(2);
  return d.drift_detected
    ? { verdict: "detected", label: "Drift detected", confidence: c }
    : { verdict: "not_detected", label: "No drift detected", confidence: c };
}

// ---- preference shares -----------------------------------------------------------------------

export const OTHER = "Other";

/**
 * The categories to draw: the `max` largest by mean share across windows, in that fixed order,
 * then everything else folded into "Other". The order is computed once for the whole history so a
 * category keeps its colour in every window.
 */
export function topCategories(h: PreferenceHistory, max = 5): { shown: string[]; folded: string[] } {
  const cats = h.categories.length ? h.categories : Array.from(new Set(h.windows.flatMap((w) => Object.keys(w.shares))));
  const n = Math.max(1, h.windows.length);
  const mean = new Map(cats.map((c) => [c, h.windows.reduce((s, w) => s + (w.shares[c] ?? 0), 0) / n]));
  const ranked = cats.map((c, i) => ({ c, i, m: mean.get(c) ?? 0 })).sort((a, b) => b.m - a.m || a.i - b.i);
  const withShare = ranked.filter((r) => r.m > 0);
  const shown = withShare.slice(0, max).map((r) => r.c);
  return { shown, folded: withShare.slice(max).map((r) => r.c) };
}

export interface ShareSegment {
  category: string;
  share: number;
  /** index into the categorical slots; null for "Other" (neutral ink) */
  slot: number | null;
}

/** A window's segments in the fixed category order, "Other" last (omitted when zero). */
export function shareSegments(shares: Record<string, number>, shown: string[]): ShareSegment[] {
  const segs: ShareSegment[] = shown.map((c, i) => ({ category: c, share: Math.max(0, shares[c] ?? 0), slot: i }));
  const inShown = segs.reduce((s, x) => s + x.share, 0);
  const total = Object.values(shares).reduce((s, v) => s + Math.max(0, v), 0);
  const rest = Math.max(0, total - inShown);
  if (rest > 1e-9) segs.push({ category: OTHER, share: rest, slot: null });
  return segs;
}

export function fmtShare(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(v < 0.1 ? 1 : 0)} %`;
}

// ---- scenarios -------------------------------------------------------------------------------

export interface ProjectedRow {
  category: string;
  baseline: number | null;
  projected: ProjectedShare | null;
}

/**
 * Rows for one scenario's projected shares, largest projected mean first, capped at `max`.
 * Categories missing from the projection stay out rather than being drawn at zero.
 */
export function projectedRows(s: Pick<PreferenceScenarioResult, "projected_shares">, baseline: Record<string, number>, max = 6): ProjectedRow[] {
  return Object.entries(s.projected_shares ?? {})
    .filter(([, p]) => p && typeof p.mean === "number")
    .sort((a, b) => b[1].mean - a[1].mean)
    .slice(0, max)
    .map(([category, p]) => ({ category, baseline: typeof baseline?.[category] === "number" ? baseline[category] : null, projected: p }));
}

/** Upper end of a shared axis for projected shares (at least 10 %, rounded up to 10 %). */
export function shareAxisMax(rows: ProjectedRow[]): number {
  const hi = rows.reduce((m, r) => Math.max(m, r.projected?.hi80 ?? 0, r.baseline ?? 0), 0);
  return Math.min(1, Math.max(0.1, Math.ceil(hi * 10) / 10));
}
