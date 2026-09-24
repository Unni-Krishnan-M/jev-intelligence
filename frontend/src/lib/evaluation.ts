/**
 * Helpers for the v1.2 evaluation reports (platform-eval, drift-eval). Pure functions only.
 */

import { humanize } from "./intel";
import type { AdaptationGroup, DriftEvalReport, PairedDelta, PlatformDomainEvaluation } from "./intel-types";

/** "0.22 [0.14, 0.32]"; a dash for a missing value, the interval only when there is one. */
export function fmtWithCi(v: number | null | undefined, ci?: [number, number] | null, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const f = (x: number) => (Math.abs(x) < 1e-12 ? 0 : x).toFixed(digits);
  return ci ? `${f(v)} [${f(ci[0])}, ${f(ci[1])}]` : f(v);
}

/** "+0.0063 [0.0014, 0.0134]" for a paired delta. */
export function fmtDelta(d: PairedDelta | null | undefined, digits = 4): string {
  if (!d || d.delta === null || d.delta === undefined) return "—";
  const f = (x: number) => (Math.abs(x) < 1e-12 ? 0 : x).toFixed(digits);
  const sign = d.delta > 0 ? "+" : d.delta < 0 ? "−" : "±";
  return `${sign}${f(Math.abs(d.delta))}${d.ci95 ? ` [${f(d.ci95[0])}, ${f(d.ci95[1])}]` : ""}`;
}

/** Does the 95 % interval exclude zero, and on which side? */
export function deltaVerdict(d: PairedDelta | null | undefined): "better" | "worse" | "unclear" {
  if (!d?.ci95) return "unclear";
  if (d.ci95[0] > 0) return "better";
  if (d.ci95[1] < 0) return "worse";
  return "unclear";
}

/** Splice sizes in numeric order ("10", "20", "40"). */
export function spliceSizes(by: Record<string, unknown>): string[] {
  return Object.keys(by).sort((a, b) => Number(a) - Number(b));
}

/** The policy's own strategies; exploratory variants (named *_strong_* / *_lambda_*) are left out. */
export const POLICY_VARIANTS = ["adapt_to_recent", "explore"] as const;

export const GROUP_LABEL: Record<string, string> = {
  preference_drift: "Preference drift detected",
  any_drift: "Any drift detected",
  no_drift_tested: "Tested, no drift",
  insufficient_history: "Too little history",
  preference_drift_event_mode: "Preference drift (event permutation)",
  all_users: "All users",
};

export interface AdaptationRow {
  group: string;
  label: string;
  n: number | null;
  standard: number | null;
  deltas: { variant: string; d: PairedDelta | null }[];
}

/** One row per user group for a setting (refit / active), in the report's group order. */
export function adaptationRows(r: Pick<DriftEvalReport, "adaptation">, setting: string): AdaptationRow[] {
  const groups: Record<string, AdaptationGroup> = r.adaptation.settings?.[setting] ?? {};
  return Object.entries(groups).map(([group, g]) => ({
    group,
    label: GROUP_LABEL[group] ?? humanize(group),
    n: r.adaptation.group_sizes?.[group] ?? null,
    standard: g.mean_ndcg10?.standard ?? null,
    deltas: POLICY_VARIANTS.map((variant) => ({ variant, d: g.vs_standard?.[variant]?.ndcg10 ?? null })),
  }));
}

/** Warnings raised per replay, as bars: labelled by as-of month when the report lists them. */
export function replayBars(counts: number[] | undefined, asOf?: string[]): { bin: string; n: number }[] {
  return (counts ?? []).map((n, i) => ({ bin: asOf?.[i]?.slice(0, 7) ?? String(i + 1), n }));
}

/** Movie reports one consistency block; generic domains pool it within their replay windows. */
export function consistencyOf(p: PlatformDomainEvaluation) {
  const c = p.consistency ?? p.consistency_pooled_within_windows ?? null;
  if (!c) return null;
  const rate = (num: number, den: number) => (den > 0 ? num / den : null);
  return {
    pairs: c.situation_pairs,
    flips: c.flips,
    flipRate: c.flip_rate ?? rate(c.flips, c.situation_pairs),
    boundaryFlips: c.warning_boundary_flips,
    boundaryRate: c.warning_boundary_flip_rate ?? rate(c.warning_boundary_flips, c.situation_pairs),
    pooled: !p.consistency,
  };
}
