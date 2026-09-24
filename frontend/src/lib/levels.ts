/**
 * The early-warning level (docs/platform.md §4): its spec, the four answers and their words. Kept
 * free of other imports so lib/intel.ts can use it without a cycle; lib/decisions.ts re-exports it.
 */

import type { Decision, EarlyWarningLevel } from "./intel-types";

export const EARLY_WARNING_SPEC = "early_warning_level";

/** Least to most severe. A warning is raised only for the last two. */
export const EARLY_WARNING_LEVELS: EarlyWarningLevel[] = ["NO_ACTION", "MONITOR", "WARNING", "URGENT_ACTION"];

export const LEVEL_LABEL: Record<EarlyWarningLevel, string> = {
  NO_ACTION: "No action",
  MONITOR: "Monitor",
  WARNING: "Warning",
  URGENT_ACTION: "Urgent action",
};

export function isEarlyWarningLevel(v: unknown): v is EarlyWarningLevel {
  return typeof v === "string" && (EARLY_WARNING_LEVELS as string[]).includes(v);
}

/** The early-warning decision by spec, key or (for older logs) its exact option set. */
export function isEarlyWarningDecision(d: Partial<Pick<Decision, "spec_id" | "key" | "options">>): boolean {
  const key = d.key ?? "";
  if (d.spec_id === EARLY_WARNING_SPEC || key === EARLY_WARNING_SPEC || key.startsWith(`${EARLY_WARNING_SPEC}:`)) return true;
  const options = d.options ?? [];
  return options.length === EARLY_WARNING_LEVELS.length && EARLY_WARNING_LEVELS.every((l) => options.includes(l));
}
