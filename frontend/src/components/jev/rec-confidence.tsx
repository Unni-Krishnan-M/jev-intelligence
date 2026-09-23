import type { RecItem } from "@/lib/types";
import { cn } from "@/lib/utils";

type WithConfidence = Pick<RecItem, "confidence" | "confidence_kind">;

/** The calibrated P(rating >= 4), or null when there is none to show. Never guessed. */
export function calibratedConfidence(item: WithConfidence): number | null {
  const p = item.confidence;
  if (item.confidence_kind !== "probability" || p === null || p === undefined || !Number.isFinite(p)) return null;
  return Math.max(0, Math.min(1, p));
}

/** "≈3 %": whole percents from 1 % up, one decimal below, where rounding would erase the figure. */
export function fmtChance(p: number): string {
  if (p < 0.001) return "under 0.1 %";
  return p < 0.01 ? `≈${(p * 100).toFixed(1)} %` : `≈${Math.round(p * 100)} %`;
}

/**
 * The calibration target is "rated 4★ or higher among the member's next five ratings". Most films
 * are never rated at all, so the figure is small by design; the sentence states exactly that and
 * never reads as a match score.
 */
export function confidenceSentence(p: number): string {
  return `${fmtChance(p)} chance this is one of your next five 4★+ ratings`;
}

/**
 * Member-facing confidence, for the Why this? dialog only: small numbers on every card would read
 * as a poor match score. Renders nothing without a calibrated value (no calibration file, beyond
 * rank 50, similar/trending lists).
 */
export function RecConfidence({ item, className }: { item: WithConfidence; className?: string }) {
  const p = calibratedConfidence(item);
  if (p === null) return null;
  return (
    <div className={cn("border-l-2 border-primary/60 pl-3", className)}>
      <p className="text-sm">
        {confidenceSentence(p)} <span className="text-muted-foreground">— calibrated on held-out ratings.</span>
      </p>
      <p className="mt-0.5 text-xs text-muted-foreground">Most films are never rated at all, so this is small by design. It is an estimate, not a match score.</p>
    </div>
  );
}
