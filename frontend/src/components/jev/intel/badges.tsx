"use client";

import {
  Archive,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  Ban,
  CheckCircle2,
  CircleAlert,
  CircleDashed,
  CircleDot,
  Clock,
  Eye,
  Info,
  Minus,
  MinusCircle,
  OctagonAlert,
  Package,
  Search,
  TriangleAlert,
} from "lucide-react";

import { Status } from "@/components/jev/admin/ui";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fmtConfidence, fmtDays } from "@/lib/intel";
import type { ConfidenceKind, Direction, Severity, SystemStatus, WarningStatus } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

/*
 * Every state in the console is carried by an icon + a word. The status colours only tint the
 * icon, so the meaning survives greyscale, colour-blindness and forced-colours.
 */

const CHIP = "inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded border hairline px-1.5 py-0.5 text-xs text-foreground";

export const SEVERITY_META: Record<Severity, { label: string; Icon: typeof Info; color: string }> = {
  critical: { label: "Critical", Icon: OctagonAlert, color: "var(--status-critical)" },
  high: { label: "High", Icon: TriangleAlert, color: "var(--status-serious)" },
  medium: { label: "Medium", Icon: CircleAlert, color: "var(--status-warning)" },
  low: { label: "Low", Icon: Info, color: "var(--muted-foreground)" },
};

export function SeverityBadge({ severity, className, prefix }: { severity: Severity; className?: string; prefix?: string }) {
  const m = SEVERITY_META[severity] ?? SEVERITY_META.low;
  return (
    <span className={cn(CHIP, className)}>
      <m.Icon className="size-3.5" style={{ color: m.color }} aria-hidden />
      {prefix && <span className="text-muted-foreground">{prefix}</span>}
      {m.label}
    </span>
  );
}

export const SYSTEM_META: Record<SystemStatus, { label: string; Icon: typeof Info; color: string; blurb: string }> = {
  nominal: { label: "Nominal", Icon: CheckCircle2, color: "var(--status-good)", blurb: "Nothing needs attention." },
  watch: { label: "Watch", Icon: CircleAlert, color: "var(--status-warning)", blurb: "Something is moving; keep an eye on it." },
  alert: { label: "Alert", Icon: OctagonAlert, color: "var(--status-critical)", blurb: "At least one warning needs a response." },
};

export function SystemStatusBadge({ status, className }: { status: SystemStatus; className?: string }) {
  const m = SYSTEM_META[status] ?? SYSTEM_META.watch;
  return (
    <span className={cn(CHIP, className)}>
      <m.Icon className="size-3.5" style={{ color: m.color }} aria-hidden />
      {m.label}
    </span>
  );
}

export const WARNING_STATUS_META: Record<WarningStatus, { label: string; Icon: typeof Info }> = {
  new: { label: "New", Icon: CircleDot },
  acknowledged: { label: "Acknowledged", Icon: Eye },
  investigating: { label: "Investigating", Icon: Search },
  resolved: { label: "Resolved", Icon: CheckCircle2 },
  dismissed: { label: "Dismissed", Icon: Ban },
};

export function WarningStatusBadge({ status, className }: { status: WarningStatus; className?: string }) {
  const m = WARNING_STATUS_META[status] ?? WARNING_STATUS_META.new;
  return (
    <span className={cn(CHIP, status === "new" && "border-primary/60", className)}>
      <m.Icon className={cn("size-3.5", status === "new" ? "text-primary" : "text-muted-foreground")} aria-hidden />
      {m.label}
    </span>
  );
}

export const CONFIDENCE_EXPLAIN: Record<ConfidenceKind, string> = {
  probability: "Probability: from a probabilistic computation (a bootstrap or a calibrated classifier). Still an estimate, not a guarantee.",
  margin: "Margin: the normalised evidence gap between the best and second-best option. It is not a probability.",
  rule: "Rule: a deterministic threshold rule. 1 means the rule fired; it says nothing about how likely the answer is to be right.",
  interval: "Interval: the nominal coverage of the range around a numeric answer. It is not a probability that the answer is right.",
  evidence: "Evidence: 1 − the smallest adjusted p-value of the tests behind it. It measures how strongly the data contradicts “no change”; it is not a probability.",
};

const SCORE_EXPLAIN = "A 0–1 confidence score from the evidence behind it. It is not a probability.";

/**
 * Confidence value + its kind. Only a probability is ever written as a percentage of being
 * right; an interval shows its nominal coverage with the range ("80 % interval [lo, hi]"), and
 * without a declared kind the value is shown as a plain 0–1 score.
 */
export function ConfidenceBadge({
  value,
  kind,
  interval,
  format,
  className,
}: {
  value: number | null;
  kind?: ConfidenceKind | null;
  /** interval kind: the [lo, hi] range the coverage refers to */
  interval?: [number, number] | null;
  /** formats the interval ends (e.g. on the decision's scale) */
  format?: (v: number) => string;
  className?: string;
}) {
  const shown = fmtConfidence(value, kind, interval, format);
  const explain = kind ? CONFIDENCE_EXPLAIN[kind] : SCORE_EXPLAIN;
  const tag = kind === "interval" ? null : (kind ?? "score");
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} className={cn(CHIP, "cursor-help", className)} aria-label={`Confidence ${shown}${tag ? `, ${tag}` : ""}. ${explain}`}>
          <span className="num">{shown}</span>
          {tag && <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-muted-foreground">{tag}</span>}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-64">{explain}</TooltipContent>
    </Tooltip>
  );
}

/** "3.8 h old", "12 d old", "2,921 d old" */
export function fmtAge(days: number | null | undefined): string {
  return days === null || days === undefined ? "age unknown" : `${fmtDays(days)} old`;
}

/**
 * Freshness depends on what the source is:
 * - static_snapshot: archival, its age is reported and never alarmed;
 * - live: "no events yet" when empty, otherwise fresh / stale against its SLA;
 * - artefact (the model): just its age.
 * Signals pass the source name instead of a kind (movielens / app / model).
 */
export function FreshnessBadge({
  kind,
  rows,
  days,
  fresh,
  className,
}: {
  kind?: string | null;
  rows?: number | null;
  days: number | null;
  fresh?: boolean | null;
  className?: string;
}) {
  const k = kind === "movielens" ? "static_snapshot" : kind === "app" ? "live" : kind === "model" ? "artefact" : kind;
  const chip = (Icon: typeof Info, label: string, detail: string | null, color?: string) => (
    <span className={cn(CHIP, className)}>
      <Icon className={cn("size-3.5", !color && "text-muted-foreground")} style={color ? { color } : undefined} aria-hidden />
      {label}
      {detail && <span className="num text-muted-foreground">{detail}</span>}
    </span>
  );
  if (k === "static_snapshot") return chip(Archive, "Archival", fmtAge(days));
  if (k === "artefact") return chip(Package, fmtAge(days), null);
  if (k === "live" && rows === 0) return chip(CircleDashed, "No events yet", null);
  if (fresh === true) return chip(CheckCircle2, "Fresh", fmtAge(days), "var(--status-good)");
  if (fresh === false) return chip(Clock, "Stale", fmtAge(days), "var(--status-serious)");
  if (k === "live") return chip(CircleDashed, "Not assessed", days === null ? null : fmtAge(days));
  return chip(Clock, fmtAge(days), null);
}

const DIRECTION_META: Record<Direction | "none", { label: string; Icon: typeof Info }> = {
  up: { label: "Rising", Icon: ArrowUpRight },
  down: { label: "Falling", Icon: ArrowDownRight },
  flat: { label: "Flat", Icon: ArrowRight },
  none: { label: "No direction", Icon: Minus },
};

export function DirectionIcon({ direction, withLabel = false, className }: { direction: Direction | null; withLabel?: boolean; className?: string }) {
  const m = DIRECTION_META[direction ?? "none"];
  return (
    <span className={cn("inline-flex items-center gap-1 text-sm", className)} title={withLabel ? undefined : m.label}>
      <m.Icon className="size-4 text-ink-2" aria-hidden />
      {withLabel ? <span>{m.label}</span> : <span className="sr-only">{m.label}</span>}
    </span>
  );
}

/** A 0..1 meter: accent fill on a lighter step of the same hue, value printed beside it. */
export function Meter({ value, label, className, digits = 2 }: { value: number | null; label?: string; className?: string; digits?: number }) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <span className={cn("inline-flex w-full items-center gap-2", className)}>
      <span
        className="relative h-1.5 min-w-12 flex-1 overflow-hidden rounded-full bg-primary/15"
        role="meter"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={value ?? 0}
        aria-label={label}
      >
        <span className="absolute inset-y-0 left-0 rounded-full bg-primary" style={{ width: `${v * 100}%` }} />
      </span>
      <span className="num w-10 shrink-0 text-right text-xs text-ink-2">{value === null ? "—" : value.toFixed(digits)}</span>
    </span>
  );
}

export function EffortBadge({ effort }: { effort: "low" | "medium" | "high" }) {
  const bars = effort === "low" ? 1 : effort === "medium" ? 2 : 3;
  return (
    <span className={CHIP}>
      <span className="flex items-end gap-[2px]" aria-hidden>
        {[1, 2, 3].map((i) => (
          <span key={i} className={cn("w-[3px] rounded-[1px]", i <= bars ? "bg-foreground" : "bg-rule")} style={{ height: 4 + i * 2 }} />
        ))}
      </span>
      {effort} effort
    </span>
  );
}

/** Model health: "not applicable" (a domain without a recommender) is neutral, never a failure. */
export function ModelHealth({ value }: { value: string }) {
  if (value === "not_applicable") {
    return (
      <span className="inline-flex items-center gap-1.5 font-sans">
        <MinusCircle className="size-3.5 shrink-0 text-muted-foreground" aria-hidden />
        <span className="text-foreground">not applicable</span>
      </span>
    );
  }
  return <Status ok={value === "ok"} label={value} />;
}
