"use client";

import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export interface IntervalRow {
  diff: number | null;
  lo: number | null;
  hi: number | null;
}

/** One shared x-domain for a stack of interval rows, always containing 0 and the margin. */
export function intervalDomain(rows: IntervalRow[], margin?: number | null): [number, number] {
  const xs = [0];
  for (const r of rows) for (const v of [r.lo, r.hi, r.diff]) if (typeof v === "number" && Number.isFinite(v)) xs.push(v);
  if (typeof margin === "number") xs.push(-margin);
  let lo = Math.min(...xs);
  let hi = Math.max(...xs);
  if (lo === hi) {
    lo -= 1;
    hi += 1;
  }
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

/**
 * A confidence interval on one axis: the zero line (no difference), an optional non-inferiority
 * margin at −margin (dashed, the pass threshold), the CI as a 2 px rule and the point estimate as a
 * dot. Several rows share `domain` so they can be compared. The numbers are always printed beside
 * it by the caller; the tooltip repeats them for pointer users.
 */
export function IntervalPlot({
  row,
  domain,
  margin,
  format,
  label,
  highlight = false,
  className,
}: {
  row: IntervalRow;
  domain: [number, number];
  margin?: number | null;
  format: (v: number) => string;
  label: string;
  highlight?: boolean;
  className?: string;
}) {
  const [a, b] = domain;
  const x = (v: number) => `${Math.max(0, Math.min(100, ((v - a) / (b - a)) * 100))}%`;
  const has = row.lo !== null && row.hi !== null;
  const text = has
    ? `${label}: ${row.diff !== null ? format(row.diff) : "—"}, CI ${format(row.lo as number)} to ${format(row.hi as number)}${typeof margin === "number" ? `; must stay above −${format(margin).replace(/^[+−-]/, "")}` : ""}`
    : `${label}: no interval (too little data)`;
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span tabIndex={0} role="img" aria-label={text} className={cn("relative block h-6 min-w-24 cursor-default rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring/50", className)}>
          {/* baseline */}
          <span className="absolute inset-x-0 top-1/2 h-px bg-rule" aria-hidden />
          {/* zero: no difference */}
          <span className="absolute inset-y-0.5 w-px bg-muted-foreground/60" style={{ left: x(0) }} aria-hidden />
          {/* the non-inferiority threshold */}
          {typeof margin === "number" && (
            <span className="absolute inset-y-0 border-l border-dashed border-primary" style={{ left: x(-margin) }} aria-hidden />
          )}
          {has && (
            <span
              className={cn("absolute top-1/2 h-0.5 -translate-y-1/2 rounded-full", highlight ? "bg-foreground" : "bg-ink-2")}
              style={{ left: x(row.lo as number), width: `calc(${x(row.hi as number)} - ${x(row.lo as number)})` }}
              aria-hidden
            />
          )}
          {row.diff !== null && (
            <span
              className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary ring-2 ring-card"
              style={{ left: x(row.diff) }}
              aria-hidden
            />
          )}
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-72">{text}</TooltipContent>
    </Tooltip>
  );
}

/** The key under a stack of interval rows. */
export function IntervalKey({ margin, level }: { margin?: boolean; level?: number | null }) {
  return (
    <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      <span className="inline-flex items-center gap-1.5"><span className="size-2.5 rounded-full bg-primary" aria-hidden /> point estimate</span>
      <span className="inline-flex items-center gap-1.5"><span className="h-0.5 w-5 rounded-full bg-ink-2" aria-hidden /> {level ? `${Math.round(level * 100)} % ` : ""}confidence interval</span>
      <span className="inline-flex items-center gap-1.5"><span className="h-3 w-px bg-muted-foreground/60" aria-hidden /> no difference</span>
      {margin && <span className="inline-flex items-center gap-1.5"><span className="h-3 border-l border-dashed border-primary" aria-hidden /> non-inferiority margin</span>}
    </p>
  );
}
