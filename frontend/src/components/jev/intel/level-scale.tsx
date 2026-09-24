"use client";

import { CheckCircle2, Eye, OctagonAlert, TriangleAlert } from "lucide-react";

import { SpecRows } from "@/components/jev/admin/ui";
import Link from "@/components/jev/intel/domain-context";
import { ewlStateGroups, LEVEL_LABEL, LEVEL_MEANING, levelScale } from "@/lib/decisions";
import { fmtValue, humanize, refHref } from "@/lib/intel";
import type { Decision, EarlyWarningLevel } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

/*
 * The early-warning level (docs/platform.md §4) as a four-step ordinal scale. Each step carries an
 * icon and a word; the status colour only tints the icon, so the level reads in greyscale too.
 */

export const LEVEL_ICON: Record<EarlyWarningLevel, { Icon: typeof Eye; color: string }> = {
  NO_ACTION: { Icon: CheckCircle2, color: "var(--status-good)" },
  MONITOR: { Icon: Eye, color: "var(--muted-foreground)" },
  WARNING: { Icon: TriangleAlert, color: "var(--status-serious)" },
  URGENT_ACTION: { Icon: OctagonAlert, color: "var(--status-critical)" },
};

/** The level as a chip: icon + word, tinted icon only. */
export function LevelBadge({ level, className }: { level: EarlyWarningLevel; className?: string }) {
  const m = LEVEL_ICON[level];
  return (
    <span className={cn("inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded border hairline px-1.5 py-0.5 text-xs", className)}>
      <m.Icon className="size-3.5" style={{ color: m.color }} aria-hidden />
      <span className="text-muted-foreground">level</span> {LEVEL_LABEL[level]}
    </span>
  );
}

/** Compact: one row of four cells, the chosen level outlined in the accent. */
export function LevelScale({
  decision,
  compact = false,
  className,
}: {
  decision: Pick<Decision, "answer" | "option_scores" | "abstained">;
  compact?: boolean;
  className?: string;
}) {
  const steps = levelScale(decision);
  const chosen = steps.find((s) => s.chosen);
  return (
    <div className={cn("min-w-0", className)}>
      <ol
        className="grid grid-cols-4 gap-[2px] overflow-hidden rounded border hairline bg-[var(--rule)]"
        aria-label={chosen ? `Early-warning level ${chosen.step} of 4: ${chosen.label}` : "Early-warning level: abstained"}
      >
        {steps.map((s) => {
          const m = LEVEL_ICON[s.level];
          return (
            <li
              key={s.level}
              aria-current={s.chosen ? "step" : undefined}
              className={cn(
                "relative flex min-w-0 flex-col gap-0.5 bg-card px-1.5 py-1.5 sm:px-2",
                s.chosen && "outline outline-2 -outline-offset-2 outline-primary",
                !s.reached && !s.chosen && "text-muted-foreground",
              )}
            >
              <span className="flex min-w-0 items-center gap-1">
                <m.Icon className="size-3.5 shrink-0" style={{ color: s.reached ? m.color : undefined }} aria-hidden />
                <span className={cn("truncate text-xs", s.chosen && "font-semibold text-foreground")}>{compact ? (s.level === "NO_ACTION" ? "None" : s.level === "URGENT_ACTION" ? "Urgent" : s.label) : s.label}</span>
              </span>
              {!compact && (
                <span className="num text-[11px] text-muted-foreground">
                  {s.score === null ? "score —" : `score ${fmtValue(s.score)}`}
                </span>
              )}
              {/* the fill below the cell: reached steps carry a solid accent rule */}
              <span className={cn("absolute inset-x-0 bottom-0 h-[3px]", s.reached ? "bg-primary" : "bg-transparent")} aria-hidden />
            </li>
          );
        })}
      </ol>
      {!compact && (
        <p className="mt-2 text-xs text-muted-foreground">
          {chosen ? `${chosen.label}: ${LEVEL_MEANING[chosen.level]}` : "JEV abstained, so no level was chosen and no warning follows from this decision."} A warning is raised only at Warning or Urgent action.
        </p>
      )}
    </div>
  );
}

/** The early-warning state, grouped signal → trend → anomaly → forecast → risk. */
export function StateSnapshot({ state }: { state: Record<string, unknown> }) {
  const { groups, other, components } = ewlStateGroups(state);
  return (
    <div className="space-y-4">
      <ol className="grid grid-cols-1 gap-[2px] overflow-hidden rounded border hairline bg-[var(--rule)] sm:grid-cols-5">
        {groups.map((g, i) => (
          <li key={g.stage} className="min-w-0 bg-card px-3 py-2.5">
            <p className="eyebrow">{String(i + 1).padStart(2, "0")} · {g.label}</p>
            {g.observed ? (
              <>
                <dl className="mt-1.5 space-y-0.5 text-xs">
                  {g.rows.map((r) => (
                    <div key={r.key} className="flex items-baseline justify-between gap-2">
                      <dt className="min-w-0 truncate text-muted-foreground" title={r.label}>{r.label}</dt>
                      <dd className="num shrink-0 text-foreground">{r.value}</dd>
                    </div>
                  ))}
                </dl>
                {g.summary && <p className="mt-1.5 line-clamp-3 text-xs text-ink-2" title={g.summary}>{g.summary}</p>}
              </>
            ) : g.skippedReason ? (
              <p className="mt-1.5 text-xs text-muted-foreground">Skipped: {g.skippedReason}</p>
            ) : (
              <p className="mt-1.5 text-xs text-muted-foreground">Nothing observed for this situation ({g.what}).</p>
            )}
          </li>
        ))}
      </ol>
      {components.length > 0 && (
        <div>
          <p className="eyebrow mb-1">Point components · highest first</p>
          <ul className="divide-y hairline border-y hairline text-sm">
            {components.map((c, i) => {
              const href = refHref(c.ref);
              return (
                <li key={`${c.ref}-${i}`} className="grid grid-cols-[88px_minmax(0,1fr)_56px] items-baseline gap-x-3 py-1.5">
                  <span className="eyebrow">{humanize(c.stage)}</span>
                  <span className="min-w-0">
                    <span className="block">{c.title ?? c.detail ?? "—"}</span>
                    {c.title && c.detail && <span className="block text-xs text-muted-foreground">{c.detail}</span>}
                    {c.ref && (href ? <Link href={href} className="font-mono text-xs text-primary hover:underline">{c.ref}</Link> : <span className="font-mono text-xs text-muted-foreground">{c.ref}</span>)}
                  </span>
                  <span className="num text-right text-ink-2">{c.points === null ? "—" : fmtValue(c.points)}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {other.length > 0 && (
        <div>
          <p className="eyebrow mb-1">Other inputs</p>
          <SpecRows rows={other.map((r) => ({ label: r.label, value: r.value }))} />
        </div>
      )}
    </div>
  );
}
