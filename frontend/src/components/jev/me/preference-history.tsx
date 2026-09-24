"use client";

import { ChartLegend, SLOT_COLORS, TableView } from "@/components/jev/intel/charts";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fmtDay, humanize } from "@/lib/intel";
import type { PreferenceHistory } from "@/lib/intel-types";
import { fmtShare, OTHER, shareSegments, topCategories } from "@/lib/me-intel";
import { cn } from "@/lib/utils";

/*
 * Part-to-whole per window: one horizontal 100 % bar per window, stacked in a fixed category
 * order computed once for the whole history (so a genre keeps its colour in every window). Five
 * categorical slots at most, the tail folded into a neutral "Other". 2px surface gaps between
 * segments, 4px rounded bar ends, a legend and a table view, so no value depends on colour or hover.
 */

const OTHER_COLOR = "var(--muted-foreground)";

function windowLabel(label: string): string {
  if (label === "historical") return "Historical";
  if (label === "recent") return "Recent";
  return humanize(label);
}

export function PreferenceHistoryChart({ history }: { history: PreferenceHistory }) {
  const { shown, folded } = topCategories(history, 5);
  const color = (slot: number | null) => (slot === null ? OTHER_COLOR : SLOT_COLORS[slot]);
  const rows = history.windows.map((w) => {
    const segs = shareSegments(w.shares, shown);
    const total = segs.reduce((s, x) => s + x.share, 0);
    return { w, segs, scale: Math.max(1, total) };
  });

  return (
    <div>
      <ul className="space-y-4">
        {rows.map(({ w, segs, scale }) => (
          <li key={`${w.label}-${w.start}`} className="grid grid-cols-1 gap-x-4 gap-y-1.5 sm:grid-cols-[150px_minmax(0,1fr)] sm:items-center">
            <div className="min-w-0">
              <p className="text-sm">{windowLabel(w.label)}</p>
              <p className="num text-[11px] text-muted-foreground">
                {fmtDay(w.start)} – {fmtDay(w.end)} · n {w.n.toLocaleString()}
              </p>
            </div>
            {w.n === 0 || !segs.length ? (
              <p className="text-xs text-muted-foreground">No events in this window.</p>
            ) : (
              <div className="flex h-[18px] w-full gap-[2px]" role="img" aria-label={`${windowLabel(w.label)}: ${segs.map((s) => `${s.category} ${fmtShare(s.share)}`).join(", ")}`}>
                {segs.map((s, i) => (
                  <Tooltip key={s.category}>
                    <TooltipTrigger asChild>
                      <span
                        className={cn("h-full min-w-[2px]", i === 0 && "rounded-l-[4px]", i === segs.length - 1 && "rounded-r-[4px]")}
                        style={{ width: `${(s.share / scale) * 100}%`, background: color(s.slot) }}
                      />
                    </TooltipTrigger>
                    <TooltipContent>
                      <span className="num">{s.category} · {fmtShare(s.share)}</span>
                    </TooltipContent>
                  </Tooltip>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>
      <ChartLegend
        className="mt-5"
        items={[
          ...shown.map((c, i) => ({ label: c, color: SLOT_COLORS[i], kind: "dot" as const })),
          ...(folded.length ? [{ label: `${OTHER} (${folded.length})`, color: OTHER_COLOR, kind: "dot" as const }] : []),
        ]}
      />
      <TableView
        caption="Genre share per time window"
        head={["window", ...shown, ...(folded.length ? [OTHER] : []), "n"]}
        rows={rows.map(({ w, segs }) => [
          windowLabel(w.label),
          ...shown.map((c) => fmtShare(w.shares[c] ?? 0)),
          ...(folded.length ? [fmtShare(segs.find((s) => s.category === OTHER)?.share ?? 0)] : []),
          w.n.toLocaleString(),
        ])}
      />
    </div>
  );
}
