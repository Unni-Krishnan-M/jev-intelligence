"use client";

import useSWR from "swr";

import { RunTimeline, TableView } from "@/components/jev/intel/charts";
import { useIntelDomain } from "@/components/jev/intel/domain-context";
import { qs } from "@/lib/api";
import { fmtDay, fmtValue, historyField, humanize, isNotDeployed } from "@/lib/intel";
import type { HistoryEntity, HistoryResponse } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

/**
 * "Across runs": how one object read in each pipeline run (GET /intel/history/{entity}?key=).
 * Quiet (the default) renders nothing until there are at least two readings, and nothing when the
 * endpoint is missing (an older API) or fails: it is context under a row, never the row itself.
 * With quiet={false} (inside a disclosure the operator opened) it says why there is no line.
 */
export function HistoryTimeline({
  entity,
  historyKey,
  labels,
  format = (v: number) => fmtValue(v),
  quiet = true,
  className,
}: {
  entity: HistoryEntity;
  historyKey: string;
  /** what each field means for this entity ({score: "risk score", value: "exposure"}) */
  labels?: { score?: string; value?: string };
  format?: (v: number) => string;
  quiet?: boolean;
  className?: string;
}) {
  const { q: dq } = useIntelDomain();
  const { data, error } = useSWR<HistoryResponse>(dq(`/intel/history/${entity}${qs({ key: historyKey })}`), { shouldRetryOnError: false, revalidateOnFocus: false });
  const items = data?.items ?? [];
  const field = historyField(items);
  if (items.length < 2 || !field) {
    if (quiet) return null;
    const note = error
      ? isNotDeployed(error) ? "Run history is not available on this API version." : "Run history could not be loaded."
      : !data ? "Loading run history…" : "Fewer than two runs have seen this yet, so there is no line to draw.";
    return <p className={cn("text-xs text-muted-foreground", className)} aria-live="polite">{note}</p>;
  }
  const points = items.map((p, i) => {
    const v = p[field] as number;
    return {
      key: `${p.run_id}-${i}`,
      v,
      title: `run ${p.run_id.slice(0, 8)} · as of ${fmtDay(p.as_of)} · ${format(v)}${p.level ? ` · ${p.level}` : ""}${p.direction ? ` · ${p.direction}` : ""}`,
    };
  });
  const what = labels?.[field] ?? humanize(field);
  return (
    <div className={cn("rounded border hairline px-3 py-2.5", className)}>
      <p className="eyebrow mb-1">Across runs · {what} · {items.length} runs</p>
      <RunTimeline points={points} format={format} />
      <TableView
        caption={`${what} across runs for ${historyKey}`}
        head={["as of", "run", what, "level", "direction"]}
        rows={items.map((p) => [fmtDay(p.as_of), p.run_id.slice(0, 8), p[field] === null ? "—" : format(p[field] as number), p.level ?? "—", p.direction ?? "—"])}
      />
    </div>
  );
}
