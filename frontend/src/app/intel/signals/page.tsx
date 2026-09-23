"use client";

import { useState } from "react";
import useSWR from "swr";

import { fmtDate } from "@/components/jev/admin/ui";
import { DirectionIcon, FreshnessBadge, Meter } from "@/components/jev/intel/badges";
import { EvidenceDisclosure } from "@/components/jev/intel/evidence-list";
import { HistoryTimeline } from "@/components/jev/intel/history-timeline";
import { PageHeader } from "@/components/jev/intel/page-header";
import { FilterRow, IntelError, Pagination, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { qs } from "@/lib/api";
import { fmtValue, humanize } from "@/lib/intel";
import type { Page, Signal } from "@/lib/intel-types";

const LIMIT = 50;
const KINDS = ["all", "trend", "anomaly", "change_point", "forecast", "quality", "live", "model"];
const ENTITY_TYPES = ["all", "genre", "platform", "user", "model", "source"];

/** Across-runs history for one signal, fetched only once the operator opens it. */
function RunsDisclosure({ dedupKey }: { dedupKey: string }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="group min-w-0 open:basis-full" onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
        <span className="transition-transform group-open:rotate-90" aria-hidden>›</span> Across runs
      </summary>
      {open && <HistoryTimeline entity="signals" historyKey={dedupKey} labels={{ score: "strength", value: "value" }} quiet={false} className="mt-2" />}
    </details>
  );
}

export default function SignalsPage() {
  const [kind, setKind] = useState("all");
  const [entityType, setEntityType] = useState("all");
  const [offset, setOffset] = useState(0);
  const key = `/intel/signals${qs({ kind: kind === "all" ? null : kind, entity_type: entityType === "all" ? null : entityType, limit: LIMIT, offset })}`;
  const { data, error, mutate, isValidating } = useSWR<Page<Signal>>(key);

  return (
    <div>
      <PageHeader
        eyebrow="detect"
        title="Signals"
        description="Every observation the pipeline raised: trends, anomalies, change points, forecasts, quality and live-app readings. Strength is a 0–1 score defined per kind."
        asOf={data?.as_of}
        runId={data?.run_id}
      />

      <div className="mb-5 flex flex-col gap-2.5">
        <FilterRow label="Kind" value={kind} onChange={(v) => { setKind(v); setOffset(0); }} options={KINDS.map((k) => ({ value: k, label: k === "all" ? "All" : humanize(k) }))} />
        <FilterRow label="Entity" value={entityType} onChange={(v) => { setEntityType(v); setOffset(0); }} options={ENTITY_TYPES.map((k) => ({ value: k, label: k === "all" ? "All" : k }))} />
      </div>

      {error ? (
        <IntelError error={error} retry={() => mutate()} />
      ) : !data ? (
        <RowsSkeleton rows={8} />
      ) : data.items.length === 0 ? (
        <EmptyState title="No signals" body={kind === "all" && entityType === "all" ? "The latest run raised no signals." : "Nothing matches these filters in the latest run."} />
      ) : (
        <>
          <p className="num mb-2 text-xs text-muted-foreground">{data.total.toLocaleString()} signals</p>
          <ul className={`divide-y hairline rounded-lg border bg-card transition-opacity ${isValidating ? "opacity-70" : ""}`}>
            {data.items.map((s) => (
              <li key={s.id} id={s.id} className="scroll-mt-24 px-4 py-3.5">
                <div className="grid grid-cols-1 gap-x-6 gap-y-2 md:grid-cols-[minmax(0,1fr)_160px_150px]">
                  <div className="min-w-0">
                    <p className="eyebrow">{humanize(s.kind)} · {s.entity_type} · {s.source}</p>
                    <p className="mt-1 flex items-start gap-2">
                      <DirectionIcon direction={s.direction} className="mt-0.5" />
                      <span className="min-w-0">
                        <span className="block">{s.title}</span>
                        <span className="block font-mono text-xs text-muted-foreground">
                          {s.entity}
                          {s.value !== null && <> · {fmtValue(s.value)}{s.unit ? ` ${s.unit}` : ""}</>}
                          {s.window && <> · window {s.window}</>}
                        </span>
                      </span>
                    </p>
                  </div>
                  <div className="self-center">
                    <p className="eyebrow mb-1 md:hidden">Strength</p>
                    <Meter value={s.strength} label={`Strength ${s.strength.toFixed(2)} of 1`} />
                  </div>
                  <div className="flex flex-wrap items-center gap-2 self-center md:justify-end">
                    <FreshnessBadge kind={s.source} days={s.freshness_days} />
                    <span className="num text-xs text-muted-foreground">{fmtDate(s.observed_at).slice(0, 10)}</span>
                  </div>
                </div>
                <div className="mt-2 flex flex-wrap gap-x-5 gap-y-2">
                  <EvidenceDisclosure items={s.evidence} />
                  <RunsDisclosure dedupKey={s.dedup_key} />
                </div>
              </li>
            ))}
          </ul>
          <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
        </>
      )}
    </div>
  );
}
