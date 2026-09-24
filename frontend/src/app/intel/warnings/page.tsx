"use client";

import { ChevronRight } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";

import { fmtDate } from "@/components/jev/admin/ui";
import { ConfidenceBadge, SeverityBadge, WarningStatusBadge } from "@/components/jev/intel/badges";
import Link, { useIntelDomain } from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { FilterRow, IntelError, Pagination, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { qs } from "@/lib/api";
import { SEVERITIES } from "@/lib/intel";
import type { IntelWarning, Page } from "@/lib/intel-types";

const LIMIT = 50;
const STATUS_OPTIONS = [
  { value: "all", label: "All" },
  { value: "new", label: "New" },
  { value: "acknowledged", label: "Acknowledged" },
  { value: "investigating", label: "Investigating" },
  { value: "resolved", label: "Resolved" },
  { value: "dismissed", label: "Dismissed" },
];

export default function WarningsPage() {
  const [status, setStatus] = useState("new");
  const [severity, setSeverity] = useState("all");
  const [offset, setOffset] = useState(0);
  const { q: dq } = useIntelDomain();
  const { data, error, mutate, isValidating } = useSWR<Page<IntelWarning>>(
    dq(`/intel/warnings${qs({ status: status === "all" ? null : status, severity: severity === "all" ? null : severity, limit: LIMIT, offset })}`),
  );

  return (
    <div>
      <PageHeader
        eyebrow="anticipate"
        title="Early warnings"
        description="Warnings are downstream of JEV's early-warning decision: one is raised only when that decision answers Warning or Urgent action. One open warning per key; a repeat updates it. Work the queue from new to resolved, or dismiss a false alarm."
      />

      <div className="mb-5 flex flex-col gap-2.5">
        <FilterRow label="Status" value={status} onChange={(v) => { setStatus(v); setOffset(0); }} options={STATUS_OPTIONS} />
        <FilterRow label="Severity" value={severity} onChange={(v) => { setSeverity(v); setOffset(0); }} options={[{ value: "all", label: "All" }, ...SEVERITIES.map((s) => ({ value: s, label: s }))]} />
      </div>

      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} />
      ) : !data ? (
        <RowsSkeleton rows={8} />
      ) : data.items.length === 0 ? (
        <EmptyState
          title={status === "new" ? "Nothing new to triage" : "No warnings here"}
          body={status === "all" && severity === "all" ? "No warnings have been raised yet. They appear after a pipeline run finds a risk, anomaly or decision that crosses its rule." : "No warnings match these filters."}
        />
      ) : (
        <>
          <p className="num mb-2 text-xs text-muted-foreground">{data.total.toLocaleString()} warnings</p>
          <ul className={`divide-y hairline rounded-lg border bg-card transition-opacity ${isValidating ? "opacity-70" : ""}`}>
            {data.items.map((w) => (
              <li key={w.id}>
                <Link href={`/intel/warnings/${w.id}`} className="grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-2 px-4 py-3.5 hover:bg-accent/40 md:grid-cols-[110px_minmax(0,1fr)_130px_110px_150px_auto]">
                  <span className="order-2 md:order-none"><SeverityBadge severity={w.severity} /></span>
                  <span className="order-1 col-span-2 min-w-0 md:order-none md:col-span-1">
                    <span className="block">{w.title}</span>
                    <span className="block truncate font-mono text-xs text-muted-foreground">{w.key}{w.early_warning_level ? ` · level ${w.early_warning_level}` : ""}{w.decision_id ? ` · from ${w.decision_id}` : ""}</span>
                  </span>
                  <span className="order-3 md:order-none"><WarningStatusBadge status={w.status} /></span>
                  <span className="order-4 md:order-none"><ConfidenceBadge value={w.confidence} kind={w.confidence_kind} /></span>
                  <span className="order-5 col-span-2 font-mono text-xs text-muted-foreground md:order-none md:col-span-1">
                    ×{w.occurrences} · last {fmtDate(w.last_seen_at).slice(0, 10)}
                  </span>
                  <ChevronRight className="hidden size-4 text-muted-foreground md:block" aria-hidden />
                </Link>
              </li>
            ))}
          </ul>
          <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
        </>
      )}
    </div>
  );
}
