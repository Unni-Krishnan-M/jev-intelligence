"use client";

/*
 * Evidence explorer. Structure borrowed from log explorers (Grafana Explore's line filter above
 * the results, expandable rows, and "filter on this field" from a row's detail): one search field
 * and two filter rows above a paged list; each row's owner type is itself a filter, and every row
 * links to the object and series it came from. Visual style stays JEV's.
 */

import { ArrowUpRight, Search } from "lucide-react";
import { useEffect, useState } from "react";
import useSWR from "swr";

import Link, { useIntelDomain } from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { FilterRow, IntelError, Pagination, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Input } from "@/components/ui/input";
import { qs } from "@/lib/api";
import { fmtValue, humanize, ownerHref, refHref } from "@/lib/intel";
import type { EvidenceOwnerType, EvidenceRow, Page } from "@/lib/intel-types";

const LIMIT = 50;
const OWNER_TYPES: ("all" | EvidenceOwnerType)[] = ["all", "signal", "trend", "anomaly", "forecast", "risk", "decision", "warning", "action"];
const KINDS = ["all", "metric", "series", "record", "test", "model"];

function Row({ e, onOwnerType }: { e: EvidenceRow; onOwnerType: (t: EvidenceOwnerType) => void }) {
  const owner = ownerHref(e.owner_type, e.owner_id);
  const ref = refHref(e.ref);
  return (
    <li className="grid grid-cols-1 gap-x-5 gap-y-1 px-4 py-3 text-sm sm:grid-cols-[72px_minmax(0,1fr)_auto]">
      <span className="eyebrow pt-0.5">{e.kind}</span>
      <div className="min-w-0">
        <p className="text-foreground">{e.label}</p>
        {e.detail && <p className="mt-0.5 text-xs text-muted-foreground">{e.detail}</p>}
        <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs">
          <button
            type="button"
            onClick={() => onOwnerType(e.owner_type)}
            className="rounded border hairline px-1.5 py-px text-muted-foreground hover:text-foreground"
            title={`Show only ${e.owner_type} evidence`}
          >
            {e.owner_type}
          </button>
          {owner ? (
            <Link href={owner} className="min-w-0 break-words text-primary hover:underline">{e.owner_title || e.owner_id}</Link>
          ) : (
            <span className="min-w-0 break-words">{e.owner_title || e.owner_id}</span>
          )}
          {e.ref && (
            ref ? (
              <Link href={ref} className="inline-flex items-center gap-0.5 break-all font-mono text-primary hover:underline">
                {e.ref}
                <ArrowUpRight className="size-3" aria-hidden />
              </Link>
            ) : (
              <span className="break-all font-mono text-muted-foreground">{e.ref}</span>
            )
          )}
        </p>
      </div>
      {e.value !== null && e.value !== undefined && <span className="num text-ink-2 sm:pt-0.5 sm:text-right">{fmtValue(e.value)}</span>}
    </li>
  );
}

export default function EvidencePage() {
  const [ownerType, setOwnerType] = useState<string>("all");
  const [kind, setKind] = useState("all");
  const [text, setText] = useState("");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);

  // search as you type, one request per pause
  useEffect(() => {
    const t = setTimeout(() => {
      setQ(text.trim());
      setOffset(0);
    }, 300);
    return () => clearTimeout(t);
  }, [text]);

  const { q: dq } = useIntelDomain();
  const key = dq(`/intel/evidence${qs({ owner_type: ownerType === "all" ? null : ownerType, kind: kind === "all" ? null : kind, q, limit: LIMIT, offset })}`);
  const { data, error, mutate, isValidating } = useSWR<Page<EvidenceRow>>(key, { keepPreviousData: true });
  const filtered = ownerType !== "all" || kind !== "all" || Boolean(q);

  return (
    <div>
      <PageHeader
        eyebrow="trace"
        title="Evidence"
        description="Every piece of evidence the pipeline attached to a signal, trend, anomaly, forecast, risk, decision, warning or action, searchable in one place. Each row links back to what it supports."
        asOf={data?.as_of}
        runId={data?.run_id}
      />

      <div className="mb-5 flex flex-col gap-2.5">
        <form role="search" onSubmit={(e) => { e.preventDefault(); setQ(text.trim()); setOffset(0); }} className="relative max-w-md">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input
            type="search"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Search label, detail or reference"
            aria-label="Search evidence"
            className="h-9 pl-8"
          />
        </form>
        <FilterRow label="Owner" value={ownerType} onChange={(v) => { setOwnerType(v); setOffset(0); }} options={OWNER_TYPES.map((k) => ({ value: k, label: k === "all" ? "All" : k }))} />
        <FilterRow label="Kind" value={kind} onChange={(v) => { setKind(v); setOffset(0); }} options={KINDS.map((k) => ({ value: k, label: k === "all" ? "All" : humanize(k) }))} />
      </div>

      {error ? (
        <IntelError error={error} retry={() => mutate()} what="the evidence explorer (GET /intel/evidence)" />
      ) : !data ? (
        <RowsSkeleton rows={8} />
      ) : data.items.length === 0 ? (
        <EmptyState title="No evidence" body={filtered ? "Nothing matches this search and these filters." : "The latest run recorded no evidence."} />
      ) : (
        <>
          <p className="num mb-2 text-xs text-muted-foreground" aria-live="polite">{data.total.toLocaleString()} rows{q ? ` matching “${q}”` : ""}</p>
          <ul className={`divide-y hairline rounded-lg border bg-card transition-opacity ${isValidating ? "opacity-70" : ""}`}>
            {data.items.map((e) => <Row key={e.id} e={e} onOwnerType={(t) => { setOwnerType(t); setOffset(0); }} />)}
          </ul>
          <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
        </>
      )}
    </div>
  );
}
