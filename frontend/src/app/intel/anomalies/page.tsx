"use client";

import { EyeOff } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";

import { SeverityBadge } from "@/components/jev/intel/badges";
import { EvidenceDisclosure } from "@/components/jev/intel/evidence-list";
import { PageHeader } from "@/components/jev/intel/page-header";
import { FilterRow, IntelError, Pagination, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Switch } from "@/components/ui/switch";
import { qs } from "@/lib/api";
import { fmtValue, fmtMonth, fmtSeriesValue, fmtSigned, humanize, SEVERITIES, SEVERITY_RANK, seriesHref, seriesLabel } from "@/lib/intel";
import type { Anomaly, Page } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

const LIMIT = 100;

function Suppressed({ a }: { a: Anomaly }) {
  if (!a.suppressed) return null;
  return (
    <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
      <EyeOff className="size-3.5" aria-hidden />
      Suppressed: {a.suppression_reason ?? "no reason recorded"}
    </span>
  );
}

function SeriesAnomalies({ items }: { items: Anomaly[] }) {
  if (!items.length) return <EmptyState title="No series anomalies" body="No month deviated enough from its robust baseline." />;
  return (
    <div className="overflow-x-auto rounded-lg border bg-card">
      <table className="w-full min-w-[760px] text-sm">
        <caption className="sr-only">Series anomalies</caption>
        <thead>
          <tr className="border-b hairline text-left text-xs text-muted-foreground">
            <th className="px-4 py-2.5 font-normal">Month · series</th>
            <th className="px-2 py-2.5 font-normal">Kind</th>
            <th className="px-2 py-2.5 text-right font-normal">Value</th>
            <th className="px-2 py-2.5 text-right font-normal">Baseline</th>
            <th className="px-2 py-2.5 text-right font-normal">Deviation</th>
            <th className="px-2 py-2.5 text-right font-normal">Robust z</th>
            <th className="px-4 py-2.5 font-normal">Severity</th>
          </tr>
        </thead>
        <tbody>
          {items.map((a) => {
            const metric = a.series_id?.split(":")[0] ?? "";
            return (
              <tr key={a.id} id={a.id} className={cn("scroll-mt-24 border-b hairline align-top last:border-0", a.suppressed && "bg-muted/50 text-muted-foreground")}>
                <td className="px-4 py-2.5">
                  <span className="num text-xs">{fmtMonth(a.detected_at)}</span>
                  {a.series_id ? (
                    <Link href={seriesHref(a.series_id)} className="block hover:underline">{seriesLabel(a.series_id)}</Link>
                  ) : (
                    <span className="block">{a.entity}</span>
                  )}
                  <Suppressed a={a} />
                  <EvidenceDisclosure items={a.evidence} className="mt-1" />
                </td>
                <td className="px-2 py-2.5 text-xs">{humanize(a.kind.replace("series_", ""))}<span className="block font-mono text-muted-foreground">{humanize(a.method)}</span></td>
                <td className="num px-2 py-2.5 text-right">{fmtSeriesValue(a.value, metric)}</td>
                <td className="num px-2 py-2.5 text-right text-ink-2">{fmtSeriesValue(a.baseline, metric)}</td>
                <td className="num px-2 py-2.5 text-right">{a.deviation === null ? "—" : metric === "share" ? fmtSigned(a.deviation * 100, 1, " pt") : fmtSigned(a.deviation, metric === "rating" ? 2 : 0)}</td>
                <td className="num px-2 py-2.5 text-right">{a.score.toFixed(1)}</td>
                <td className="px-4 py-2.5"><SeverityBadge severity={a.severity} /></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function RaterAnomalies({ items }: { items: Anomaly[] }) {
  if (!items.length) return <EmptyState title="No unusual raters" body="No account's rating behaviour stood out in the latest run." />;
  const keys = Array.from(new Set(items.flatMap((a) => Object.keys(a.features ?? {}))));
  return (
    <div className="overflow-x-auto rounded-lg border bg-card">
      <table className="w-full text-sm" style={{ minWidth: 420 + keys.length * 96 }}>
        <caption className="sr-only">Rater-behaviour anomalies with their behavioural features</caption>
        <thead>
          <tr className="border-b hairline text-left text-xs text-muted-foreground">
            <th className="px-4 py-2.5 font-normal">Rater</th>
            <th className="px-2 py-2.5 text-right font-normal">Score (pctl)</th>
            <th className="px-2 py-2.5 font-normal">Severity</th>
            {keys.map((k) => <th key={k} className="px-2 py-2.5 text-right font-normal">{humanize(k)}</th>)}
          </tr>
        </thead>
        <tbody>
          {items.map((a) => (
            <tr key={a.id} id={a.id} className={cn("scroll-mt-24 border-b hairline align-top last:border-0", a.suppressed && "bg-muted/50 text-muted-foreground")}>
              <td className="px-4 py-2.5">
                <span className="num">{a.entity_type === "user" ? `user ${a.entity}` : a.entity}</span>
                <span className="block font-mono text-xs text-muted-foreground">{humanize(a.method)} · {a.detected_at.slice(0, 10)}</span>
                <Suppressed a={a} />
                <EvidenceDisclosure items={a.evidence} className="mt-1" />
              </td>
              <td className="num px-2 py-2.5 text-right">{a.score.toFixed(a.score <= 1 ? 3 : 1)}</td>
              <td className="px-2 py-2.5"><SeverityBadge severity={a.severity} /></td>
              {keys.map((k) => <td key={k} className="num px-2 py-2.5 text-right text-ink-2">{fmtValue(a.features?.[k], k)}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AnomaliesPage() {
  const [view, setView] = useState<"series" | "rater">("series");
  const [severity, setSeverity] = useState("all");
  const [showSuppressed, setShowSuppressed] = useState(true);
  const [offset, setOffset] = useState(0);
  const { data, error, mutate } = useSWR<Page<Anomaly>>(`/intel/anomalies${qs({ limit: LIMIT, offset })}`);

  const all = data?.items ?? [];
  const filtered = all
    .filter((a) => (view === "rater" ? a.kind === "rater_behaviour" : a.kind !== "rater_behaviour"))
    .filter((a) => severity === "all" || a.severity === severity)
    .filter((a) => showSuppressed || !a.suppressed)
    .sort((a, b) => Number(a.suppressed) - Number(b.suppressed) || SEVERITY_RANK[b.severity] - SEVERITY_RANK[a.severity] || b.score - a.score);
  const suppressedCount = all.filter((a) => a.suppressed).length;

  return (
    <div>
      <PageHeader
        eyebrow="detect"
        title="Anomalies"
        description="Months that broke from their robust baseline (median ± MAD), and raters whose behaviour an isolation forest ranks as unusual. Suppressed items stay visible with the reason."
        asOf={data?.as_of}
        runId={data?.run_id}
      />

      <div className="mb-5 flex flex-col gap-2.5">
        <FilterRow label="View" value={view} onChange={(v) => setView(v as "series" | "rater")} options={[{ value: "series", label: "Series" }, { value: "rater", label: "Rater behaviour" }]} />
        <FilterRow label="Severity" value={severity} onChange={setSeverity} options={[{ value: "all", label: "All" }, ...SEVERITIES.map((s) => ({ value: s, label: s }))]} />
        <label className="flex items-center gap-2 text-xs text-muted-foreground">
          <Switch checked={showSuppressed} onCheckedChange={setShowSuppressed} aria-label="Show suppressed anomalies" />
          Show suppressed{data ? ` (${suppressedCount} on this page)` : ""}
        </label>
      </div>

      {error ? (
        <IntelError error={error} retry={() => mutate()} />
      ) : !data ? (
        <RowsSkeleton rows={8} />
      ) : (
        <>
          {view === "series" ? <SeriesAnomalies items={filtered} /> : <RaterAnomalies items={filtered} />}
          <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
        </>
      )}
    </div>
  );
}
