"use client";

import useSWR from "swr";

import { Panel } from "@/components/jev/admin/ui";
import { ConfidenceBadge, Meter, SeverityBadge } from "@/components/jev/intel/badges";
import { RiskMatrix } from "@/components/jev/intel/charts";
import { EvidenceList } from "@/components/jev/intel/evidence-list";
import { HistoryTimeline } from "@/components/jev/intel/history-timeline";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Skeleton } from "@/components/ui/skeleton";
import { qs } from "@/lib/api";
import { fmtValue, humanize, riskHistoryKey } from "@/lib/intel";
import type { Page, Risk } from "@/lib/intel-types";

function Factors({ r }: { r: Risk }) {
  if (!r.factors.length) return <p className="text-sm text-muted-foreground">No factors recorded.</p>;
  const max = Math.max(...r.factors.map((f) => Math.abs(f.contribution)), 1e-9);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-sm">
        <caption className="sr-only">Contributing factors for {r.title}</caption>
        <thead>
          <tr className="border-b hairline text-left text-xs text-muted-foreground">
            <th className="py-2 pr-2 font-normal">Factor</th>
            <th className="px-2 py-2 text-right font-normal">Value</th>
            <th className="px-2 py-2 text-right font-normal">Weight</th>
            <th className="px-2 py-2 font-normal">Contribution</th>
          </tr>
        </thead>
        <tbody>
          {[...r.factors].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).map((f) => (
            <tr key={f.name} className="border-b hairline align-top last:border-0">
              <td className="py-2 pr-2">
                {humanize(f.name)}
                {f.detail && <span className="block text-xs text-muted-foreground">{f.detail}</span>}
              </td>
              <td className="num px-2 py-2 text-right">{fmtValue(f.value)}</td>
              <td className="num px-2 py-2 text-right text-ink-2">{f.weight.toFixed(2)}</td>
              <td className="w-48 px-2 py-2">
                <span className="flex items-center gap-2">
                  <span className="h-2 flex-1">
                    <span className="block h-full rounded-r-[4px] bg-sig-content" style={{ width: `${(Math.abs(f.contribution) / max) * 100}%` }} />
                  </span>
                  <span className="num w-12 text-right text-xs">{f.contribution.toFixed(3)}</span>
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function RisksPage() {
  const { data, error, mutate } = useSWR<Page<Risk>>(`/intel/risks${qs({ limit: 100 })}`);
  const risks = [...(data?.items ?? [])].sort((a, b) => b.score - a.score);

  return (
    <div>
      <PageHeader
        eyebrow="anticipate"
        title="Risk register"
        description="Each risk scores 100 × likelihood × impact, shrunk by the confidence in its evidence and the quality of the data behind it."
        asOf={data?.as_of}
        runId={data?.run_id}
      />

      {error ? (
        <IntelError error={error} retry={() => mutate()} />
      ) : !data ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[380px_1fr]">
          <Skeleton className="aspect-square w-full" />
          <RowsSkeleton rows={6} />
        </div>
      ) : risks.length === 0 ? (
        <EmptyState title="No risks assessed" body="The latest run found nothing to put on the register." />
      ) : (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,380px)_1fr]">
            <Panel title="Likelihood × impact">
              <RiskMatrix risks={risks} />
            </Panel>
            <Panel title="Register">
              <ol className="divide-y hairline">
                {risks.map((r, i) => (
                  <li key={r.id} className="grid grid-cols-[24px_1fr_auto] items-center gap-3 py-2 first:pt-0 last:pb-0">
                    <span className="num text-xs text-muted-foreground">{i + 1}</span>
                    <a href={`#${r.id}`} className="min-w-0 truncate text-sm hover:underline">{r.title}</a>
                    <span className="flex items-center gap-2">
                      <span className="num hidden text-sm sm:inline">{r.score.toFixed(0)}</span>
                      <SeverityBadge severity={r.level} />
                    </span>
                  </li>
                ))}
              </ol>
            </Panel>
          </div>

          {risks.map((r, i) => (
            <article key={r.id} id={r.id} className="scroll-mt-24 rounded-lg border bg-card p-4 sm:p-5">
              <header className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="eyebrow">{String(i + 1).padStart(2, "0")} · {humanize(r.kind)} · {r.entity_type} {r.entity}</p>
                  <h2 className="font-display mt-1 text-2xl leading-tight">{r.title}</h2>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-right">
                    <span className="block text-3xl leading-none">{r.score.toFixed(0)}</span>
                    <span className="eyebrow">score / 100</span>
                  </span>
                  <SeverityBadge severity={r.level} />
                </div>
              </header>

              <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-5">
                {([
                  ["Likelihood", r.likelihood],
                  ["Impact", r.impact],
                  ["Exposure", r.exposure],
                  ["Data quality", r.data_quality],
                ] as const).map(([label, v]) => (
                  <div key={label}>
                    <dt className="eyebrow mb-1">{label}</dt>
                    <dd><Meter value={v} label={`${label} ${v.toFixed(2)}`} /></dd>
                  </div>
                ))}
                <div>
                  <dt className="eyebrow mb-1">Confidence</dt>
                  <dd><ConfidenceBadge value={r.confidence} kind={r.confidence_kind} /></dd>
                </div>
              </dl>

              <div className="mt-5 grid grid-cols-1 gap-5 lg:grid-cols-2">
                <div>
                  <p className="eyebrow mb-2">Contributing factors</p>
                  <Factors r={r} />
                </div>
                <div className="space-y-4">
                  <div>
                    <p className="eyebrow mb-1.5">Recommended response</p>
                    <p className="text-sm leading-relaxed">{r.recommended_response}</p>
                  </div>
                  <HistoryTimeline entity="risks" historyKey={riskHistoryKey(r)} labels={{ score: "risk score", value: "exposure" }} format={(v) => fmtValue(v)} />
                  <details className="group">
                    <summary className="eyebrow inline-flex cursor-pointer list-none items-center gap-1 hover:text-foreground [&::-webkit-details-marker]:hidden">
                      <span className="transition-transform group-open:rotate-90" aria-hidden>›</span> Evidence ({r.evidence.length})
                    </summary>
                    <EvidenceList items={r.evidence} className="mt-2" />
                  </details>
                </div>
              </div>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}
