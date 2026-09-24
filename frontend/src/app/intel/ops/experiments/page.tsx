"use client";

import { Plus } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";

import { fmtDate } from "@/components/jev/admin/ui";
import Link from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { FilterRow, IntelError, RowsSkeleton } from "@/components/jev/intel/states";
import { ConclusionBadge, DataSourceBadge, ExperimentStatusBadge } from "@/components/jev/ops/badges";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { qs } from "@/lib/api";
import { EXPERIMENT_STATUS_LABEL, labelKind, METRIC_LABEL } from "@/lib/ops";
import type { ExperimentList, ExperimentStatus } from "@/lib/ops-types";

const STATUSES: ("all" | ExperimentStatus)[] = ["all", "draft", "running", "paused", "stopped", "concluded"];

export default function ExperimentsPage() {
  const [status, setStatus] = useState<"all" | ExperimentStatus>("all");
  const { data, error, mutate } = useSWR<ExperimentList>(`/experiments/online/list${qs({ status: status === "all" ? null : status, limit: 100 })}`, { keepPreviousData: true });

  return (
    <div>
      <PageHeader
        eyebrow="operations"
        title="Online experiments"
        description="A/B tests on the recommendations surface. Members are assigned by hash and never told their variant; at most one experiment runs per surface."
        action={<Button asChild><Link href="/intel/ops/experiments/new"><Plus aria-hidden /> New experiment</Link></Button>}
      />
      <div className="mb-4">
        <FilterRow label="Status" value={status} onChange={(v) => setStatus(v as typeof status)} options={STATUSES.map((s) => ({ value: s, label: s === "all" ? "All" : EXPERIMENT_STATUS_LABEL[s] }))} />
      </div>
      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} what="online experiments (GET /experiments/online/list)" since="Phase 2" />
      ) : !data ? (
        <RowsSkeleton rows={4} />
      ) : !data.items.length ? (
        <EmptyState
          title={status === "all" ? "No online experiments yet" : `No ${EXPERIMENT_STATUS_LABEL[status as ExperimentStatus].toLowerCase()} experiments`}
          body="Create a draft with a control and one or more treatments, then start it to begin serving. The offline replay report in experiments/ab-replay-* is not served by the API, so it is not listed here."
          action={<Button asChild variant="outline"><Link href="/intel/ops/experiments/new">New experiment</Link></Button>}
        />
      ) : (
        <div className="relative overflow-x-auto rounded-lg border bg-card">
          <table className="w-full min-w-[860px] text-sm">
            <caption className="sr-only">Online experiments</caption>
            <thead>
              <tr className="border-b hairline text-left text-xs text-muted-foreground">
                <th className="px-4 py-2.5 font-normal">Experiment</th>
                <th className="px-2 py-2.5 font-normal">Status</th>
                <th className="px-2 py-2.5 font-normal">Data</th>
                <th className="px-2 py-2.5 font-normal">Primary metric</th>
                <th className="px-2 py-2.5 text-right font-normal">Traffic</th>
                <th className="px-2 py-2.5 font-normal">Variants · members</th>
                <th className="px-2 py-2.5 font-normal">Conclusion</th>
                <th className="px-4 py-2.5 font-normal">Next</th>
              </tr>
            </thead>
            <tbody>
              {data.items.map((e) => (
                <tr key={e.key} className="border-b hairline align-top last:border-0">
                  <td className="px-4 py-2.5">
                    <Link href={`/intel/ops/experiments/${encodeURIComponent(e.key)}`} className="font-medium hover:underline">{e.name}</Link>
                    <span className="num block text-xs text-muted-foreground">{e.key} · created {fmtDate(e.created_at)}</span>
                  </td>
                  <td className="px-2 py-2.5"><ExperimentStatusBadge status={e.status} /></td>
                  <td className="px-2 py-2.5"><DataSourceBadge kind={labelKind(e.data_source, null)} /></td>
                  <td className="px-2 py-2.5">{METRIC_LABEL[e.primary_metric] ?? e.primary_metric}</td>
                  <td className="num px-2 py-2.5 text-right">{e.traffic_percent} %</td>
                  <td className="px-2 py-2.5 text-xs">
                    {e.variants.map((v) => (
                      <span key={v.name} className="block">
                        <span className="font-mono">{v.name}</span>{v.is_control && <span className="text-muted-foreground"> (control)</span>} · <span className="num">{v.assigned_users.toLocaleString("en")}</span>
                      </span>
                    ))}
                  </td>
                  <td className="px-2 py-2.5">{e.conclusion ? <ConclusionBadge conclusion={e.conclusion} /> : <span className="text-xs text-muted-foreground">—</span>}</td>
                  <td className="px-4 py-2.5 text-xs text-muted-foreground">{e.allowed_actions.filter((a) => a !== "update").join(" · ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
