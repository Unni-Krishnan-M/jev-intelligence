"use client";

import useSWR from "swr";

import { fmtDate, Panel, SpecRows, Status } from "@/components/jev/admin/ui";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError } from "@/lib/api";
import { fmtValue, fmtMs, humanize, isNoRun } from "@/lib/intel";
import type { AdminMetrics, IntelStatus, RunList } from "@/lib/intel-types";

interface Health {
  status: string;
  version?: string;
  database?: string;
  cache?: { backend: string; status: string };
  error?: string;
}
interface MlHealth {
  status: string;
  error?: string;
  model_version?: string;
  dataset_version?: string;
}

// Health endpoints answer 503 with a JSON body when degraded; show that instead of an error.
async function tolerant<T>(path: string): Promise<T> {
  try {
    return await api<T>(path);
  } catch (e) {
    if (e instanceof ApiError && e.status === 503) return { status: "degraded", error: e.message } as T;
    throw e;
  }
}

/** Flatten the counters object into rows grouped by its top-level keys. */
function metricGroups(m: AdminMetrics): { title: string; rows: { label: string; value: string }[] }[] {
  const scalars: { label: string; value: string }[] = [];
  const groups: { title: string; rows: { label: string; value: string }[] }[] = [];
  const walk = (obj: Record<string, unknown>, prefix: string, out: { label: string; value: string }[]) => {
    for (const [k, v] of Object.entries(obj)) {
      const label = prefix ? `${prefix} · ${humanize(k)}` : humanize(k);
      if (v && typeof v === "object" && !Array.isArray(v)) walk(v as Record<string, unknown>, label, out);
      else out.push({ label, value: fmtValue(v, k) });
    }
  };
  for (const [k, v] of Object.entries(m)) {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      const rows: { label: string; value: string }[] = [];
      walk(v as Record<string, unknown>, "", rows);
      groups.push({ title: humanize(k), rows });
    } else {
      scalars.push({ label: humanize(k), value: fmtValue(v, k) });
    }
  }
  return scalars.length ? [{ title: "Counters", rows: scalars }, ...groups] : groups;
}

export default function HealthPage() {
  const health = useSWR<Health>("intel:health", () => tolerant<Health>("/health"), { refreshInterval: 30000 });
  const ml = useSWR<MlHealth>("intel:health-ml", () => tolerant<MlHealth>("/health/ml"), { refreshInterval: 30000 });
  const status = useSWR<IntelStatus>("/intel/status", { refreshInterval: 60000 });
  const metrics = useSWR<AdminMetrics>("/admin/metrics", { refreshInterval: 30000 });
  const runs = useSWR<RunList>("/intel/runs");
  const ih = status.data?.health;

  return (
    <div>
      <PageHeader
        eyebrow="review"
        title="System health"
        description="The API, database, cache and model, the intelligence pipeline, in-process counters and every recorded run."
      />

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel title="API">
          {health.error ? (
            <IntelError error={health.error} retry={() => health.mutate()} runBacked={false} />
          ) : !health.data ? (
            <Skeleton className="h-28 w-full" />
          ) : (
            <SpecRows
              rows={[
                { label: "API", value: <Status ok={health.data.status === "ok"} label={`${health.data.status}${health.data.version ? ` · v${health.data.version}` : ""}`} /> },
                { label: "Database", value: <Status ok={health.data.database === "ok"} label={health.data.database ?? "unknown"} /> },
                { label: "Cache", value: <Status ok={health.data.cache?.status === "ok"} label={health.data.cache ? `${health.data.cache.backend} · ${health.data.cache.status}` : "unknown"} /> },
              ]}
            />
          )}
        </Panel>
        <Panel title="Recommendation model">
          {ml.error ? (
            <IntelError error={ml.error} retry={() => ml.mutate()} runBacked={false} />
          ) : !ml.data ? (
            <Skeleton className="h-28 w-full" />
          ) : (
            <SpecRows
              rows={[
                { label: "Status", value: <Status ok={ml.data.status === "ok"} label={ml.data.status} /> },
                { label: "Version", value: ml.data.model_version ?? "—" },
                { label: "Dataset", value: ml.data.dataset_version ?? "—" },
                ...(ml.data.error ? [{ label: "Error", value: <span title={ml.data.error}>{ml.data.error}</span> }] : []),
              ]}
            />
          )}
        </Panel>
        <Panel title="Intelligence layer">
          {status.error && !isNoRun(status.error) ? (
            <IntelError error={status.error} retry={() => status.mutate()} />
          ) : !status.data && !status.error ? (
            <Skeleton className="h-28 w-full" />
          ) : !ih ? (
            <p className="text-sm text-muted-foreground"><Status ok={false} label="never run" /> No pipeline run is recorded yet.</p>
          ) : (
            <SpecRows
              rows={[
                { label: "Database", value: <Status ok={ih.database === "ok"} label={ih.database} /> },
                { label: "Cache", value: <Status ok={ih.cache === "ok"} label={ih.cache} /> },
                { label: "Model", value: <Status ok={ih.model === "ok"} label={ih.model} /> },
                { label: "Pipeline", value: <Status ok={ih.pipeline === "ok"} label={humanize(ih.pipeline)} /> },
              ]}
            />
          )}
        </Panel>
      </div>

      <section aria-labelledby="counters-heading" className="mt-8">
        <h2 id="counters-heading" className="eyebrow mb-3">In-process counters · since API start</h2>
        {metrics.error ? (
          <IntelError error={metrics.error} retry={() => metrics.mutate()} runBacked={false} />
        ) : !metrics.data ? (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">{Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-40" />)}</div>
        ) : Object.keys(metrics.data).length === 0 ? (
          <p className="text-sm text-muted-foreground">No counters reported yet.</p>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-3">
            {metricGroups(metrics.data).map((g) => (
              <Panel key={g.title} title={g.title}>
                {g.rows.length ? <SpecRows rows={g.rows} /> : <p className="text-sm text-muted-foreground">Nothing counted yet.</p>}
              </Panel>
            ))}
          </div>
        )}
      </section>

      <section aria-labelledby="runs-heading" className="mt-8">
        <h2 id="runs-heading" className="eyebrow mb-3">Pipeline runs</h2>
        {runs.error ? (
          <IntelError error={runs.error} retry={() => runs.mutate()} runBacked={false} />
        ) : !runs.data ? (
          <RowsSkeleton rows={5} />
        ) : runs.data.items.length === 0 ? (
          <EmptyState title="No runs recorded" body="Run the pipeline from the overview; every run is recorded here with its timings and any error." />
        ) : (
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full min-w-[860px] text-sm">
              <caption className="sr-only">Recorded pipeline runs</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2.5 font-normal">Run</th>
                  <th className="px-2 py-2.5 font-normal">Status</th>
                  <th className="px-2 py-2.5 font-normal">As of</th>
                  <th className="px-2 py-2.5 font-normal">Started</th>
                  <th className="px-2 py-2.5 text-right font-normal">Duration</th>
                  <th className="px-2 py-2.5 font-normal">Versions</th>
                  <th className="px-4 py-2.5 font-normal">Summary / error</th>
                </tr>
              </thead>
              <tbody>
                {runs.data.items.map((r) => (
                  <tr key={r.run_id} className="border-b hairline align-top last:border-0">
                    <td className="px-4 py-2.5">
                      <span className="num" title={r.run_id}>{r.run_id.slice(0, 8)}</span>
                      <span className="block text-xs text-muted-foreground">{r.trigger}</span>
                    </td>
                    <td className="px-2 py-2.5">
                      {r.status === "running" ? <span className="text-muted-foreground">running…</span> : <Status ok={r.status === "succeeded"} label={r.status} />}
                    </td>
                    <td className="num px-2 py-2.5 text-xs">{fmtDate(r.as_of)}</td>
                    <td className="num px-2 py-2.5 text-xs text-muted-foreground">{fmtDate(r.started_at)}</td>
                    <td className="num px-2 py-2.5 text-right">{fmtMs(r.duration_ms)}</td>
                    <td className="px-2 py-2.5 font-mono text-xs text-muted-foreground">
                      {r.pipeline_version}
                      <span className="block max-w-[180px] truncate" title={r.model_version ?? undefined}>{r.model_version ?? "no model"}</span>
                    </td>
                    <td className="max-w-[280px] px-4 py-2.5 text-xs">
                      {r.error ? <span className="text-destructive">{r.error}</span> : r.summary ? <span className="text-ink-2">{r.summary.status} · {r.summary.headline}</span> : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
