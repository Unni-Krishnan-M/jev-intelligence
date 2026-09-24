"use client";

import { CheckCircle2, CircleAlert, GitBranch } from "lucide-react";
import useSWR from "swr";

import { fmtDate, Panel, SpecRows } from "@/components/jev/admin/ui";
import Link from "@/components/jev/intel/domain-context";
import { IntelError } from "@/components/jev/intel/states";
import { RunModeBadge } from "@/components/jev/ops/badges";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError } from "@/lib/api";
import { fmtValue, refHref, seriesHref } from "@/lib/intel";
import { LINEAGE_LAYERS, lineageLayers } from "@/lib/ops";
import type { IntelLineage, LineageNode } from "@/lib/ops-types";

function nodeHref(n: LineageNode): string | null {
  if (n.type === "series" && n.ref) return seriesHref(n.ref);
  if (n.type === "run" || n.type === "source") return null;
  return refHref(n.ref ?? null);
}

/**
 * The evidence bundle behind a decision or warning (GET /intel/{decisions|warnings}/{id}/lineage):
 * the root, the run objects its evidence resolves to, their series and data sources, and the run
 * that read them (mode, versions, config hash, input fingerprint, events watermark). An API without
 * the endpoint (a route-miss 404) says "not available" instead of failing.
 */
export function LineageViewer({ path }: { path: string | null }) {
  const { data, error, mutate } = useSWR<IntelLineage>(path, { shouldRetryOnError: false });
  const title = <span className="inline-flex items-center gap-1.5"><GitBranch className="size-3.5" aria-hidden /> Lineage</span>;
  if (error) {
    const missing = error instanceof ApiError && error.status === 404;
    return (
      <Panel title={title}>
        {missing ? (
          <p className="text-sm text-muted-foreground">Lineage not available{error.message && error.message !== "Not Found" ? `: ${error.message}` : " on this API version"}.</p>
        ) : (
          <IntelError error={error} retry={() => mutate()} runBacked={false} />
        )}
      </Panel>
    );
  }
  if (!data) return <Panel title={title}><Skeleton className="h-40 w-full" /></Panel>;

  const run = data.run;
  const wm = data.event_watermark ?? run.event_watermark;
  const layers = lineageLayers(data);
  return (
    <Panel title={title} action={
      data.complete ? (
        <span className="inline-flex items-center gap-1.5 text-xs"><CheckCircle2 className="size-3.5" style={{ color: "var(--status-good)" }} aria-hidden /> Complete</span>
      ) : (
        <span className="inline-flex items-center gap-1.5 text-xs"><CircleAlert className="size-3.5" style={{ color: "var(--status-serious)" }} aria-hidden /> Incomplete</span>
      )
    }>
      <p className="text-sm text-ink-2">
        {data.nodes.length} nodes, {data.edges.length} links, from {data.root.type} to {data.sources.length} data source{data.sources.length === 1 ? "" : "s"}.
        {!data.complete && data.unresolved.length > 0 && ` ${data.unresolved.length} reference${data.unresolved.length === 1 ? "" : "s"} did not resolve inside the run.`}
      </p>

      <ol className="mt-4 space-y-3">
        {LINEAGE_LAYERS.map((layer) => {
          const nodes = layers[layer.key];
          if (!nodes.length) return null;
          return (
            <li key={layer.key} className="border-l-2 border-rule pl-3">
              <p className="eyebrow">{layer.label}</p>
              <ul className="mt-1 space-y-1 text-sm">
                {nodes.map((n) => {
                  const href = nodeHref(n);
                  const rel = data.edges.filter((e) => e.to === n.id).map((e) => e.relation);
                  return (
                    <li key={n.id} className="min-w-0">
                      {href ? <Link href={href} className="break-words hover:underline">{n.title ?? n.ref}</Link> : <span className="break-words">{n.title ?? n.ref}</span>}
                      <span className="block break-all font-mono text-[11px] text-muted-foreground">
                        {n.id}{rel.length ? ` · ${[...new Set(rel)].join(", ").replaceAll("_", " ")}` : ""}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </li>
          );
        })}
      </ol>

      <p className="eyebrow mb-1 mt-5">Run</p>
      <SpecRows
        rows={[
          { label: "Run", value: <span className="inline-flex items-center gap-2"><span title={run.run_id}>{run.run_id.slice(0, 8)}</span><RunModeBadge mode={run.mode} /></span> },
          { label: "As of", value: run.as_of ? fmtDate(run.as_of) : "—" },
          { label: "Pipeline · core", value: `${run.pipeline_version}${run.core_version ? ` · ${run.core_version}` : ""}` },
          { label: "Model", value: run.model_version ?? "—" },
          { label: "Data", value: <span title={run.data_version ?? undefined}>{run.data_version ?? "—"}</span> },
          { label: "Config hash", value: run.config_hash ?? "—" },
          { label: "Input fingerprint", value: <span title={run.input_fingerprint ?? undefined}>{run.input_fingerprint ?? "—"}</span> },
          { label: "Events watermark", value: wm ? `#${wm.max_event_id ?? "—"} · ${fmtValue(wm.n_events)} events` : "— (before the event log)" },
          ...(wm?.knowledge_time ? [{ label: "Knowledge time", value: fmtDate(wm.knowledge_time) }] : []),
        ]}
      />

      {data.evidence.length > 0 && (
        <details className="mt-4">
          <summary className="cursor-pointer text-xs text-muted-foreground">{data.evidence.length} evidence item{data.evidence.length === 1 ? "" : "s"} and where they resolve</summary>
          <ul className="mt-2 space-y-1 text-xs">
            {data.evidence.map((e) => (
              <li key={`${e.owner}-${e.position}`} className="flex flex-wrap gap-x-2">
                <span>{e.label ?? e.kind}</span>
                <span className="num text-ink-2">{fmtValue(e.value)}</span>
                <span className="break-all font-mono text-muted-foreground">{e.ref ? `${e.ref} → ${e.resolves_to ?? "unresolved"}` : "no ref"}</span>
              </li>
            ))}
          </ul>
        </details>
      )}
      {data.unresolved.length > 0 && (
        <ul className="mt-3 space-y-0.5 text-xs text-muted-foreground">
          {data.unresolved.map((u, i) => <li key={i} className="break-all font-mono">unresolved: {u.ref} (from {u.from})</li>)}
        </ul>
      )}
    </Panel>
  );
}
