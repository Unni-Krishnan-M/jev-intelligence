"use client";

import { OctagonAlert } from "lucide-react";
import useSWR from "swr";

import Link from "@/components/jev/intel/domain-context";
import { ExperimentStatusBadge, GateVerdictBadge, LagBadge } from "@/components/jev/ops/badges";
import { isNotDeployed } from "@/lib/intel";
import { fmtSeconds, gateVerdict, ingestLagLevel } from "@/lib/ops";
import type { EventHealth, ExperimentList, GovernedModelList } from "@/lib/ops-types";

/** The overview's one-line view of operations: the serving model and its gate, the event feed, running experiments. */
export function OpsStrip() {
  const models = useSWR<GovernedModelList>("/governance/models", { shouldRetryOnError: false });
  const events = useSWR<EventHealth>("/events/health", { refreshInterval: 60000, shouldRetryOnError: false });
  const exps = useSWR<ExperimentList>("/experiments/online/list?status=running", { shouldRetryOnError: false });

  if ([models.error, events.error, exps.error].every((e) => e && isNotDeployed(e))) return null;

  const active = models.data?.items.find((m) => m.is_active);
  const candidate = models.data?.items.find((m) => m.state === "candidate");
  const doms = events.data?.domains ?? [];
  const rejected = doms.reduce((s, d) => s + d.today.rejected, 0);
  const lag = doms.length ? Math.min(...doms.map((d) => d.ingest_lag_s ?? Infinity)) : null;
  const pending = doms.reduce((s, d) => s + d.events_since_last_run, 0);
  const running = exps.data?.items ?? [];
  const na = <span className="text-muted-foreground">not available</span>;
  const dots = <span className="text-muted-foreground">…</span>;

  return (
    <section aria-label="Operations" className="grid grid-cols-1 gap-x-6 gap-y-3 rounded-lg border bg-card px-4 py-3 text-sm md:grid-cols-3">
      <div className="min-w-0">
        <p className="eyebrow flex items-center justify-between gap-2">Model <Link href="/intel/ops/models" className="normal-case tracking-normal text-primary hover:underline">Lifecycle →</Link></p>
        {models.error ? na : !models.data ? dots : (
          <div className="mt-1 flex flex-wrap items-center gap-2">
            <span className="min-w-0 truncate font-mono text-xs" title={models.data.active ?? undefined}>{models.data.active ?? "none active"}</span>
            {candidate ? (
              <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">candidate <GateVerdictBadge verdict={gateVerdict(candidate, models.data.active)} /></span>
            ) : active ? (
              <span className="text-xs text-muted-foreground">no candidate waiting</span>
            ) : null}
          </div>
        )}
      </div>
      <div className="min-w-0">
        <p className="eyebrow flex items-center justify-between gap-2">Events <Link href="/intel/ops/events" className="normal-case tracking-normal text-primary hover:underline">Event log →</Link></p>
        {events.error ? na : !events.data ? dots : !doms.length ? <p className="mt-1 text-xs text-muted-foreground">no events logged yet</p> : (
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs">
            {rejected > 0 ? (
              <span className="inline-flex items-center gap-1 font-semibold"><OctagonAlert className="size-3.5 text-destructive" aria-hidden />{rejected} rejected today</span>
            ) : (
              <span className="text-muted-foreground">0 rejected today</span>
            )}
            <span className="inline-flex items-center gap-1.5">last ingest {lag !== null && Number.isFinite(lag) ? `${fmtSeconds(lag)} ago` : "—"} <LagBadge level={ingestLagLevel(lag !== null && Number.isFinite(lag) ? lag : null)} /></span>
            <span className="text-muted-foreground">{pending.toLocaleString()} pending</span>
          </div>
        )}
      </div>
      <div className="min-w-0">
        <p className="eyebrow flex items-center justify-between gap-2">Experiments <Link href="/intel/ops/experiments" className="normal-case tracking-normal text-primary hover:underline">All →</Link></p>
        {exps.error ? na : !exps.data ? dots : !running.length ? <p className="mt-1 text-xs text-muted-foreground">none running</p> : (
          <ul className="mt-1 space-y-1 text-xs">
            {running.map((e) => (
              <li key={e.key} className="flex flex-wrap items-center gap-2">
                <Link href={`/intel/ops/experiments/${encodeURIComponent(e.key)}`} className="hover:underline">{e.name}</Link>
                <ExperimentStatusBadge status={e.status} />
                <span className="num text-muted-foreground">{e.traffic_percent} %</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}
