"use client";

/*
 * Event log health (docs/STREAMING_ARCHITECTURE.md §5–6): one card per domain with today's and the
 * week's accepted / duplicate / rejected counts, ingest and event-time lag against the domain's own
 * expectation, the events not yet read by intelligence, the refresh state and the watermark of the
 * last live run. Rejections are the loudest alert: a producer is sending bad data.
 */

import { CheckCircle2, OctagonAlert, RefreshCw, ShieldCheck, Wrench } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

import { fmtDate, Panel, SpecRows, Status } from "@/components/jev/admin/ui";
import Link, { useIntelDomain } from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, PanelsSkeleton } from "@/components/jev/intel/states";
import { LagBadge } from "@/components/jev/ops/badges";
import { ConfirmDialog } from "@/components/jev/ops/confirm-dialog";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { api, errorMessage } from "@/lib/api";
import { eventLagLevel, eventsAlert, expectedEventLagS, fmtSeconds, ingestLagLevel, pendingText } from "@/lib/ops";
import type { EventDomainHealth, EventHealth, ReplayOut } from "@/lib/ops-types";
import { cn } from "@/lib/utils";

function Counts({ label, c, strong }: { label: string; c: EventDomainHealth["today"]; strong?: boolean }) {
  return (
    <div>
      <p className="eyebrow">{label}</p>
      <dl className="mt-1.5 grid grid-cols-3 gap-2">
        <div>
          <dt className="text-xs text-muted-foreground">accepted</dt>
          <dd className={cn("num leading-none", strong ? "mt-1 text-2xl" : "mt-0.5 text-base")}>{c.accepted.toLocaleString("en")}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted-foreground">duplicates</dt>
          <dd className={cn("num leading-none", strong ? "mt-1 text-2xl" : "mt-0.5 text-base")}>{c.duplicates.toLocaleString("en")}</dd>
        </div>
        <div>
          <dt className="flex items-center gap-1 text-xs text-muted-foreground">
            {c.rejected > 0 && <OctagonAlert className="size-3.5 text-destructive" aria-hidden />}rejected
          </dt>
          <dd className={cn("num leading-none", strong ? "mt-1 text-2xl" : "mt-0.5 text-base", c.rejected > 0 && "font-semibold text-destructive")}>{c.rejected.toLocaleString("en")}</dd>
        </div>
      </dl>
    </div>
  );
}

function DomainCard({ d, name, frequency }: { d: EventDomainHealth; name: string; frequency: string | null }) {
  const alert = eventsAlert(d);
  const expected = expectedEventLagS(d.domain, frequency);
  const ingest = ingestLagLevel(d.ingest_lag_s);
  const evt = eventLagLevel(d.event_time_lag_s, expected);
  const wm = d.watermark;
  const last = d.refresh?.last_refresh ?? null;
  const q = d.domain === "movie" ? "" : `?domain=${encodeURIComponent(d.domain)}`;
  return (
    <section aria-labelledby={`dom-${d.domain}`} className={cn("rounded-lg border bg-card p-4 sm:p-5", alert === "rejected_today" && "border-destructive/60")}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id={`dom-${d.domain}`} className="font-display text-2xl leading-tight">{name}</h2>
          <p className="break-all font-mono text-xs text-muted-foreground">{d.domain} · {d.events.toLocaleString("en")} events in the log</p>
        </div>
        {alert ? (
          <p role="status" className="inline-flex items-center gap-1.5 rounded border border-destructive/50 bg-destructive/5 px-2 py-1 text-sm">
            <OctagonAlert className="size-4 text-destructive" aria-hidden />
            {alert === "rejected_today" ? `${d.today.rejected} rejected today` : `${d.last_7_days.rejected} rejected this week`}
          </p>
        ) : (
          <p className="inline-flex items-center gap-1.5 text-sm text-muted-foreground"><ShieldCheck className="size-4" style={{ color: "var(--status-good)" }} aria-hidden /> No rejections this week</p>
        )}
      </div>

      <div className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-2">
        <Counts label="Today (UTC)" c={d.today} strong />
        <Counts label="Last 7 days" c={d.last_7_days} />
      </div>
      {alert && <p className="mt-2 text-xs text-ink-2">Rejected items never reach the log. Each response carries the reason per item; fix the producer, then resend with new idempotency keys.</p>}

      <div className="mt-5 grid grid-cols-1 gap-x-6 sm:grid-cols-2">
        <SpecRows
          rows={[
            { label: "Last ingest", value: <span className="inline-flex items-center gap-2">{d.ingest_lag_s != null ? `${fmtSeconds(d.ingest_lag_s)} ago` : "—"}<LagBadge level={ingest} /></span> },
            { label: "Newest event time", value: <span className="inline-flex items-center gap-2">{d.event_time_lag_s != null ? `${fmtSeconds(d.event_time_lag_s)} old` : "—"}<LagBadge level={evt} /></span> },
            { label: "Expected event age", value: `≤ ${fmtSeconds(expected)}` },
          ]}
        />
        <SpecRows
          rows={[
            { label: "Refresh", value: d.refresh ? (d.refresh.dirty ? <span className="inline-flex items-center gap-1.5 font-sans"><RefreshCw className="size-3.5 text-primary" aria-hidden />dirty · run pending</span> : <Status ok label="clean" />) : "—" },
            { label: "Last refresh", value: last ? <span title={`${fmtDate(last.at)} UTC`}>{`${last.status ?? "?"} · ${fmtDate(last.at).slice(11)}`}</span> : "none in this process" },
            { label: "Last live run", value: d.last_live_run ? <Link href={`/intel/health${q}`} className="text-primary hover:underline" title={d.last_live_run.run_id}>{d.last_live_run.run_id.slice(0, 8)} · {d.last_live_run.trigger}</Link> : "none yet" },
          ]}
        />
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 border-t hairline pt-3 text-sm">
        <span className={cn("inline-flex items-center gap-1.5", d.events_since_last_run > 0 ? "text-foreground" : "text-muted-foreground")}>
          {d.events_since_last_run > 0 ? <RefreshCw className="size-4 text-primary" aria-hidden /> : <CheckCircle2 className="size-4" style={{ color: "var(--status-good)" }} aria-hidden />}
          {pendingText(d.events_since_last_run)}
        </span>
        <span className="font-mono text-xs text-muted-foreground">
          watermark {wm ? `#${wm.max_event_id ?? "—"} · ingested ${wm.max_ingested_at ? fmtDate(wm.max_ingested_at) : "—"}` : "— (no live run has read the log)"} · log head #{d.max_event_id ?? "—"}
        </span>
      </div>
    </section>
  );
}

function ReplayPanel() {
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ReplayOut | null>(null);

  async function run(apply: boolean) {
    setBusy(true);
    try {
      const r = await api<ReplayOut>("/events/replay", { json: { apply } });
      setResult(r);
      if (apply) toast.success(r.consistent_after ? "Projections rebuilt from the log; consistent." : "Rebuilt, but still inconsistent. See the diff.");
      else toast.message(r.consistent_before ? "Projections match the log." : "Projections differ from the log.");
      setConfirm(false);
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  const diff = result?.diff;
  return (
    <Panel title="Projections vs the log" action={
      <div className="flex gap-2">
        <Button size="sm" variant="outline" onClick={() => run(false)} disabled={busy}><ShieldCheck aria-hidden /> Verify</Button>
        <Button size="sm" variant="outline" onClick={() => setConfirm(true)} disabled={busy}><Wrench aria-hidden /> Repair…</Button>
      </div>
    }>
      <p className="text-sm text-ink-2">
        Ratings, favourites, recommendation feedback and watch history are projections of the movie event log. Verify folds the log and compares; repair rewrites the tables from it.
      </p>
      {result && diff && (
        <div className="mt-4">
          <p className="text-sm">
            <Status ok={result.consistent_before} label={result.consistent_before ? "consistent with the log" : "differs from the log"} />
            {result.applied && <span className="ml-3"><Status ok={result.consistent_after} label={`after repair: ${result.consistent_after ? "consistent" : "still differs"}`} /></span>}
          </p>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[420px] text-sm">
              <caption className="sr-only">Projection differences found {result.applied ? "before the repair" : ""}</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-3 font-normal">Projection</th>
                  <th className="px-2 py-2 text-right font-normal">Missing</th>
                  <th className="px-2 py-2 text-right font-normal">Extra</th>
                  <th className="px-2 py-2 text-right font-normal">Changed</th>
                  <th className="py-2 pl-3 font-normal">Sample</th>
                </tr>
              </thead>
              <tbody>
                {(["ratings", "favorites", "recommendation_feedback", "watch_history"] as const).map((k) => (
                  <tr key={k} className="border-b hairline last:border-0">
                    <td className="py-2 pr-3">{k.replaceAll("_", " ")}</td>
                    <td className="num px-2 py-2 text-right">{diff[k].missing}</td>
                    <td className="num px-2 py-2 text-right">{diff[k].extra}</td>
                    <td className="num px-2 py-2 text-right">{diff[k].changed}</td>
                    <td className="max-w-[240px] truncate py-2 pl-3 font-mono text-xs text-muted-foreground" title={diff[k].sample.join(", ")}>{diff[k].sample.join(", ") || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      <ConfirmDialog
        open={confirm}
        onOpenChange={setConfirm}
        title="Rebuild projections from the log?"
        description="Every movie projection row that differs from the event log is rewritten, rows without events are removed, and cached recommendations are cleared. The replay is audited as events.replay."
        confirmLabel="Rebuild projections"
        destructive
        onConfirm={() => run(true)}
      />
    </Panel>
  );
}

export default function EventsPage() {
  const { domains } = useIntelDomain();
  const { data, error, mutate } = useSWR<EventHealth>("/events/health", { refreshInterval: 15000 });
  const rejecting = data?.domains.filter((d) => d.today.rejected > 0) ?? [];
  const r = data?.refresher;

  return (
    <div>
      <PageHeader
        eyebrow="operations"
        title="Event log"
        description="What is flowing into JEV: accepted, duplicate and rejected events per domain, how far behind the feed is, and what the next live run will pick up."
      />

      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} what="event health (GET /events/health)" since="Phase 2" />
      ) : !data ? (
        <PanelsSkeleton n={2} className="lg:grid-cols-2" />
      ) : (
        <div className="space-y-6">
          {rejecting.length > 0 && (
            <div role="alert" className="flex items-start gap-3 rounded-lg border border-destructive/50 bg-destructive/5 px-4 py-3">
              <OctagonAlert className="mt-0.5 size-5 shrink-0 text-destructive" aria-hidden />
              <p className="text-sm">
                <strong className="font-semibold">A producer is sending bad data.</strong>{" "}
                {rejecting.map((d) => `${d.today.rejected.toLocaleString("en")} rejected today in ${d.domain}`).join("; ")}.
              </p>
            </div>
          )}

          <section aria-label="Refresher" className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border bg-card px-4 py-3 text-sm">
            <span className="eyebrow">Near-real-time refresh</span>
            <span className="flex items-center gap-1.5 text-muted-foreground">Enabled <Status ok={Boolean(r?.enabled)} label={r?.enabled ? "yes" : "no"} /></span>
            <span className="flex items-center gap-1.5 text-muted-foreground">Worker {r?.running ? <Status ok label="running" /> : <span className="text-foreground">idle (starts on the first event)</span>}</span>
            {r?.debounce_seconds !== undefined && <span className="num text-xs text-muted-foreground">debounce {fmtSeconds(r.debounce_seconds)} · at most {fmtSeconds(r.max_delay_seconds)} under steady traffic</span>}
            <span className="text-xs text-muted-foreground">{r?.dirty_domains?.length ? `pending: ${r.dirty_domains.join(", ")}` : "nothing pending"}</span>
            <span className="num ml-auto text-xs text-muted-foreground">checked {fmtDate(data.generated_at)} UTC</span>
          </section>

          {data.domains.length === 0 ? (
            <EmptyState
              title="No events yet"
              body="Members' ratings, favourites, watches and feedback land here through POST /events and the member endpoints; generic domains push observations. Nothing has been logged on this database."
            />
          ) : (
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
              {data.domains.map((d) => {
                const info = domains?.find((x) => x.key === d.domain);
                return <DomainCard key={d.domain} d={d} name={info?.name ?? d.domain} frequency={info?.frequency ?? null} />;
              })}
            </div>
          )}
          <p className="text-xs text-muted-foreground">
            Lag is measured against each domain&apos;s own expectation: the movie domain&apos;s app events are live, a monthly statistic is weeks old by nature. The API reports the week as one total, so no daily trend is drawn.
          </p>

          <ReplayPanel />
        </div>
      )}
    </div>
  );
}
