"use client";

import { Play } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

import { fmtDate, Panel, SpecRows, Status } from "@/components/jev/admin/ui";
import {
  ConfidenceBadge,
  DirectionIcon,
  FreshnessBadge,
  Meter,
  ModelHealth,
  SEVERITY_META,
  SeverityBadge,
  SYSTEM_META,
} from "@/components/jev/intel/badges";
import { CountBars, NamedBars, TableView } from "@/components/jev/intel/charts";
import Link, { useIntelDomain } from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, PanelsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { api, errorMessage } from "@/lib/api";
import { DEFAULT_DOMAIN } from "@/lib/domain";
import { daysBetween, fmtAnswer, fmtDays, fmtMs, fmtPct, humanize, orderStages, SEVERITIES, useIntelRevalidate } from "@/lib/intel";
import type { IntelStatus, Run } from "@/lib/intel-types";

function RunControl({ lastAsOf }: { lastAsOf?: string | null }) {
  const [asOf, setAsOf] = useState("");
  const [busy, setBusy] = useState(false);
  const revalidate = useIntelRevalidate();
  const { domain, name } = useIntelDomain();

  async function run() {
    setBusy(true);
    try {
      // the domain goes in the body (platform.md §8); the API defaults to movie without it
      const r = await api<Run>("/intel/runs", { json: asOf ? { domain, as_of: asOf } : { domain } });
      if (r.status === "failed") toast.error(`Run failed${r.error ? `: ${r.error}` : ""}`);
      else toast.success(`Run ${r.status} · as of ${fmtDate(r.as_of)} · ${fmtMs(r.duration_ms)}`);
      await revalidate();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="w-full rounded-lg border bg-card p-4 sm:w-auto sm:min-w-[300px]">
      <p className="eyebrow">Run the pipeline · {name}</p>
      <div className="mt-3 flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-1 space-y-1">
          <Label htmlFor="replay" className="text-xs font-normal text-muted-foreground">Replay as of (optional)</Label>
          <Input id="replay" type="date" value={asOf} onChange={(e) => setAsOf(e.target.value)} className="num h-8" max={new Date().toISOString().slice(0, 10)} />
        </div>
        <Button onClick={run} disabled={busy}>
          <Play aria-hidden /> {busy ? "Running…" : asOf ? "Replay" : "Run now"}
        </Button>
      </div>
      <p className="mt-2 text-xs text-muted-foreground" aria-live="polite">
        {busy
          ? "Running synchronously; this usually takes a few seconds."
          : asOf
            ? "Uses only data at or before that date, so the run is leak-free."
            : `Defaults to the latest event in the ${name} data${lastAsOf ? ` (last run: ${fmtDate(lastAsOf)})` : ""}.`}
      </p>
    </div>
  );
}

export default function IntelOverview() {
  const { domain, name, q: dq } = useIntelDomain();
  const { data, error, mutate } = useSWR<IntelStatus>(dq("/intel/status"), { refreshInterval: 60000 });
  const run = data?.latest_run ?? null;

  return (
    <div>
      <PageHeader
        eyebrow={`overview · ${name}`}
        title="Situation report"
        description={`What the ${name} data says right now: the status, the evidence behind it, what JEV decided and what is waiting for a person.`}
        asOf={run?.as_of}
        runId={run?.run_id}
        action={<RunControl lastAsOf={run?.as_of} />}
      />

      {error ? (
        <IntelError error={error} retry={() => mutate()} />
      ) : !data ? (
        <div className="space-y-4">
          <Skeleton className="h-28 w-full" />
          <PanelsSkeleton />
          <PanelsSkeleton />
        </div>
      ) : !run ? (
        <EmptyState
          title="No intelligence run yet — run the pipeline"
          body={domain === DEFAULT_DOMAIN ? "Use “Run now” above. The first run reads the MovieLens snapshot, the app database and the active model." : `Use “Run now” above. The first run reads the ${name} dataset through its domain adapter.`}
        />
      ) : (
        <Overview s={data} run={run} />
      )}
    </div>
  );
}

function Overview({ s, run }: { s: IntelStatus; run: Run }) {
  const summary = s.summary ?? run.summary;
  const sys = summary ? SYSTEM_META[summary.status] : null;
  const lag = daysBetween(run.as_of, run.started_at);
  const stages = orderStages(run.stage_ms);
  const checks = [...(s.data?.quality.checks ?? [])].sort((a, b) => Number(a.passed) - Number(b.passed));

  return (
    <div className="space-y-8">
      {/* headline */}
      <section aria-labelledby="status-heading" className="grid grid-cols-1 gap-6 border-b hairline pb-8 lg:grid-cols-[minmax(0,1fr)_auto]">
        <div className="min-w-0">
          {sys && summary ? (
            <>
              <p className="flex items-center gap-2">
                <sys.Icon className="size-5" style={{ color: sys.color }} aria-hidden />
                <span id="status-heading" className="text-lg">System status: <strong className="font-semibold">{sys.label}</strong></span>
                <span className="hidden text-sm text-muted-foreground sm:inline">— {sys.blurb}</span>
              </p>
              <p className="font-display mt-3 text-[28px] leading-tight text-balance sm:text-[34px]">{summary.headline}</p>
            </>
          ) : (
            <p id="status-heading" className="text-muted-foreground">The latest run has no summary{run.status === "failed" ? " because it failed" : ""}.</p>
          )}
          {run.status === "failed" && <p className="mt-2 text-sm"><Status ok={false} label={`Last run failed: ${run.error ?? "no error recorded"}`} /></p>}
        </div>
        {summary && (
          <dl className="grid grid-cols-2 gap-x-6 min-[420px]:grid-cols-4 gap-y-3 self-end">
            {([
              ["signals", "Signals"],
              ["trends_up", "Trends ↑"],
              ["trends_down", "Trends ↓"],
              ["anomalies", "Anomalies"],
              ["risks_high", "High risks"],
              ["warnings", "Warnings"],
              ["decisions", "Decisions"],
              ["actions", "Actions"],
            ] as const).map(([k, label]) => (
              <div key={k}>
                <dt className="eyebrow">{label}</dt>
                <dd className="mt-0.5 text-2xl leading-none">{summary.counts[k].toLocaleString()}</dd>
              </div>
            ))}
          </dl>
        )}
      </section>

      {/* health strip */}
      <section aria-label="System health" className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-lg border bg-card px-4 py-3 text-sm">
        <span className="eyebrow">Health</span>
        <span className="flex items-center gap-1.5 text-muted-foreground">Database <Status ok={s.health.database === "ok"} label={s.health.database} /></span>
        <span className="flex items-center gap-1.5 text-muted-foreground">Cache <Status ok={s.health.cache === "ok"} label={s.health.cache} /></span>
        <span className="flex items-center gap-1.5 text-muted-foreground">Model <ModelHealth value={s.health.model} /></span>
        <span className="flex items-center gap-1.5 text-muted-foreground">Pipeline <Status ok={s.health.pipeline === "ok"} label={humanize(s.health.pipeline)} /></span>
        <Link href="/intel/health" className="ml-auto text-xs text-primary hover:underline">System health →</Link>
      </section>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel title="Latest run">
          <SpecRows
            rows={[
              { label: "Status", value: <Status ok={run.status === "succeeded"} label={`${run.status} · ${run.trigger}`} /> },
              { label: "As of (data clock)", value: fmtDate(run.as_of) },
              { label: "Run at (wall clock)", value: fmtDate(run.started_at) },
              { label: "Data clock behind", value: fmtDays(lag) },
              { label: "Duration", value: fmtMs(run.duration_ms) },
              { label: "Pipeline", value: run.pipeline_version },
              { label: "Model", value: run.model_version ?? "— (no model)" },
              { label: "Data", value: <span title={run.data_version}>{run.data_version}</span> },
            ]}
          />
        </Panel>

        <Panel title="Data freshness">
          {!s.data?.sources.length ? (
            <p className="text-sm text-muted-foreground">No sources reported by the latest run.</p>
          ) : (
            <ul className="divide-y hairline text-sm">
              {s.data.sources.map((src) => (
                <li key={src.source} className="py-2.5 first:pt-0 last:pb-0">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-medium">{src.source}</span>
                    <FreshnessBadge kind={src.kind} rows={src.rows} days={src.age_days} fresh={src.fresh} />
                  </div>
                  <p className="mt-1 font-mono text-xs text-muted-foreground">
                    {humanize(src.kind)} · {src.rows.toLocaleString()} rows · last {src.last_event ? fmtDate(src.last_event) : "—"}
                    {src.expected_update ? ` · expected ${src.expected_update}` : " · no update SLA"}
                  </p>
                  {src.detail && <p className="mt-0.5 text-xs text-muted-foreground">{src.detail}</p>}
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Data quality">
          {!s.data ? (
            <p className="text-sm text-muted-foreground">No quality checks in the latest run.</p>
          ) : (
            <>
              <p className="flex items-baseline gap-2">
                <span className="text-4xl leading-none">{fmtPct(s.data.quality.score)}</span>
                <span className="text-xs text-muted-foreground">of weighted checks passed</span>
              </p>
              <ul className="mt-3 space-y-1.5 text-sm">
                {checks.slice(0, 6).map((c) => (
                  <li key={c.name} className="flex items-start justify-between gap-3">
                    <Status ok={c.passed} label={humanize(c.name)} />
                    <span className="text-right text-xs text-muted-foreground">{c.detail}</span>
                  </li>
                ))}
              </ul>
              {checks.length > 6 && <p className="mt-2 text-xs text-muted-foreground">+ {checks.length - 6} more passed checks</p>}
            </>
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <Panel title="Open warnings" action={<Link href="/intel/warnings" className="text-xs text-primary hover:underline">Triage queue →</Link>}>
          <p className="flex items-baseline gap-2">
            <span className="text-4xl leading-none">{s.warnings_open.total.toLocaleString()}</span>
            <span className="text-xs text-muted-foreground">open (new, acknowledged or investigating)</span>
          </p>
          <ul className="mt-4 grid grid-cols-2 gap-2">
            {SEVERITIES.map((sev) => (
              <li key={sev} className="flex items-center justify-between gap-2 rounded border hairline px-2.5 py-2">
                <SeverityBadge severity={sev} className="border-0 px-0" />
                <span className="num text-lg">{s.warnings_open.by_severity[sev] ?? 0}</span>
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Top risks" className="lg:col-span-2" action={<Link href="/intel/risks" className="text-xs text-primary hover:underline">Risk register →</Link>}>
          {!s.top_risks.length ? (
            <p className="text-sm text-muted-foreground">No risks assessed in the latest run.</p>
          ) : (
            <ul className="divide-y hairline">
              {s.top_risks.slice(0, 5).map((r) => (
                <li key={r.id} className="grid grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1 py-2.5 first:pt-0 last:pb-0 sm:grid-cols-[1fr_120px_auto]">
                  <Link href={`/intel/risks#${r.id}`} className="min-w-0 text-sm hover:underline">
                    {r.title}
                    <span className="block font-mono text-xs text-muted-foreground">{humanize(r.kind)} · {r.entity}</span>
                  </Link>
                  <span className="hidden sm:block"><Meter value={r.score / 100} label={`Risk score ${r.score.toFixed(0)} of 100`} digits={2} /></span>
                  <SeverityBadge severity={r.level} />
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Latest signals" action={<Link href="/intel/signals" className="text-xs text-primary hover:underline">All signals →</Link>}>
          {!s.top_signals.length ? (
            <p className="text-sm text-muted-foreground">No signals in the latest run.</p>
          ) : (
            <ul className="divide-y hairline">
              {s.top_signals.slice(0, 6).map((g) => (
                <li key={g.id} className="grid grid-cols-[auto_1fr_96px] items-center gap-3 py-2.5 first:pt-0 last:pb-0">
                  <DirectionIcon direction={g.direction} />
                  <span className="min-w-0 text-sm">
                    <span className="line-clamp-2 block break-words">{g.title}</span>
                    <span className="block font-mono text-xs text-muted-foreground">{humanize(g.kind)} · {g.entity}</span>
                  </span>
                  <Meter value={g.strength} label={`Strength ${g.strength.toFixed(2)}`} />
                </li>
              ))}
            </ul>
          )}
        </Panel>

        <Panel title="Recent decisions" action={<Link href="/intel/decisions" className="text-xs text-primary hover:underline">Decision log →</Link>}>
          {!s.recent_decisions.length ? (
            <p className="text-sm text-muted-foreground">No decisions recorded yet.</p>
          ) : (
            <ul className="divide-y hairline">
              {s.recent_decisions.slice(0, 6).map((d) => (
                <li key={`${d.id}-${d.db_id ?? ""}`} className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 py-2.5 first:pt-0 last:pb-0">
                  <Link href={d.db_id !== undefined ? `/intel/decisions/${d.db_id}` : "/intel/decisions"} className="min-w-0 flex-1 text-sm hover:underline">
                    {d.question}
                  </Link>
                  <span className="flex items-center gap-2">
                    <span className="text-sm">{d.abstained ? <span className="text-muted-foreground">abstained</span> : fmtAnswer(d)}</span>
                    {!d.abstained && <ConfidenceBadge value={d.confidence} kind={d.confidence_kind} interval={d.answer_interval} />}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Decision confidence, latest run">
          {!s.confidence_histogram.length || s.confidence_histogram.every((b) => b.n === 0) ? (
            <p className="text-sm text-muted-foreground">No decisions to distribute yet.</p>
          ) : (
            <>
              <CountBars data={s.confidence_histogram} />
              <p className="mt-2 text-xs text-muted-foreground">
                Counts of decisions per confidence bin. Bins mix kinds (probability, margin, rule and interval coverage), so read this as a spread,
                not as calibration.
              </p>
              <TableView caption="Decisions per confidence bin" head={["bin", "decisions"]} rows={s.confidence_histogram.map((b) => [b.bin, b.n])} />
            </>
          )}
        </Panel>
        <Panel title="Stage timings, latest run">
          {!stages.rows.length ? (
            <p className="text-sm text-muted-foreground">No stage timings recorded.</p>
          ) : (
            <>
              <NamedBars rows={stages.rows} format={fmtMs} />
              <p className="mt-3 text-xs text-muted-foreground">
                In pipeline order. Pipeline total {fmtMs(stages.total)}; the whole run took {fmtMs(run.duration_ms)} wall time.
              </p>
            </>
          )}
        </Panel>
      </div>

      <p className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
        Severity key:
        {SEVERITIES.map((sev) => {
          const m = SEVERITY_META[sev];
          return (
            <span key={sev} className="inline-flex items-center gap-1">
              <m.Icon className="size-3.5" style={{ color: m.color }} aria-hidden /> {m.label}
            </span>
          );
        })}
      </p>
    </div>
  );
}
