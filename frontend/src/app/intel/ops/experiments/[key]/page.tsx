"use client";

import { AlertTriangle, ArrowLeft, Star } from "lucide-react";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

import { fmtDate, Panel, SpecRows, Status } from "@/components/jev/admin/ui";
import Link from "@/components/jev/intel/domain-context";
import { IntelError, PanelsSkeleton } from "@/components/jev/intel/states";
import { ConclusionBadge, ExperimentStatusBadge, GuardrailBadge } from "@/components/jev/ops/badges";
import { LabelBanner } from "@/components/jev/ops/label-banner";
import { ConfirmDialog } from "@/components/jev/ops/confirm-dialog";
import { IntervalKey, IntervalPlot, intervalDomain } from "@/components/jev/ops/interval-plot";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError, errorMessage } from "@/lib/api";
import { fmtP } from "@/lib/intel";
import { conclusionWording, EXPERIMENT_ACTIONS, fmtMetric, fmtSrm, lifecycleActions, MEAN_METRICS, METRIC_LABEL, RATE_METRICS, srmRows } from "@/lib/ops";
import type { ExperimentOut, ExperimentResults, SrmTest, VariantConfig } from "@/lib/ops-types";
import { cn } from "@/lib/utils";

const METRICS = [...RATE_METRICS, ...MEAN_METRICS] as const;

function configText(c: VariantConfig): string {
  const bits: string[] = [];
  for (const [k, v] of Object.entries(c.hybrid_overrides ?? {})) bits.push(`${k === "diversity_lambda" ? "MMR λ" : k} ${JSON.stringify(v)}`);
  if (c.recency_half_life_days) bits.push(`recency half-life ${c.recency_half_life_days} d`);
  if (c.strategy_decision === false) bits.push("strategy off");
  if (c.model_version) bits.push(`model ${c.model_version}`);
  return bits.join(" · ") || "champion as served";
}

function Lifecycle({ exp, onDone }: { exp: ExperimentOut; onDone: () => Promise<unknown> }) {
  const router = useRouter();
  const [action, setAction] = useState<keyof typeof EXPERIMENT_ACTIONS | null>(null);
  const [ramp, setRamp] = useState(String(exp.traffic_percent));
  const [rampOpen, setRampOpen] = useState(false);
  const actions = lifecycleActions(exp.allowed_actions);
  const canRamp = exp.allowed_actions.includes("ramp");
  const rampVal = Number(ramp);
  const rampOk = ramp.trim() !== "" && rampVal >= 0 && rampVal <= 100 && rampVal !== exp.traffic_percent;

  async function run() {
    if (!action) return;
    try {
      if (action === "delete") {
        await api(`/experiments/online/${encodeURIComponent(exp.key)}`, { method: "DELETE" });
        toast.success(`Deleted ${exp.key}`);
        router.push("/intel/ops/experiments");
        return;
      }
      await api<ExperimentOut>(`/experiments/online/${encodeURIComponent(exp.key)}/${action}`, { method: "POST" });
      toast.success(`${EXPERIMENT_ACTIONS[action].label}: ${exp.key}`);
      setAction(null);
      await onDone();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }
  async function doRamp() {
    try {
      await api<ExperimentOut>(`/experiments/online/${encodeURIComponent(exp.key)}/ramp`, { json: { traffic_percent: rampVal } });
      toast.success(`Traffic set to ${rampVal} %`);
      setRampOpen(false);
      await onDone();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  return (
    <Panel title="Lifecycle">
      {actions.length ? (
        <div className="flex flex-wrap gap-2">
          {actions.map((a) => (
            <Button key={a} size="sm" variant={EXPERIMENT_ACTIONS[a].destructive ? "outline" : a === "start" ? "default" : "outline"} className={cn(EXPERIMENT_ACTIONS[a].destructive && "text-destructive")} onClick={() => setAction(a)}>
              {EXPERIMENT_ACTIONS[a].label}
            </Button>
          ))}
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">Concluded: no further actions.</p>
      )}
      {canRamp && (
        <form className="mt-4 flex flex-wrap items-end gap-2" onSubmit={(e) => { e.preventDefault(); if (rampOk) setRampOpen(true); }}>
          <div className="space-y-1">
            <Label htmlFor="ramp" className="text-xs font-normal text-muted-foreground">Traffic enrolled (%) · now {exp.traffic_percent} %</Label>
            <Input id="ramp" type="number" min={0} max={100} step={1} value={ramp} onChange={(e) => setRamp(e.target.value)} className="num h-8 w-28" />
          </div>
          <Button size="sm" variant="outline" type="submit" disabled={!rampOk}>Ramp…</Button>
        </form>
      )}
      <p className="mt-3 text-xs text-muted-foreground">Enrolment is sticky: a ramp down stops new enrolment; members already in keep their variant.</p>
      <ConfirmDialog
        open={action !== null}
        onOpenChange={(o) => !o && setAction(null)}
        title={action ? `${EXPERIMENT_ACTIONS[action].label} ${exp.key}?` : ""}
        description={action ? EXPERIMENT_ACTIONS[action].confirm : ""}
        confirmLabel={action ? EXPERIMENT_ACTIONS[action].label : "Confirm"}
        destructive={action ? Boolean(EXPERIMENT_ACTIONS[action].destructive) : false}
        onConfirm={run}
      />
      <ConfirmDialog
        open={rampOpen}
        onOpenChange={setRampOpen}
        title={`Ramp ${exp.key} to ${rampVal} %?`}
        description={`From ${exp.traffic_percent} % to ${rampVal} % of members on the surface. Audited as experiment.ramp.`}
        confirmLabel="Ramp"
        onConfirm={doRamp}
      />
    </Panel>
  );
}

function SrmTable({ title, names, t }: { title: string; names: string[]; t: SrmTest }) {
  const rows = srmRows(names, t);
  return (
    <div>
      <p className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <span className="font-medium">{title}</span>
        <Status ok={!t.detected} label={t.detected ? "Mismatch" : "No mismatch"} />
      </p>
      <p className="num mt-0.5 text-xs text-muted-foreground">{fmtSrm(t)}</p>
      <table className="mt-2 w-full text-sm">
        <caption className="sr-only">{title}: observed and expected members per variant</caption>
        <thead>
          <tr className="border-b hairline text-left text-xs text-muted-foreground">
            <th className="py-1.5 font-normal">Variant</th>
            <th className="py-1.5 text-right font-normal">Observed</th>
            <th className="py-1.5 text-right font-normal">Expected</th>
            <th className="py-1.5 text-right font-normal">Δ</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className="border-b hairline last:border-0">
              <td className="py-1.5 font-mono text-xs">{r.name}</td>
              <td className="num py-1.5 text-right">{r.observed.toLocaleString("en")}</td>
              <td className="num py-1.5 text-right">{r.expected === null ? "—" : r.expected.toLocaleString("en", { maximumFractionDigits: 1 })}</td>
              <td className="num py-1.5 text-right text-muted-foreground">{r.deltaPct === null ? "—" : `${r.deltaPct >= 0 ? "+" : "−"}${Math.abs(r.deltaPct * 100).toFixed(1)} %`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Results({ r }: { r: ExperimentResults }) {
  const names = r.variants.map((v) => v.name);
  const treatments = Object.keys(r.comparisons);
  const w = conclusionWording(r.conclusion);
  const primaryRows = treatments.map((t) => {
    const c = r.comparisons[t][r.primary_metric];
    return { t, c, row: { diff: c?.diff ?? null, lo: c?.ci[0] ?? null, hi: c?.ci[1] ?? null } };
  });
  const primaryDomain = intervalDomain(primaryRows.map((x) => x.row));

  return (
    <div className="space-y-4">
      {/* conclusion first: what may be done with this */}
      <section aria-labelledby="concl-h" className={cn("rounded-lg border bg-card p-4 sm:p-5", w.tone === "good" && "border-primary/50")}>
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 id="concl-h" className="font-display text-2xl">{r.status === "concluded" ? "Conclusion" : "Conclusion if stopped now"}</h2>
          <ConclusionBadge conclusion={r.conclusion} />
        </div>
        <p className="mt-2 text-sm text-ink-2">{w.body}</p>
        {r.conclusion.reasons.length > 0 && (
          <ul className="mt-3 space-y-1 text-sm">
            {r.conclusion.reasons.map((x) => <li key={x} className="border-l-2 border-rule pl-3 font-mono text-xs leading-relaxed">{x}</li>)}
          </ul>
        )}
      </section>

      {r.warnings.length > 0 && (
        <ul aria-label="Sample-size warnings" className="space-y-1.5 rounded-lg border border-dashed px-4 py-3 text-sm">
          {r.warnings.map((x) => (
            <li key={x} className="flex items-start gap-2"><AlertTriangle className="mt-0.5 size-4 shrink-0" style={{ color: "var(--status-warning)" }} aria-hidden />{x}</li>
          ))}
        </ul>
      )}

      <Panel title={`Per variant · ${r.totals.assigned_users.toLocaleString("en")} assigned · ${r.totals.exposures.toLocaleString("en")} exposures · ${r.totals.outcomes.toLocaleString("en")} outcomes`}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[900px] whitespace-nowrap text-sm">
            <caption className="sr-only">Metrics per variant; the unit of analysis is the member</caption>
            <thead>
              <tr className="border-b hairline text-left text-xs text-muted-foreground">
                <th className="py-2 pr-3 font-normal">Variant</th>
                <th className="px-2 py-2 text-right font-normal">Members</th>
                {METRICS.map((m) => <th key={m} className={cn("px-2 py-2 text-right font-normal", m === r.primary_metric && "text-foreground")}>{m === r.primary_metric && <Star className="mr-1 inline size-3 text-primary" aria-label="primary metric" />}{METRIC_LABEL[m]}</th>)}
                <th className="px-2 py-2 text-right font-normal">Coverage</th>
                <th className="py-2 pl-2 text-right font-normal">p95 ms</th>
              </tr>
            </thead>
            <tbody>
              {r.variants.map((v) => (
                <tr key={v.name} className="border-b hairline last:border-0">
                  <td className="whitespace-normal py-2 pr-3"><span className="font-mono text-xs">{v.name}</span>{v.is_control && <span className="text-xs text-muted-foreground"> · control</span>}<span className="block text-xs text-muted-foreground">{configText(v.config)}</span></td>
                  <td className="num px-2 py-2 text-right">{v.exposed_users.toLocaleString("en")}<span className="block text-xs text-muted-foreground">of {v.assigned_users.toLocaleString("en")}</span></td>
                  {METRICS.map((m) => {
                    const val = (RATE_METRICS as readonly string[]).includes(m) ? v.rates[m as (typeof RATE_METRICS)[number]]?.rate : v.means[m as (typeof MEAN_METRICS)[number]]?.mean;
                    return <td key={m} className={cn("num px-2 py-2 text-right", m === r.primary_metric && "bg-primary/5 font-medium")}>{fmtMetric(m, val)}</td>;
                  })}
                  <td className="num px-2 py-2 text-right">{fmtMetric("coverage", v.coverage.value)}</td>
                  <td className="num py-2 pl-2 text-right">{v.latency_ms.p95 === null ? "—" : v.latency_ms.p95.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      <Panel title={`Treatment − control · α ${r.alpha} (primary ${r.alpha_primary.toFixed(3)})`}>
        {!treatments.length ? (
          <p className="text-sm text-muted-foreground">No treatment to compare.</p>
        ) : (
          <>
            <div className="mb-4 space-y-2">
              <p className="eyebrow">{METRIC_LABEL[r.primary_metric]} · primary</p>
              {primaryRows.map(({ t, c, row }) => (
                <div key={t} className="grid grid-cols-1 items-center gap-x-4 gap-y-1 sm:grid-cols-[120px_minmax(0,1fr)_minmax(0,240px)]">
                  <span className="font-mono text-xs">{t}</span>
                  <IntervalPlot row={row} domain={primaryDomain} format={(v) => fmtMetric(r.primary_metric, v, true)} label={`${t} ${r.primary_metric}`} highlight />
                  <span className="num text-xs text-ink-2">{fmtMetric(r.primary_metric, c?.diff, true)} · p {fmtP(c?.p_value)} · {c?.significant ? "significant" : "not significant"}</span>
                </div>
              ))}
              <IntervalKey />
            </div>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[760px] whitespace-nowrap text-sm">
                <caption className="sr-only">Every metric, each treatment against the control</caption>
                <thead>
                  <tr className="border-b hairline text-left text-xs text-muted-foreground">
                    <th className="py-2 pr-3 font-normal">Metric</th>
                    <th className="px-2 py-2 font-normal">Treatment</th>
                    <th className="px-2 py-2 text-right font-normal">Δ</th>
                    <th className="px-2 py-2 text-right font-normal">Relative</th>
                    <th className="px-2 py-2 text-right font-normal">CI</th>
                    <th className="px-2 py-2 text-right font-normal">p</th>
                    <th className="py-2 pl-2 font-normal">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {METRICS.flatMap((m) =>
                    treatments.map((t) => {
                      const c = r.comparisons[t][m];
                      if (!c) return null;
                      const primary = m === r.primary_metric;
                      return (
                        <tr key={`${m}-${t}`} className={cn("border-b hairline last:border-0", primary && "bg-primary/5")}>
                          <td className={cn("py-2 pr-3", primary && "font-medium")}>{primary && <Star className="mr-1 inline size-3 text-primary" aria-label="primary metric" />}{METRIC_LABEL[m]}</td>
                          <td className="px-2 py-2 font-mono text-xs">{t}</td>
                          <td className="num px-2 py-2 text-right">{fmtMetric(m, c.diff, true)}</td>
                          <td className="num px-2 py-2 text-right text-muted-foreground">{c.relative_lift === null ? "—" : `${c.relative_lift >= 0 ? "+" : "−"}${Math.abs(c.relative_lift * 100).toFixed(1)} %`}</td>
                          <td className="num px-2 py-2 text-right text-xs">{c.ci[0] === null || c.ci[1] === null ? "—" : `[${fmtMetric(m, c.ci[0], true)}, ${fmtMetric(m, c.ci[1], true)}]`}</td>
                          <td className="num px-2 py-2 text-right">{fmtP(c.p_value)}</td>
                          <td className="py-2 pl-2 text-xs">{c.significant ? <Status ok={(c.diff ?? 0) > 0 === (m !== "negative_rate")} label="significant" /> : <span className="text-muted-foreground">not significant</span>}</td>
                        </tr>
                      );
                    }),
                  )}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">Rates are two-proportion z-tests on members; NDCG, diversity and novelty use a member bootstrap. Only the primary metric decides; the rest are descriptive.</p>
          </>
        )}
      </Panel>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Sample ratio mismatch">
          {r.srm.detected && <p role="alert" className="mb-3 text-sm"><Status ok={false} label="Detected: the comparison must not be trusted" /></p>}
          <div className="space-y-5">
            <SrmTable title="Assigned members" names={names} t={r.srm.assigned} />
            <SrmTable title="Exposed members" names={names} t={r.srm.exposed} />
          </div>
          <p className="mt-3 text-xs text-muted-foreground">Chi-square goodness of fit of arm sizes to the weights. A mismatch means assignment or exposure logging is broken.</p>
        </Panel>
        <div className="space-y-4">
          <Panel title="Guardrails">
            {!r.guardrails.length ? (
              <p className="text-sm text-muted-foreground">No guardrails configured.</p>
            ) : (
              <ul className="divide-y hairline">
                {r.guardrails.map((g) => (
                  <li key={`${g.metric}-${g.variant}`} className="py-2.5 first:pt-0 last:pb-0">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <p className="text-sm">{METRIC_LABEL[g.metric] ?? g.metric} <span className="font-mono text-xs text-muted-foreground">· {g.variant}</span></p>
                      <GuardrailBadge breached={g.breached} />
                    </div>
                    <p className="num text-xs text-muted-foreground">
                      control {fmtMetric(g.metric, g.control_value)} · treatment {fmtMetric(g.metric, g.treatment_value)} · {Object.entries(g.threshold).map(([k, v]) => `${k.replaceAll("_", " ")} ${v}`).join(", ")}
                    </p>
                    {g.reason && <p className="text-xs text-destructive">{g.reason}</p>}
                  </li>
                ))}
              </ul>
            )}
          </Panel>
          <Panel title="Sample size">
            <SpecRows
              rows={[
                { label: "Smallest variant", value: r.sample_size.smallest_variant_users.toLocaleString("en") },
                { label: "Minimum per variant", value: r.sample_size.min_users_per_variant.toLocaleString("en") },
                { label: `Needed per variant (${(r.sample_size.mde_relative * 100).toFixed(0)} % MDE)`, value: r.sample_size.required_per_variant === null ? "not estimable" : r.sample_size.required_per_variant.toLocaleString("en") },
                { label: "Attribution window", value: `${r.attribution_window_hours} h` },
                { label: "Computed", value: fmtDate(r.computed_at) },
              ]}
            />
          </Panel>
        </div>
      </div>
    </div>
  );
}

export default function ExperimentDetailPage() {
  const { key } = useParams<{ key: string }>();
  const path = `/experiments/online/${encodeURIComponent(key)}`;
  const exp = useSWR<ExperimentOut>(key ? path : null);
  const live = exp.data?.status === "running" || exp.data?.status === "paused";
  const res = useSWR<ExperimentResults>(exp.data && exp.data.status !== "draft" ? `${path}/results` : null, { refreshInterval: live ? 20000 : 0 });
  const back = (
    <Link href="/intel/ops/experiments" className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
      <ArrowLeft className="size-4" aria-hidden /> Experiments
    </Link>
  );
  if (exp.error instanceof ApiError && exp.error.status === 404) return <div>{back}<EmptyState title="Experiment not found" body={`No online experiment with key ${key}.`} /></div>;
  if (exp.error) return <div>{back}<IntelError error={exp.error} retry={() => exp.mutate()} runBacked={false} /></div>;
  const e = exp.data;
  if (!e) return <div className="space-y-4">{back}<Skeleton className="h-12 w-2/3" /><PanelsSkeleton /></div>;

  return (
    <div>
      {back}
      <header className="mb-6 border-b hairline pb-6">
        <p className="eyebrow">Experiment · {e.key} · {e.surface}</p>
        <h1 className="font-display mt-2 text-[34px] leading-[1.02] tracking-tight text-balance sm:text-[44px]">{e.name}</h1>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-sm">
          <ExperimentStatusBadge status={e.status} />
          <span className="text-muted-foreground">primary {METRIC_LABEL[e.primary_metric]} · {e.traffic_percent} % traffic · {e.variants.length} variants</span>
        </div>
        {e.hypothesis && <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-ink-2">{e.hypothesis}</p>}
      </header>

      <div className="mb-6">
        <LabelBanner dataSource={res.data?.data_source ?? e.data_source} label={res.data?.label ?? (e.data_source === "live" ? "live traffic" : null)} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0 order-2 lg:order-1">
          {e.status === "draft" ? (
            <EmptyState title="Not started" body="A draft serves nobody and has no results. Start it to begin assigning members." />
          ) : res.error ? (
            <IntelError error={res.error} retry={() => res.mutate()} runBacked={false} />
          ) : !res.data ? (
            <PanelsSkeleton n={2} className="lg:grid-cols-1" />
          ) : (
            <Results r={res.data} />
          )}
        </div>
        <div className="order-1 space-y-4 lg:order-2">
          <Lifecycle exp={e} onDone={() => Promise.all([exp.mutate(), res.mutate()])} />
          <Panel title="Variants">
            <ul className="divide-y hairline text-sm">
              {e.variants.map((v) => (
                <li key={v.name} className="py-2 first:pt-0 last:pb-0">
                  <p className="flex items-baseline justify-between gap-2"><span className="font-mono text-xs">{v.name}{v.is_control ? " · control" : ""}</span><span className="num text-xs text-muted-foreground">weight {v.weight} · {v.assigned_users} members</span></p>
                  <p className="text-xs text-ink-2">{configText(v.config)}</p>
                  {v.description && <p className="text-xs text-muted-foreground">{v.description}</p>}
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title="Timeline">
            <SpecRows
              rows={[
                { label: "Created", value: fmtDate(e.created_at) },
                { label: "Started", value: e.started_at ? fmtDate(e.started_at) : "—" },
                { label: "Stopped", value: e.stopped_at ? fmtDate(e.stopped_at) : "—" },
                { label: "Concluded", value: e.concluded_at ? fmtDate(e.concluded_at) : "—" },
              ]}
            />
          </Panel>
        </div>
      </div>
    </div>
  );
}
