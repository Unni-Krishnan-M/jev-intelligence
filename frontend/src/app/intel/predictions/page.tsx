"use client";

import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";

import { fmtDate, fmtNum, Panel, SpecRows, StatTile } from "@/components/jev/admin/ui";
import { FanChart, ReliabilityDiagram, TableView } from "@/components/jev/intel/charts";
import { FeedbackButtons } from "@/components/jev/intel/feedback-buttons";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, PanelsSkeleton } from "@/components/jev/intel/states";
import { EmptyState, SectionHeader } from "@/components/jev/states";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { fmtValue, fmtMonth, fmtPct, fmtSeriesValue, humanize, seriesHref, seriesLabel } from "@/lib/intel";
import type { Forecast, Lapse, PredictionsResponse, SeriesDetail } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

function ForecastView({ f }: { f: Forecast }) {
  const { data, error } = useSWR<SeriesDetail>(`/intel/series/${encodeURIComponent(f.series_id)}`);
  const history = (data?.series.points ?? []).slice(-36);
  const b = f.backtest;
  const beats = b.mase !== null && b.naive_mase !== null ? b.mase < b.naive_mase : null;

  return (
    <div id={f.id} className="grid grid-cols-1 scroll-mt-24 gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
      <Panel title={`${seriesLabel(f.series_id)} · ${f.horizon_months}-month forecast`}>
        {error ? (
          <p className="text-sm text-muted-foreground">History for this series could not be loaded; the forecast table below still applies.</p>
        ) : !data ? (
          <Skeleton className="h-[280px] w-full" />
        ) : (
          <FanChart history={history} points={f.points} metric={f.metric} />
        )}
        <p className="mt-3 text-xs text-muted-foreground">
          The dashed line is the model&apos;s central estimate. If the model is well calibrated, about one month in five should
          land outside the shaded 80 % band — these are estimates, not commitments.
        </p>
        <TableView
          caption="Forecast points"
          head={["month", "mean", "lo 80", "hi 80"]}
          rows={f.points.map((p) => [p.t.slice(0, 7), fmtSeriesValue(p.mean, f.metric), fmtSeriesValue(p.lo80, f.metric), fmtSeriesValue(p.hi80, f.metric)])}
        />
      </Panel>

      <div className="space-y-4">
        <Panel title="Backtest (rolling origin, ≤ as of)">
          <SpecRows
            rows={[
              { label: "MASE", value: fmtNum(b.mase, 3) },
              { label: "Naive MASE", value: fmtNum(b.naive_mase, 3) },
              { label: "vs naive", value: beats === null ? "—" : beats ? "beats naive" : "does not beat naive" },
              { label: "80 % coverage", value: `${fmtPct(b.coverage80)} (target 80 %)` },
              { label: "sMAPE", value: fmtNum(b.smape, 3) },
              { label: "MAE", value: fmtSeriesValue(b.mae, f.metric) },
              { label: "Origins", value: b.origins },
            ]}
          />
          <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
            MASE is scaled by the in-sample one-step naive error, so compare it with the naive MASE on the same points rather
            than with 1. This in-run backtest scores all {b.origins} origins with the model already chosen; the{" "}
            <Link href="/intel/evaluation" className="text-primary hover:underline">Evaluation</Link> report uses a nested protocol
            (chosen on origins 1–12, scored on 13–24), so its MASE and coverage differ.
          </p>
        </Panel>
        <Panel title="Model">
          <SpecRows
            rows={[
              { label: "Model", value: humanize(f.model) },
              { label: "Version", value: <span title={f.model_version}>{f.model_version}</span> },
              { label: "Horizon", value: `${f.horizon_months} months` },
              { label: "Issued at", value: fmtDate(f.issued_at) },
            ]}
          />
          <p className="eyebrow mt-3">Features used</p>
          <ul className="mt-1 space-y-0.5 text-xs text-ink-2">
            {f.features_used.map((x) => <li key={x} className="font-mono">{x}</li>)}
          </ul>
          <div className="mt-4 border-t hairline pt-3">
            <p className="mb-2 text-xs text-muted-foreground">Did this forecast hold up?</p>
            <FeedbackButtons key={f.id} targetType="prediction" targetId={f.id} verdicts={["correct", "incorrect"]} noteField="outcome" />
          </div>
        </Panel>
      </div>
    </div>
  );
}

/** "1998-01 → 2011-07 (27)" rather than a list that cannot fit a row. */
function cutoffs(c: string[]): string {
  if (!c.length) return "—";
  if (c.length === 1) return c[0].slice(0, 7);
  return `${c[0].slice(0, 7)} → ${c[c.length - 1].slice(0, 7)} (${c.length})`;
}

function LapseView({ lapse }: { lapse: Lapse }) {
  if (lapse.status === "insufficient_data") {
    return <EmptyState title="Not enough data for the lapse model" body={lapse.detail ?? "Too few raters with history before the training cut-offs."} />;
  }
  const m = lapse.metrics;
  const featureKeys = Array.from(new Set(lapse.top.flatMap((u) => Object.keys(u.features ?? {})))).slice(0, 5);
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <StatTile label="AUC" value={fmtNum(m.auc, 3)} hint={`baseline AUC ${fmtNum(m.baseline_auc, 3)}`} />
        <StatTile label="Brier score" value={fmtNum(m.brier, 3)} hint="lower is better" />
        <StatTile label="Calibration error" value={fmtNum(m.ece, 3)} hint="ECE, lower is better" />
        <StatTile label="Base rate" value={fmtPct(m.base_rate)} hint={`${m.n_train.toLocaleString()} train · ${m.n_test.toLocaleString()} test rows`} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Panel title="Reliability diagram (held-out cut-offs)">
          {lapse.calibration.length === 0 ? (
            <p className="text-sm text-muted-foreground">No calibration bins reported.</p>
          ) : (
            <>
              <ReliabilityDiagram bins={lapse.calibration} />
              <p className="mt-2 text-xs text-muted-foreground">Points on the diagonal mean a predicted 30 % lapsed about 30 % of the time. Above it, the model under-predicts.</p>
              <TableView
                caption="Calibration bins"
                head={["bin", "predicted", "observed", "n"]}
                rows={lapse.calibration.map((c) => [c.bin, c.predicted.toFixed(3), c.observed.toFixed(3), c.n.toLocaleString()])}
              />
            </>
          )}
        </Panel>
        <Panel title="Population">
          <SpecRows
            rows={[
              { label: "Question", value: `P(no rating in next ${lapse.horizon_days} days)` },
              { label: "Raters scored", value: lapse.population.n_scored.toLocaleString() },
              { label: "Expected lapses (sum of p)", value: lapse.population.expected_lapses.toFixed(1) },
              { label: `At or above ${lapse.population.threshold.toFixed(2)}`, value: lapse.population.high_risk.toLocaleString() },
              { label: "Model version", value: <span title={lapse.model_version}>{lapse.model_version}</span> },
              { label: "Train cut-offs", value: cutoffs(m.train_cutoffs) },
              { label: "Test cut-offs", value: cutoffs(m.test_cutoffs) },
            ]}
          />
          <p className="eyebrow mt-3">Features</p>
          <p className="mt-1 font-mono text-xs text-ink-2">{lapse.features.join(" · ")}</p>
        </Panel>
      </div>

      <Panel title="Highest estimated lapse risk">
        {lapse.top.length === 0 ? (
          <p className="text-sm text-muted-foreground">No raters scored.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm" style={{ minWidth: 280 + featureKeys.length * 110 }}>
              <caption className="sr-only">Raters with the highest estimated probability of lapsing</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-2 font-normal">User</th>
                  <th className="px-2 py-2 text-right font-normal">Estimated p</th>
                  {featureKeys.map((k) => <th key={k} className="px-2 py-2 text-right font-normal">{humanize(k)}</th>)}
                </tr>
              </thead>
              <tbody>
                {lapse.top.map((u) => (
                  <tr key={u.user_id} className="border-b hairline last:border-0">
                    <td className="num py-2 pr-2">user {u.user_id}</td>
                    <td className="num px-2 py-2 text-right">{u.p.toFixed(2)}</td>
                    {featureKeys.map((k) => <td key={k} className="num px-2 py-2 text-right text-ink-2">{fmtValue(u.features?.[k], k)}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-3 text-xs text-muted-foreground">
          Estimated probabilities from a calibrated classifier. A rater at 0.9 may well rate again; treat this as a
          ranking for outreach, not a verdict on any person.
        </p>
      </Panel>
    </div>
  );
}

export default function PredictionsPage() {
  const { data, error, mutate } = useSWR<PredictionsResponse>("/intel/predictions");
  const [picked, setPicked] = useState<string | null>(null);
  const forecasts = data?.forecasts ?? [];
  const current = forecasts.find((f) => f.id === picked) ?? forecasts[0];

  return (
    <div>
      <PageHeader
        eyebrow="anticipate"
        title="Predictions"
        description="Monthly forecasts with 80 % intervals and a rolling-origin backtest, and the audience-lapse model with its calibration. All of these are estimates."
        asOf={data?.as_of}
        runId={data?.run_id}
      />

      {error ? (
        <IntelError error={error} retry={() => mutate()} />
      ) : !data ? (
        <div className="space-y-4">
          <Skeleton className="h-[360px] w-full" />
          <PanelsSkeleton n={2} />
        </div>
      ) : (
        <div className="space-y-12">
          <section aria-labelledby="fc-heading">
            <SectionHeader
              id="fc-heading"
              kicker="forecasts"
              title="Where the series may go"
              action={
                forecasts.length > 1 ? (
                  <Select value={current?.id} onValueChange={setPicked}>
                    <SelectTrigger className="h-9 w-[240px] max-w-full" aria-label="Forecast series"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {forecasts.map((f) => <SelectItem key={f.id} value={f.id}>{seriesLabel(f.series_id)}</SelectItem>)}
                    </SelectContent>
                  </Select>
                ) : undefined
              }
            />
            {!current ? (
              <EmptyState title="No forecasts" body="The latest run produced no forecasts (series too short for a backtest)." />
            ) : (
              <div className="space-y-4">
                <ForecastView key={current.id} f={current} />
                {forecasts.length > 1 && (
                  <div className="overflow-x-auto rounded-lg border bg-card">
                    <table className="w-full min-w-[640px] text-sm">
                      <caption className="sr-only">All forecasts with backtest scores</caption>
                      <thead>
                        <tr className="border-b hairline text-left text-xs text-muted-foreground">
                          <th className="px-4 py-2.5 font-normal">Series</th>
                          <th className="px-2 py-2.5 font-normal">Model</th>
                          <th className="px-2 py-2.5 text-right font-normal">MASE</th>
                          <th className="px-2 py-2.5 text-right font-normal">Naive</th>
                          <th className="px-2 py-2.5 text-right font-normal">Coverage 80</th>
                          <th className="px-4 py-2.5 text-right font-normal">Next month (mean)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {forecasts.map((f) => (
                          <tr key={f.id} className={cn("border-b hairline last:border-0 hover:bg-accent/40", f.id === current.id && "bg-accent/60")}>
                            <td className="px-4 py-2">
                              <button type="button" className="text-left hover:underline" onClick={() => setPicked(f.id)} aria-pressed={f.id === current.id}>
                                {seriesLabel(f.series_id)}
                              </button>
                              <Link href={seriesHref(f.series_id)} className="ml-2 text-xs text-primary hover:underline">series</Link>
                            </td>
                            <td className="px-2 py-2 text-xs text-ink-2">{humanize(f.model)}</td>
                            <td className="num px-2 py-2 text-right">{fmtNum(f.backtest.mase, 2)}</td>
                            <td className="num px-2 py-2 text-right text-ink-2">{fmtNum(f.backtest.naive_mase, 2)}</td>
                            <td className="num px-2 py-2 text-right">{fmtPct(f.backtest.coverage80)}</td>
                            <td className="num px-4 py-2 text-right">
                              {f.points[0] ? <>{fmtSeriesValue(f.points[0].mean, f.metric)} <span className="text-xs text-muted-foreground">{fmtMonth(f.points[0].t)}</span></> : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </section>

          <section aria-labelledby="lapse-heading">
            <SectionHeader id="lapse-heading" kicker={data.lapse ? `lapse model · ${data.lapse.horizon_days}-day horizon` : "lapse model"} title="Who may stop rating" />
            {!data.lapse ? <EmptyState title="No lapse model in this run" /> : <LapseView lapse={data.lapse} />}
          </section>
        </div>
      )}
    </div>
  );
}
