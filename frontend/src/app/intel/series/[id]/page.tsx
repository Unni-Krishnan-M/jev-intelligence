"use client";

import { useParams } from "next/navigation";
import { useState } from "react";
import useSWR from "swr";

import { Panel, SpecRows } from "@/components/jev/admin/ui";
import { DirectionIcon, SeverityBadge } from "@/components/jev/intel/badges";
import { TableView, TimeSeriesChart } from "@/components/jev/intel/charts";
import Link, { useIntelDomain } from "@/components/jev/intel/domain-context";
import { HistoryTimeline } from "@/components/jev/intel/history-timeline";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ApiError } from "@/lib/api";
import { fmtMonth, fmtP, fmtSeriesValue, fmtSigned, humanize, seriesLabel } from "@/lib/intel";
import type { SeriesDetail } from "@/lib/intel-types";

function decodeParam(v: string | string[] | undefined): string {
  const s = Array.isArray(v) ? v.join("/") : (v ?? "");
  try {
    return decodeURIComponent(s);
  } catch {
    return s;
  }
}

export default function SeriesPage() {
  const params = useParams<{ id: string }>();
  const id = decodeParam(params.id);
  const { q: dq } = useIntelDomain();
  const { data, error, mutate } = useSWR<SeriesDetail>(id ? dq(`/intel/series/${encodeURIComponent(id)}`) : null);
  const [showForecast, setShowForecast] = useState(true);

  if (error instanceof ApiError && error.status === 404) {
    return (
      <EmptyState
        title="Series not available"
        body={`“${id}” is not in the latest run, or no run exists yet. Series ids change only when the catalogue's genres change.`}
        action={<Button asChild variant="outline"><Link href="/intel/trends">Back to trends</Link></Button>}
      />
    );
  }

  const t = data?.trend ?? null;
  const metric = data?.series.metric ?? id.split(":")[0];
  const history = data?.series.points ?? [];
  const markers = (data?.anomalies ?? [])
    .filter((a) => a.value !== null && !a.suppressed)
    .map((a) => ({ t: a.detected_at, v: a.value as number, severity: a.severity, label: `robust z ${a.score.toFixed(1)}` }));
  const fc = showForecast ? data?.forecast ?? null : null;

  return (
    <div>
      <PageHeader
        eyebrow="series"
        title={seriesLabel(id)}
        description={
          data ? (
            <>
              Unit: {data.series.unit}.{t ? " The shaded span is the trend window." : ""}
              {t?.change_point ? " The vertical rule marks a detected change point." : ""}
            </>
          ) : undefined
        }
        asOf={data?.as_of}
        runId={data?.run_id}
        action={<Button asChild variant="outline" size="sm"><Link href={`/intel/scenarios?series=${encodeURIComponent(id)}`}>What-if on this series</Link></Button>}
      />

      {error ? (
        <IntelError error={error} retry={() => mutate()} />
      ) : !data ? (
        <div className="space-y-4">
          <Skeleton className="h-[320px] w-full" />
          <Skeleton className="h-48 w-full" />
        </div>
      ) : (
        <div className="space-y-6">
          <Panel
            title={`${seriesLabel(id)} · monthly`}
            action={
              data.forecast ? (
                <label className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Switch checked={showForecast} onCheckedChange={setShowForecast} aria-label="Show forecast" /> Forecast
                </label>
              ) : undefined
            }
          >
            {history.length < 2 ? (
              <p className="text-sm text-muted-foreground">Not enough points to draw this series.</p>
            ) : (
              <>
                <TimeSeriesChart
                  history={history}
                  metric={metric}
                  window={t ? t.window : null}
                  changePoint={t?.change_point ? { t: t.change_point.date, label: "change point" } : null}
                  markers={markers}
                  band={fc?.points}
                  means={fc ? [{ key: "mean", label: "Forecast mean (estimate)", color: "var(--sig-content)", dashed: true, points: fc.points }] : []}
                  height={320}
                />
                <TableView
                  caption={`${seriesLabel(id)} by month`}
                  head={["month", "value", "partial"]}
                  rows={history.map((p) => [p.t.slice(0, 7), fmtSeriesValue(p.v, metric), p.partial ? "yes" : ""])}
                />
              </>
            )}
          </Panel>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Panel title="Trend test">
              {!t ? (
                <p className="text-sm text-muted-foreground">No trend was tested on this series (too short or too sparse).</p>
              ) : (
                <SpecRows
                  rows={[
                    { label: "Direction", value: <DirectionIcon direction={t.direction} withLabel /> },
                    { label: "Window", value: `${fmtMonth(t.window.start)} → ${fmtMonth(t.window.end)} (${t.window.months} m)` },
                    { label: "Theil–Sen slope", value: t.slope.toPrecision(3) },
                    { label: "95 % interval", value: `[${t.slope_ci[0].toPrecision(3)}, ${t.slope_ci[1].toPrecision(3)}]` },
                    { label: "Change per month", value: fmtSigned(t.change_rate_pct_per_month, 2, " %") },
                    { label: "Kendall τ", value: t.kendall_tau.toFixed(3) },
                    { label: "p-value (Mann–Kendall)", value: fmtP(t.p_value) },
                    { label: "Evidence strength (1 − p)", value: <span title="Not a probability that the trend is real">{t.evidence_strength.toFixed(3)} · not a probability</span> },
                  ]}
                />
              )}
              {t && <HistoryTimeline entity="trends" historyKey={t.series_id} labels={{ score: "evidence (1 − p)", value: "slope" }} className="mt-4" />}
            </Panel>
            <Panel title="Recent vs prior window">
              {!t ? (
                <p className="text-sm text-muted-foreground">No comparison without a trend window.</p>
              ) : (
                <>
                  <p className="text-sm leading-relaxed text-ink-2">
                    The last {t.window.months} months averaged <span className="num text-foreground">{fmtSeriesValue(t.recent_mean, metric)}</span>, against{" "}
                    <span className="num text-foreground">{fmtSeriesValue(t.prior_mean, metric)}</span> in the {t.window.months} months before
                    {t.ratio !== null && <> — <span className="num text-foreground">×{t.ratio.toFixed(2)}</span></>}.
                  </p>
                  <div className="mt-4 space-y-2" aria-hidden>
                    {[["Prior", t.prior_mean], ["Recent", t.recent_mean]].map(([label, v]) => (
                      <div key={label as string} className="grid grid-cols-[56px_1fr] items-center gap-2 text-xs text-muted-foreground">
                        {label}
                        <span className="h-2">
                          <span
                            className={`block h-full rounded-r-[4px] ${label === "Recent" ? "bg-primary" : "bg-muted-foreground/40"}`}
                            style={{ width: `${((v as number) / Math.max(t.recent_mean, t.prior_mean, 1e-9)) * 100}%` }}
                          />
                        </span>
                      </div>
                    ))}
                  </div>
                  {t.change_point && (
                    <div className="mt-5 border-t hairline pt-3">
                      <p className="eyebrow mb-1">Change point</p>
                      <SpecRows
                        rows={[
                          { label: "Date", value: fmtMonth(t.change_point.date) },
                          { label: "Mean before", value: fmtSeriesValue(t.change_point.before_mean, metric) },
                          { label: "Mean after", value: fmtSeriesValue(t.change_point.after_mean, metric) },
                          { label: "p-value", value: fmtP(t.change_point.p_value) },
                        ]}
                      />
                    </div>
                  )}
                </>
              )}
            </Panel>
          </div>

          <Panel title="Anomalies on this series">
            {data.anomalies.length === 0 ? (
              <p className="text-sm text-muted-foreground">None detected.</p>
            ) : (
              <ul className="divide-y hairline text-sm">
                {data.anomalies.map((a) => (
                  <li key={a.id} className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 py-2">
                    <span>
                      {fmtMonth(a.detected_at)} · {humanize(a.kind)}
                      <span className="block font-mono text-xs text-muted-foreground">
                        {fmtSeriesValue(a.value, metric)} vs baseline {fmtSeriesValue(a.baseline, metric)} · robust z {a.score.toFixed(1)}
                        {a.suppressed && ` · suppressed: ${a.suppression_reason ?? "no reason given"}`}
                      </span>
                    </span>
                    <SeverityBadge severity={a.severity} />
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          {data.forecast && (
            <p className="text-xs text-muted-foreground">
              Forecast by {data.forecast.model} ({data.forecast.model_version}), {data.forecast.horizon_months} months ahead, 80 % interval. It is an estimate;
              see <Link href="/intel/predictions" className="text-primary hover:underline">Predictions</Link> for its backtest.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
