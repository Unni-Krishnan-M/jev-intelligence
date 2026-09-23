"use client";

import Link from "next/link";
import useSWR from "swr";

import { fmtDate, fmtNum, Panel, SpecRows, StatTile } from "@/components/jev/admin/ui";
import { NamedBars, ReliabilityDiagram, TableView } from "@/components/jev/intel/charts";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, NotDeployedState, PanelsSkeleton, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState, SectionHeader } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { fmtMs, fmtPct, humanize, isNotDeployed, orderStages, seriesHref, seriesLabel } from "@/lib/intel";
import type { EvaluationReport, EvaluationResponse, EvaluationRunList } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

function Protocol({ text }: { text: string }) {
  return <p className="mb-3 max-w-3xl border-l-2 border-rule pl-3 text-sm leading-relaxed text-ink-2">{text}</p>;
}

/** A metric next to its baseline, with the better one set in the foreground. */
function VsCell({ v, base, higherIsBetter = true }: { v: number | null; base: number | null; higherIsBetter?: boolean }) {
  const better = v !== null && base !== null && (higherIsBetter ? v > base : v < base);
  return (
    <td className="num px-2 py-2 text-right">
      <span className={cn(better ? "font-semibold text-foreground" : "text-ink-2")}>{fmtNum(v, 3)}</span>
      <span className="block text-xs text-muted-foreground">base {fmtNum(base, 3)}</span>
    </td>
  );
}

function Report({ r }: { r: EvaluationReport }) {
  const fs = r.forecast.summary;
  const lm = r.lapse.metrics;
  const inj = r.anomaly.injection;
  const cp = r.change_point;
  const stages = orderStages(r.latency.stage_ms);

  return (
    <div className="space-y-12">
      <section aria-labelledby="ev-fc">
        <SectionHeader id="ev-fc" index={1} kicker="forecasts · rolling origin" title="Do the forecasts beat naive?" />
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatTile label="Series" value={fs.n_series.toLocaleString()} />
          <StatTile label="Median MASE" value={fmtNum(fs.median_mase, 2)} hint={fs.median_naive_mase != null ? `naive on the same points: ${fmtNum(fs.median_naive_mase, 2)}` : "compare with naive MASE per series"} />
          <StatTile label="Beating naive" value={fmtPct(fs.share_beating_naive)} hint="MASE below naive MASE, same points" />
          <StatTile label="Mean 80 % coverage" value={fmtPct(fs.mean_coverage80)} hint="target 80 %" />
        </div>
        <p className="mt-4 max-w-3xl text-xs leading-relaxed text-muted-foreground">
          Nested protocol: for each series the model is chosen on origins 1–12 and scored only on the later origins 13–24,
          so these numbers are out-of-sample. MASE is scaled by the in-sample one-step naive error, so a value above 1 is
          normal for bursty monthly counts; what matters is MASE against naive MASE on the same points (bold = better).
          The in-run backtest on <Link href="/intel/predictions" className="text-primary hover:underline">Predictions</Link> scores
          all 24 origins with the model already chosen, so its figures differ.
        </p>
        {r.forecast.per_series.length > 0 && (
          <div className="mt-3 overflow-x-auto rounded-lg border bg-card">
            <table className="w-full min-w-[680px] text-sm">
              <caption className="sr-only">Forecast backtest per series</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2.5 font-normal">Series</th>
                  <th className="px-2 py-2.5 font-normal">Model</th>
                  <th className="px-2 py-2.5 text-right font-normal">MASE (vs naive)</th>
                  <th className="px-2 py-2.5 text-right font-normal">sMAPE</th>
                  <th className="px-2 py-2.5 text-right font-normal">Coverage 80</th>
                  <th className="px-4 py-2.5 text-right font-normal">Origins</th>
                </tr>
              </thead>
              <tbody>
                {r.forecast.per_series.map((s) => (
                  <tr key={s.series_id} className="border-b hairline last:border-0">
                    <td className="px-4 py-2"><Link href={seriesHref(s.series_id)} className="hover:underline">{seriesLabel(s.series_id)}</Link></td>
                    <td className="px-2 py-2 text-xs text-ink-2">{humanize(s.model)}</td>
                    <VsCell v={s.mase} base={s.naive_mase} higherIsBetter={false} />
                    <td className="num px-2 py-2 text-right text-ink-2">{fmtNum(s.smape, 3)}</td>
                    <td className="num px-2 py-2 text-right">{fmtPct(s.coverage80)}</td>
                    <td className="num px-4 py-2 text-right text-muted-foreground">{s.origins}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section aria-labelledby="ev-lapse">
        <SectionHeader id="ev-lapse" index={2} kicker="lapse model · held-out cut-offs" title="Is the lapse model calibrated?" />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel title="Reliability diagram">
            {r.lapse.calibration.length ? (
              <>
                <ReliabilityDiagram bins={r.lapse.calibration} />
                <TableView
                  caption="Calibration bins"
                  head={["bin", "predicted", "observed", "n"]}
                  rows={r.lapse.calibration.map((c) => [c.bin, c.predicted.toFixed(3), c.observed.toFixed(3), c.n.toLocaleString()])}
                />
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No calibration bins in the report.</p>
            )}
          </Panel>
          <Panel title="Against baselines">
            <table className="w-full text-sm">
              <caption className="sr-only">Lapse model against baselines</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-2 font-normal">Model</th>
                  <th className="px-2 py-2 text-right font-normal">AUC ↑</th>
                  <th className="py-2 pl-2 text-right font-normal">Brier ↓</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b hairline bg-accent/40">
                  <td className="py-2 pr-2">JEV lapse model</td>
                  <td className="num px-2 py-2 text-right font-semibold">{fmtNum(lm.auc, 3)}</td>
                  <td className="num py-2 pl-2 text-right font-semibold">{fmtNum(lm.brier, 3)}</td>
                </tr>
                {r.lapse.baselines.map((b) => (
                  <tr key={b.name} className="border-b hairline last:border-0 text-ink-2">
                    <td className="py-2 pr-2">{humanize(b.name)}</td>
                    <td className="num px-2 py-2 text-right">{fmtNum(b.auc, 3)}</td>
                    <td className="num py-2 pl-2 text-right">{fmtNum(b.brier, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-4">
              <SpecRows
                rows={[
                  { label: "Calibration error (ECE)", value: fmtNum(lm.ece, 3) },
                  { label: "Base rate", value: fmtPct(lm.base_rate) },
                  { label: "Train / test rows", value: `${lm.n_train.toLocaleString()} / ${lm.n_test.toLocaleString()}` },
                  { label: "Test cut-offs", value: lm.test_cutoffs.map((c) => c.slice(0, 7)).join(", ") || "—" },
                ]}
              />
            </div>
          </Panel>
        </div>
      </section>

      <section aria-labelledby="ev-anom">
        <SectionHeader id="ev-anom" index={3} kicker="anomalies · synthetic injection" title="Would it catch a shilling attack?" />
        <Protocol text={inj.protocol} />
        <p className="num mb-3 text-xs text-muted-foreground">{inj.n_genuine.toLocaleString()} genuine raters · {inj.n_injected.toLocaleString()} injected profiles</p>
        {inj.attack_types.length === 0 ? (
          <p className="text-sm text-muted-foreground">No attack types evaluated.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full min-w-[620px] text-sm">
              <caption className="sr-only">Detection of injected attack profiles against a baseline detector</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2.5 font-normal">Attack</th>
                  <th className="px-2 py-2.5 text-right font-normal">n</th>
                  <th className="px-2 py-2.5 text-right font-normal">Precision</th>
                  <th className="px-2 py-2.5 text-right font-normal">Recall</th>
                  <th className="px-2 py-2.5 text-right font-normal">F1</th>
                  <th className="px-4 py-2.5 text-right font-normal">AUC</th>
                </tr>
              </thead>
              <tbody>
                {inj.attack_types.map((a) => (
                  <tr key={a.type} className="border-b hairline last:border-0">
                    <td className="px-4 py-2">{humanize(a.type)}</td>
                    <td className="num px-2 py-2 text-right text-muted-foreground">{a.n}</td>
                    <VsCell v={a.precision} base={a.baseline_precision} />
                    <VsCell v={a.recall} base={a.baseline_recall} />
                    <td className="num px-2 py-2 text-right">{fmtNum(a.f1, 3)}</td>
                    <td className="num px-4 py-2 text-right">{fmtNum(a.auc, 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel title="Series anomalies (injected spikes)">
            <Protocol text={r.anomaly.series.protocol} />
            <SpecRows
              rows={[
                { label: "Detection rate", value: fmtPct(r.anomaly.series.detection_rate, 1) },
                { label: "False-alarm rate", value: fmtPct(r.anomaly.series.false_alarm_rate, 1) },
                { label: "Trials", value: r.anomaly.series.n_trials.toLocaleString() },
              ]}
            />
          </Panel>
          <Panel title="Change points (planted shifts)">
            <Protocol text={cp.protocol} />
            <SpecRows
              rows={[
                { label: "Detection rate", value: fmtPct(cp.detection_rate, 1) },
                { label: "False-alarm rate", value: fmtPct(cp.false_alarm_rate, 1) },
                { label: "Mean location error", value: cp.mean_abs_location_error_months === null ? "—" : `${cp.mean_abs_location_error_months.toFixed(1)} months` },
                { label: "Trials", value: cp.n_trials.toLocaleString() },
              ]}
            />
          </Panel>
        </div>
      </section>

      <section aria-labelledby="ev-lat">
        <SectionHeader id="ev-lat" index={4} kicker="latency" title="How long a run takes" />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)]">
          <Panel title="Pipeline">
            <SpecRows
              rows={[
                { label: "Mean", value: fmtMs(r.latency.pipeline_ms_mean) },
                { label: "p95", value: fmtMs(r.latency.pipeline_ms_p95) },
                { label: "Runs measured", value: r.latency.n_runs },
              ]}
            />
          </Panel>
          <Panel title="Mean time per stage">
            {stages.rows.length ? (
              <>
                <NamedBars rows={stages.rows} format={fmtMs} />
                <p className="mt-3 text-xs text-muted-foreground">In pipeline order{stages.total !== null ? `; mean pipeline total ${fmtMs(stages.total)}` : ""}.</p>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No stage timings.</p>
            )}
          </Panel>
        </div>
      </section>

      {r.notes.length > 0 && (
        <section aria-labelledby="ev-notes">
          <SectionHeader id="ev-notes" kicker="notes" title="Caveats" />
          <ul className="max-w-3xl space-y-2 text-sm text-ink-2">
            {r.notes.map((n) => <li key={n} className="border-l-2 border-rule pl-3">{n}</li>)}
          </ul>
        </section>
      )}
    </div>
  );
}

/** Every synced evaluation report (GET /intel/evaluation/runs), newest first, headline figures only. */
function EvaluationRuns({ current }: { current: string | null }) {
  const { data, error, mutate } = useSWR<EvaluationRunList>("/intel/evaluation/runs");
  return (
    <section aria-labelledby="ev-runs" className="mt-12">
      <SectionHeader id="ev-runs" kicker="history" title="Evaluation runs" />
      {error ? (
        isNotDeployed(error) ? <NotDeployedState what="the evaluation run history (GET /intel/evaluation/runs)" /> : <IntelError error={error} retry={() => mutate()} runBacked={false} />
      ) : !data ? (
        <RowsSkeleton rows={3} />
      ) : data.items.length === 0 ? (
        <EmptyState title="No evaluation runs recorded" body="Reports under experiments/intel-eval-* are synced into the database when the API starts." />
      ) : (
        <>
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full min-w-[860px] text-sm">
              <caption className="sr-only">Evaluation runs, newest first</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2.5 font-normal">Created</th>
                  <th className="px-2 py-2.5 font-normal">Versions</th>
                  <th className="px-2 py-2.5 text-right font-normal">Median MASE ↓</th>
                  <th className="px-2 py-2.5 text-right font-normal">Beating naive ↑</th>
                  <th className="px-2 py-2.5 text-right font-normal">Lapse AUC ↑</th>
                  <th className="px-2 py-2.5 text-right font-normal">Lapse ECE ↓</th>
                  <th className="px-2 py-2.5 text-right font-normal">Shilling AUC ↑</th>
                  <th className="px-4 py-2.5 text-right font-normal">Run time</th>
                </tr>
              </thead>
              <tbody>
                {[...data.items].sort((a, b) => (a.created_at < b.created_at ? 1 : -1)).map((e) => {
                  const h = e.headline ?? ({} as EvaluationRunList["items"][number]["headline"]);
                  const isCurrent = current !== null && e.run_dir === current;
                  return (
                    <tr key={e.id} className={cn("border-b hairline align-top last:border-0", isCurrent && "bg-accent/40")}>
                      <td className="px-4 py-2">
                        <span className="num block text-xs">{fmtDate(e.created_at)}</span>
                        <span className="block break-all font-mono text-[11px] text-muted-foreground">{e.run_dir}{isCurrent ? " · shown above" : ""}</span>
                      </td>
                      <td className="px-2 py-2 font-mono text-[11px] text-ink-2">
                        <span className="block">{e.pipeline_version}</span>
                        <span className="block break-all text-muted-foreground">{e.data_version}</span>
                      </td>
                      <td className="num px-2 py-2 text-right">{fmtNum(h.median_mase, 2)}</td>
                      <td className="num px-2 py-2 text-right">{fmtPct(h.share_beating_naive)}</td>
                      <td className="num px-2 py-2 text-right">{fmtNum(h.lapse_auc, 3)}</td>
                      <td className="num px-2 py-2 text-right">{fmtNum(h.lapse_ece, 3)}</td>
                      <td className="num px-2 py-2 text-right">{fmtNum(h.shilling_auc_mean, 3)}</td>
                      <td className="num px-4 py-2 text-right text-ink-2">{fmtMs(h.pipeline_ms_mean)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{data.total.toLocaleString()} runs. Figures come from each run&apos;s own report; compare runs only when the data version matches.</p>
        </>
      )}
    </section>
  );
}

export default function EvaluationPage() {
  const { data, error, mutate } = useSWR<EvaluationResponse>("/intel/evaluation");
  const r = data?.report ?? null;

  return (
    <div>
      <PageHeader
        eyebrow="review"
        title="Evaluation"
        description={r ? `Offline evaluation of the intelligence layer itself · ${r.pipeline_version} · data ${r.data_version} · created ${fmtDate(r.created_at)}.` : "Offline evaluation of the intelligence layer itself: forecasts, the lapse model, anomaly and change-point detection, and latency."}
        asOf={r?.as_of}
        action={<Button asChild variant="outline" size="sm"><Link href="/admin/experiments">Recommender metrics →</Link></Button>}
      />
      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} />
      ) : !data ? (
        <PanelsSkeleton n={6} />
      ) : !data.available || !r ? (
        <EmptyState
          title="No evaluation report yet"
          body={`Run the intelligence evaluation script to produce one; the console reads the latest report from disk.${data.run_dir ? ` Looked in ${data.run_dir}.` : ""}`}
        />
      ) : (
        <Report r={r} />
      )}
      {(data || error) && <EvaluationRuns current={data?.run_dir ?? null} />}
    </div>
  );
}
