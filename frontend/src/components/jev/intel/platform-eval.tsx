"use client";

import { CheckCircle2, CircleX } from "lucide-react";

import { fmtDate, fmtNum, Panel, StatTile } from "@/components/jev/admin/ui";
import { CountBars, TableView } from "@/components/jev/intel/charts";
import { SectionHeader } from "@/components/jev/states";
import { consistencyOf, replayBars, replayMonths } from "@/lib/evaluation";
import { fmtMs, fmtPct, fmtValue, humanize, seriesLabel } from "@/lib/intel";
import type { PlatformEvaluation, WarningOutcomeCounts } from "@/lib/intel-types";

function OutcomeTable({ rows, caption, first }: { rows: [string, WarningOutcomeCounts][]; caption: string; first: string }) {
  return (
    <div className="relative overflow-x-auto rounded-lg border bg-card">
      <table className="w-full min-w-[720px] text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b hairline text-left text-xs text-muted-foreground">
            <th className="px-4 py-2.5 font-normal">{first}</th>
            <th className="px-2 py-2.5 text-right font-normal">Observable</th>
            <th className="px-2 py-2.5 text-right font-normal">Warned</th>
            <th className="px-2 py-2.5 text-right font-normal">Confirmed</th>
            <th className="px-2 py-2.5 text-right font-normal">False alarms</th>
            <th className="px-2 py-2.5 text-right font-normal">Missed</th>
            <th className="px-2 py-2.5 text-right font-normal">Precision ↑</th>
            <th className="px-2 py-2.5 text-right font-normal">FPR ↓</th>
            <th className="px-2 py-2.5 text-right font-normal">Recall ↑</th>
            <th className="px-4 py-2.5 text-right font-normal">Base rate</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([k, c]) => (
            <tr key={k} className="border-b hairline last:border-0">
              <td className="px-4 py-2">{k}</td>
              <td className="num px-2 py-2 text-right text-ink-2">{c.observable.toLocaleString()}</td>
              <td className="num px-2 py-2 text-right">{c.warned.toLocaleString()}</td>
              <td className="num px-2 py-2 text-right">{c.tp.toLocaleString()}</td>
              <td className="num px-2 py-2 text-right">{c.fp.toLocaleString()}</td>
              <td className="num px-2 py-2 text-right">{c.fn.toLocaleString()}</td>
              <td className="num px-2 py-2 text-right">{fmtNum(c.precision, 3)}</td>
              <td className="num px-2 py-2 text-right">{fmtNum(c.false_positive_rate, 3)}</td>
              <td className="num px-2 py-2 text-right">{fmtNum(c.recall, 3)}</td>
              <td className="num px-4 py-2 text-right text-ink-2">{fmtNum(c.base_rate, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Warnings per replay; bars carry month labels only when the replays are provably monthly. */
function ReplayPanel({ title, counts, range, note }: { title: string; counts: number[]; range?: string[] | null; note?: string }) {
  const months = replayMonths(range, counts.length);
  const bars = replayBars(counts, months);
  return (
    <Panel title={title}>
      <CountBars data={bars} name="warnings" binLabel={months ? "as of" : "replay"} />
      <p className="mt-2 text-xs text-muted-foreground">
        {months ? "One bar per monthly replay, labelled by its as-of month" : "Replays in replay order; the report gives no date per replay"}
        {note ? ` · ${note}` : ""}.
      </p>
      <TableView caption="Warnings per replay" head={[months ? "as of" : "replay", "warnings"]} rows={bars.map((b) => [b.bin, b.n])} />
    </Panel>
  );
}

/**
 * The domain's section of the newest platform-eval report: are early warnings confirmed by what
 * happened next, how stable is the early-warning decision between monthly replays, and do the
 * forecasts beat naive.
 */
export function PlatformEvaluationView({ p, index }: { p: PlatformEvaluation; index?: number }) {
  const r = p.report;
  const w = r.warnings.overall;
  const c = consistencyOf(r);
  const fs = r.forecast.summary;
  const kinds = Object.entries(r.warnings.per_kind ?? {});
  const windows = r.windows ?? [];
  const i0 = index ?? 1;

  return (
    <div className="space-y-12">
      <section aria-labelledby="pe-warn">
        <SectionHeader
          id="pe-warn"
          index={i0}
          kicker={`platform evaluation · ${r.replays.n} leak-free replays · horizon ${p.horizon ?? "—"} periods`}
          title="Do the early warnings come true?"
        />
        <p className="mb-4 font-mono text-xs text-muted-foreground">
          {p.run_dir}
          {p.created_at ? ` · created ${fmtDate(p.created_at)}` : ""} · {fmtMs(r.replays.ms_mean)} per replay
        </p>
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatTile label="Precision" value={fmtNum(w.precision, 2)} hint={`${w.tp} of ${w.warned} warnings confirmed`} />
          <StatTile label="False-positive rate" value={fmtNum(w.false_positive_rate, 2)} hint={`${w.fp} false alarms among ${w.fp + w.tn} quiet situations`} />
          <StatTile label="Recall" value={fmtNum(w.recall, 2)} hint={`${w.tp} of ${w.tp + w.fn} adverse outcomes warned`} />
          <StatTile label="Base rate" value={fmtNum(w.base_rate, 2)} hint="share of situations that turned adverse" />
        </div>
        <p className="mt-3 max-w-3xl text-xs leading-relaxed text-muted-foreground">
          A warning is confirmed when its adverse condition is observed within the next {p.horizon ?? "h"} periods of the replay. Compare
          precision with the base rate: a warning is only informative when it beats it.
        </p>
        {kinds.length > 0 && (
          <div className="mt-4">
            <OutcomeTable rows={kinds.map(([k, v]) => [humanize(k), v])} caption="Warning outcomes per kind" first="Kind" />
          </div>
        )}
        {windows.length > 0 && (
          <div className="mt-4">
            <OutcomeTable rows={windows.map((x) => [`${x.window[0].slice(0, 7)} – ${x.window[1].slice(0, 7)}`, x.warnings])} caption="Warning outcomes per replay window" first="Window" />
          </div>
        )}
        {r.warnings.examples && r.warnings.examples.length > 0 && (
          <Panel title="Examples" className="mt-4">
            <ul className="divide-y hairline text-sm">
              {r.warnings.examples.slice(0, 8).map((e, i) => (
                <li key={`${e.as_of}-${e.unit}-${i}`} className="grid grid-cols-1 gap-x-4 gap-y-0.5 py-2 sm:grid-cols-[140px_minmax(0,1fr)]">
                  <span className="inline-flex items-center gap-1.5">
                    {e.confirmed ? <CheckCircle2 className="size-3.5" style={{ color: "var(--status-good)" }} aria-hidden /> : <CircleX className="size-3.5 text-muted-foreground" aria-hidden />}
                    {e.confirmed ? "Confirmed" : "Not confirmed"}
                  </span>
                  <span className="min-w-0">
                    <span className="font-mono text-xs text-muted-foreground">{e.as_of} · {humanize(e.kind)} · {e.unit}</span>
                    <span className="block text-ink-2">{e.detail}</span>
                  </span>
                </li>
              ))}
            </ul>
          </Panel>
        )}
        {r.warnings.excluded_kinds && Object.keys(r.warnings.excluded_kinds).length > 0 && (
          <div className="mt-4">
            <p className="eyebrow mb-1.5">Not scored, and why</p>
            <ul className="space-y-1 text-xs text-muted-foreground">
              {Object.entries(r.warnings.excluded_kinds).map(([k, why]) => <li key={k}><span className="font-mono text-ink-2">{k}</span> — {why}</li>)}
            </ul>
          </div>
        )}
      </section>

      <section aria-labelledby="pe-cons">
        <SectionHeader id="pe-cons" index={i0 + 1} kicker="decision consistency · consecutive replays" title="Does the early-warning level hold steady?" />
        {!c ? (
          <p className="text-sm text-muted-foreground">No consistency figures in this report.</p>
        ) : (
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatTile label="Situation pairs" value={c.pairs.toLocaleString()} hint={c.pooled ? "pooled within the replay windows" : "consecutive monthly replays"} />
            <StatTile label="Level flip rate" value={fmtPct(c.flipRate, 1)} hint={`${c.flips} level changes`} />
            <StatTile label="Warning-boundary flips" value={fmtPct(c.boundaryRate, 1)} hint={`${c.boundaryFlips} crossed Monitor ↔ Warning`} />
            <StatTile label="Replays" value={r.replays.n.toLocaleString()} hint={`${fmtMs(r.replays.ms_mean)} each on average`} />
          </div>
        )}
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
          {r.warnings_per_replay && r.warnings_per_replay.length > 0 && (
            <ReplayPanel
              title={`Warnings raised per replay${r.replays.as_of?.length === 2 ? ` · ${r.replays.as_of[0].slice(0, 7)} – ${r.replays.as_of[1].slice(0, 7)}` : ""}`}
              counts={r.warnings_per_replay}
              range={r.replays.as_of}
            />
          )}
          {windows.map((x) =>
            x.warnings_per_replay?.length ? (
              <ReplayPanel
                key={x.window[0]}
                title={`Warnings per replay · ${x.window[0].slice(0, 7)} – ${x.window[1].slice(0, 7)}`}
                counts={x.warnings_per_replay}
                range={x.window}
                note={`flip rate ${fmtPct(x.consistency.flip_rate ?? null, 1)} · boundary flips ${fmtPct(x.consistency.warning_boundary_flip_rate ?? null, 1)}`}
              />
            ) : null,
          )}
        </div>
      </section>

      <section aria-labelledby="pe-fc">
        <SectionHeader id="pe-fc" index={i0 + 2} kicker="forecasts · rolling origin, per domain" title="Do the forecasts beat naive here?" />
        <p className="mb-3 max-w-3xl border-l-2 border-rule pl-3 text-sm leading-relaxed text-ink-2">{r.forecast.protocol}</p>
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatTile label="Median MASE" value={fmtNum(fs.median_mase, 2)} hint={`naive on the same points: ${fmtNum(fs.median_naive_mase, 2)}`} />
          <StatTile label="Beating naive" value={fmtPct(fs.share_beating_naive)} hint={`of ${fs.n_series} series`} />
          <StatTile label="Mean 80 % coverage" value={fmtPct(fs.mean_coverage80)} hint="target 80 %" />
          <StatTile label="Median RMSE" value={fmtValue(fs.median_rmse)} hint={`median MAE ${fmtValue(fs.median_mae)}`} />
        </div>
        <TableView
          caption="Forecast backtest per series"
          head={["series", "model", "MASE", "naive MASE", "RMSE", "coverage 80"]}
          rows={r.forecast.per_series.map((s) => [seriesLabel(s.series_id), humanize(s.model), fmtNum(s.mase, 2), fmtNum(s.naive_mase, 2), fmtValue(s.rmse), fmtPct(s.coverage80)])}
        />
      </section>
    </div>
  );
}
