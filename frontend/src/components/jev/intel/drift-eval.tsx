"use client";

import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";
import { useState } from "react";
import useSWR from "swr";

import { fmtNum, StatTile } from "@/components/jev/admin/ui";
import { useIntelDomain } from "@/components/jev/intel/domain-context";
import { FilterRow, IntelError, PanelsSkeleton } from "@/components/jev/intel/states";
import { EmptyState, SectionHeader } from "@/components/jev/states";
import { strategyLabel } from "@/lib/decisions";
import { adaptationRows, deltaVerdict, fmtDelta, fmtWithCi, spliceSizes } from "@/lib/evaluation";
import { humanize } from "@/lib/intel";
import type { DetectorMetrics, DriftEvalReport, DriftEvalResponse } from "@/lib/intel-types";
import { ASPECT_LABEL } from "@/lib/me-intel";
import { cn } from "@/lib/utils";

const MODE_LABEL: Record<string, string> = {
  session_permutation: "Session permutation (served)",
  event_permutation: "Event permutation",
};

const SETTING_LABEL: Record<string, string> = {
  refit: "Refit (leak-free)",
  active: "Active model (leaky)",
};

function MetricCells({ m }: { m: DetectorMetrics | undefined }) {
  if (!m) return <td className="px-2 py-2 text-muted-foreground" colSpan={3}>—</td>;
  return (
    <>
      <td className="num px-2 py-2 text-right">{fmtNum(m.precision, 2)}</td>
      <td className="num whitespace-nowrap px-2 py-2 text-right">{fmtWithCi(m.recall, m.recall_ci95)}</td>
      <td className="num whitespace-nowrap px-2 py-2 text-right text-ink-2">≤ {fmtWithCi(m.fpr_upper_bound, m.fpr_ci95)}</td>
    </>
  );
}

const VERDICT_META = {
  better: { Icon: ArrowUpRight, label: "better" },
  worse: { Icon: ArrowDownRight, label: "worse" },
  unclear: { Icon: Minus, label: "no clear change" },
} as const;

function Report({ r }: { r: DriftEvalReport }) {
  const modes = Object.keys(r.detector.modes ?? {});
  const [mode, setMode] = useState(modes.includes("session_permutation") ? "session_permutation" : (modes[0] ?? ""));
  const by = r.detector.modes?.[mode]?.by_splice_size ?? {};
  const sizes = spliceSizes(by);
  const [size, setSize] = useState(sizes[0] ?? "");
  const aspects = Object.entries(by[size]?.per_aspect ?? {});
  const settings = Object.keys(r.adaptation.settings ?? {});
  const [setting, setSetting] = useState(settings.includes("refit") ? "refit" : (settings[0] ?? ""));
  const rows = adaptationRows(r, setting);
  const lat = r.latency ?? {};

  return (
    <div className="space-y-10">
      <div>
        <p className="mb-3 font-mono text-xs text-muted-foreground">
          run {r.run} · model {r.model_version ?? "—"} · {r.detector.n_users.toLocaleString()} users with enough history
        </p>
        <div className="mb-4 flex flex-col gap-2.5">
          <FilterRow label="Permutation" value={mode} onChange={setMode} options={modes.map((m) => ({ value: m, label: MODE_LABEL[m] ?? humanize(m) }))} />
        </div>
        <div className="overflow-x-auto rounded-lg border bg-card">
          <table className="w-full min-w-[760px] text-sm">
            <caption className="sr-only">Drift detector precision, recall and false-positive rate per splice size</caption>
            <thead>
              <tr className="border-b hairline text-left text-xs text-muted-foreground">
                <th className="px-4 py-2 font-normal" rowSpan={2}>Spliced events</th>
                <th className="border-l hairline px-2 py-2 text-center font-normal" colSpan={3}>Any drift detected</th>
                <th className="border-l hairline px-2 py-2 text-center font-normal" colSpan={3}>Preference drift</th>
                <th className="px-4 py-2 text-right font-normal" rowSpan={2}>Pos / neg</th>
              </tr>
              <tr className="border-b hairline text-right text-xs text-muted-foreground">
                <th className="border-l hairline px-2 py-1.5 font-normal">Precision</th>
                <th className="px-2 py-1.5 font-normal">Recall [95 %]</th>
                <th className="px-2 py-1.5 font-normal">FPR [95 %]</th>
                <th className="border-l hairline px-2 py-1.5 font-normal">Precision</th>
                <th className="px-2 py-1.5 font-normal">Recall [95 %]</th>
                <th className="px-2 py-1.5 font-normal">FPR [95 %]</th>
              </tr>
            </thead>
            <tbody>
              {sizes.map((s) => (
                <tr key={s} className="border-b hairline last:border-0">
                  <td className="num px-4 py-2">{s}</td>
                  <MetricCells m={by[s]?.overall_drift_detected} />
                  <MetricCells m={by[s]?.preference_drift} />
                  <td className="num px-4 py-2 text-right text-muted-foreground">{by[s]?.preference_drift?.n_pos ?? "—"} / {by[s]?.preference_drift?.n_neg ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 max-w-3xl text-xs leading-relaxed text-muted-foreground">
          Labelled drift is made by splicing another member&apos;s recent events onto a real history; untouched histories are the negatives.
          False-positive rates are upper bounds, since real histories can drift on their own.
        </p>

        {sizes.length > 0 && (
          <div className="mt-6">
            <div className="mb-3"><FilterRow label="Per aspect at" value={size} onChange={setSize} options={sizes.map((s) => ({ value: s, label: `${s} events` }))} /></div>
            <div className="overflow-x-auto rounded-lg border bg-card">
              <table className="w-full min-w-[620px] text-sm">
                <caption className="sr-only">Detector metrics per aspect</caption>
                <thead>
                  <tr className="border-b hairline text-left text-xs text-muted-foreground">
                    <th className="px-4 py-2.5 font-normal">Aspect</th>
                    <th className="px-2 py-2.5 text-right font-normal">Precision</th>
                    <th className="px-2 py-2.5 text-right font-normal">Recall [95 %]</th>
                    <th className="px-2 py-2.5 text-right font-normal">FPR [95 %]</th>
                    <th className="px-4 py-2.5 text-right font-normal">Pos / neg</th>
                  </tr>
                </thead>
                <tbody>
                  {aspects.map(([a, m]) => (
                    <tr key={a} className="border-b hairline last:border-0">
                      <td className="px-4 py-2">{ASPECT_LABEL[a] ?? humanize(a)}</td>
                      <MetricCells m={m} />
                      <td className="num px-4 py-2 text-right text-muted-foreground">{m.n_pos} / {m.n_neg}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
        {r.detector.notes && r.detector.notes.length > 0 && (
          <ul className="mt-3 space-y-1 text-xs text-muted-foreground">{r.detector.notes.map((n) => <li key={n} className="border-l-2 border-rule pl-3">{n}</li>)}</ul>
        )}
      </div>

      <div>
        <p className="eyebrow mb-2">Adaptation effect · NDCG@10 against standard, paired bootstrap</p>
        <p className="mb-3 max-w-3xl border-l-2 border-rule pl-3 text-sm leading-relaxed text-ink-2">{r.adaptation.protocol}</p>
        <div className="mb-3"><FilterRow label="Setting" value={setting} onChange={setSetting} options={settings.map((s) => ({ value: s, label: SETTING_LABEL[s] ?? humanize(s) }))} /></div>
        <div className="overflow-x-auto rounded-lg border bg-card">
          <table className="w-full min-w-[760px] text-sm">
            <caption className="sr-only">Change in NDCG@10 per strategy against standard, per user group</caption>
            <thead>
              <tr className="border-b hairline text-left text-xs text-muted-foreground">
                <th className="px-4 py-2.5 font-normal">Users</th>
                <th className="px-2 py-2.5 text-right font-normal">n</th>
                <th className="px-2 py-2.5 text-right font-normal">Standard NDCG@10</th>
                <th className="px-2 py-2.5 text-right font-normal">{strategyLabel("adapt_to_recent")} Δ [95 %]</th>
                <th className="px-4 py-2.5 text-right font-normal">{strategyLabel("explore")} Δ [95 %]</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.group} className="border-b hairline last:border-0">
                  <td className="px-4 py-2">{row.label}</td>
                  <td className="num px-2 py-2 text-right text-muted-foreground">{row.n?.toLocaleString() ?? "—"}</td>
                  <td className="num px-2 py-2 text-right">{fmtNum(row.standard, 4)}</td>
                  {row.deltas.map(({ variant, d }, i) => {
                    const v = VERDICT_META[deltaVerdict(d)];
                    return (
                      <td key={variant} className={cn("num whitespace-nowrap py-2 text-right", i === row.deltas.length - 1 ? "px-4" : "px-2")}>
                        <span className="inline-flex items-center gap-1">
                          <v.Icon className="size-3.5 text-muted-foreground" aria-hidden />
                          {fmtDelta(d)}
                          <span className="sr-only"> ({v.label})</span>
                        </span>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-2 text-xs text-muted-foreground">
          The arrow marks a 95 % interval that excludes zero (↗ better, ↘ worse); a dash means no clear change. Exploratory variants in the report are not
          used by the policy and are left out.
        </p>
        {r.notes && r.notes.length > 0 && (
          <ul className="mt-3 space-y-1 text-xs text-muted-foreground">{r.notes.map((n) => <li key={n} className="border-l-2 border-rule pl-3">{n}</li>)}</ul>
        )}
      </div>

      {(lat.strategy_for_ms || lat.user_intelligence_ms) && (
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          <StatTile label="Strategy decision p50" value={`${fmtNum(Number(lat.strategy_for_ms?.p50), 1)} ms`} hint="per member" />
          <StatTile label="Strategy decision p95" value={`${fmtNum(Number(lat.strategy_for_ms?.p95), 1)} ms`} hint={String(lat.strategy_for_ms?.typical_users ?? "")} />
          <StatTile label="My intelligence p50" value={`${fmtNum(Number(lat.user_intelligence_ms?.p50), 1)} ms`} />
          <StatTile label="My intelligence p95" value={`${fmtNum(Number(lat.user_intelligence_ms?.p95), 1)} ms`} />
        </div>
      )}
    </div>
  );
}

/** GET /intel/evaluation/drift (movie): the preference-drift detector and the adaptation effect. */
export function DriftEvaluation({ index }: { index: number }) {
  const { q: dq } = useIntelDomain();
  const { data, error, mutate } = useSWR<DriftEvalResponse>(dq("/intel/evaluation/drift"));
  return (
    <section aria-labelledby="ev-drift">
      <SectionHeader id="ev-drift" index={index} kicker="preference drift · spliced histories" title="Does the drift test find real drift, and does adapting help?" />
      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} what="the drift evaluation (GET /intel/evaluation/drift)" since="v1.2" />
      ) : !data ? (
        <PanelsSkeleton n={2} />
      ) : !data.available || !data.report ? (
        <EmptyState title="No drift evaluation yet" body={`Run scripts/evaluate_drift.py to produce one.${data.run_dir ? ` Looked in ${data.run_dir}.` : ""}`} />
      ) : (
        <Report r={data.report} />
      )}
    </section>
  );
}
