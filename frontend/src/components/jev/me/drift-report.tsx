"use client";

import { CircleDashed, GitCompareArrows, Minus } from "lucide-react";

import { EvidenceDisclosure } from "@/components/jev/intel/evidence-list";
import { fmtDay } from "@/lib/intel";
import type { DriftReport as Drift } from "@/lib/intel-types";
import { driftAspectView, driftHeadline, EVIDENCE_NOTE, type AspectVerdict, type DriftVerdict } from "@/lib/me-intel";
import { cn } from "@/lib/utils";

const HEADLINE_ICON: Record<DriftVerdict, typeof Minus> = {
  detected: GitCompareArrows,
  not_detected: Minus,
  insufficient_data: CircleDashed,
};

const VERDICT_ICON: Record<AspectVerdict, typeof Minus> = {
  significant: GitCompareArrows,
  not_significant: Minus,
  insufficient_data: CircleDashed,
};

/** The drift report: headline, the two windows, one row per tested aspect (Holm-adjusted). */
export function DriftReport({ drift }: { drift: Drift }) {
  const head = driftHeadline(drift);
  const Icon = HEADLINE_ICON[head.verdict];
  const rows = drift.aspects.map(driftAspectView);

  return (
    <div>
      <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2">
        <p className="inline-flex items-center gap-2 text-lg">
          <Icon className={cn("size-5", head.verdict === "detected" ? "text-primary" : "text-muted-foreground")} aria-hidden />
          <strong className="font-semibold">{head.label}</strong>
        </p>
        {head.verdict !== "insufficient_data" && (
          <p className="text-sm">
            <span className="num">{head.confidence}</span> <span className="text-muted-foreground">— {EVIDENCE_NOTE}</span>
          </p>
        )}
      </div>
      {drift.summary && <p className="mt-2 max-w-2xl text-[15px] leading-relaxed text-ink-2">{drift.summary}</p>}

      <dl className="mt-4 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
        {([
          ["Historical window", drift.historical_window],
          ["Recent window", drift.recent_window],
        ] as const).map(([label, w]) => (
          <div key={label} className="rounded border hairline px-3 py-2">
            <dt className="eyebrow">{label}</dt>
            <dd className="num mt-1 text-xs">{fmtDay(w.start)} – {fmtDay(w.end)} · {w.n.toLocaleString()} events</dd>
          </div>
        ))}
      </dl>

      {rows.length === 0 ? (
        <p className="mt-4 text-sm text-muted-foreground">No aspects were tested.</p>
      ) : (
        <div className="mt-4 overflow-x-auto rounded-lg border bg-card">
          <table className="w-full min-w-[640px] text-sm">
            <caption className="sr-only">Drift tests per aspect, historical against recent window, with Holm-adjusted p-values</caption>
            <thead>
              <tr className="border-b hairline text-left text-xs text-muted-foreground">
                <th className="px-4 py-2.5 font-normal">Aspect</th>
                <th className="px-2 py-2.5 font-normal">Test</th>
                <th className="px-2 py-2.5 text-right font-normal">Statistic</th>
                <th className="px-2 py-2.5 text-right font-normal">p</th>
                <th className="px-2 py-2.5 text-right font-normal">Adjusted p</th>
                <th className="px-4 py-2.5 font-normal">Result</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const VIcon = VERDICT_ICON[r.verdict];
                return (
                  <tr key={r.aspect} className="border-b hairline align-top last:border-0">
                    <td className="px-4 py-2.5">
                      <p>{r.label}</p>
                      {r.detail && <p className="mt-0.5 max-w-xs text-xs text-muted-foreground">{r.detail}</p>}
                    </td>
                    <td className="px-2 py-2.5 text-xs text-ink-2">{r.test}</td>
                    <td className="num px-2 py-2.5 text-right">{r.statistic}</td>
                    <td className="num px-2 py-2.5 text-right text-ink-2">{r.p}</td>
                    <td className="num px-2 py-2.5 text-right">{r.pAdjusted}</td>
                    <td className="px-4 py-2.5">
                      <span className={cn("inline-flex items-center gap-1.5 whitespace-nowrap text-xs", r.verdict === "significant" ? "text-foreground" : "text-muted-foreground")}>
                        <VIcon className={cn("size-3.5", r.verdict === "significant" && "text-primary")} aria-hidden />
                        {r.verdictLabel}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <p className="mt-2 text-xs text-muted-foreground">
        Each aspect needs a minimum sample; below it the aspect reports insufficient data instead of a number. p-values are Holm-adjusted across
        aspects, and drift is detected when at least one adjusted p is significant.
      </p>
      <EvidenceDisclosure items={drift.evidence} className="mt-3" linkRefs={false} />
    </div>
  );
}
