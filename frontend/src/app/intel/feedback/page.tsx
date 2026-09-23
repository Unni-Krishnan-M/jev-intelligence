"use client";

import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";

import { fmtDate, Panel } from "@/components/jev/admin/ui";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, Pagination, PanelsSkeleton, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { qs } from "@/lib/api";
import { fmtPct, humanize, refHref } from "@/lib/intel";
import type { FeedbackList, FeedbackRecord } from "@/lib/intel-types";

const LIMIT = 50;
/** Below this many verdicts a rate is shown as counts only: too few to mean much. */
const MIN_FOR_RATE = 5;

function Rate({ label, value, n, parts, what }: { label: string; value: number | null; n: number; parts: string; what: string }) {
  const enough = value !== null && n >= MIN_FOR_RATE;
  return (
    <Panel title={label}>
      {enough ? (
        <p className="text-4xl leading-none">{fmtPct(value, 0)}</p>
      ) : (
        <p className="font-display text-2xl leading-tight">Not enough feedback yet</p>
      )}
      <p className="num mt-2 text-xs text-muted-foreground">{parts}</p>
      <p className="mt-2 text-xs text-muted-foreground">
        {enough ? what : `Needs at least ${MIN_FOR_RATE} verdicts before a rate is shown (${n} so far).`}
      </p>
    </Panel>
  );
}

function targetHref(f: FeedbackRecord): string | null {
  if (f.target_type === "warning" && /^\d+$/.test(f.target_id)) return `/intel/warnings/${f.target_id}`;
  if (f.target_type === "decision" && /^\d+$/.test(f.target_id)) return `/intel/decisions/${f.target_id}`;
  return refHref(f.target_id);
}

export default function FeedbackPage() {
  const [offset, setOffset] = useState(0);
  const { data, error, mutate } = useSWR<FeedbackList>(`/intel/feedback${qs({ limit: LIMIT, offset })}`);
  const s = data?.summary;

  return (
    <div>
      <PageHeader
        eyebrow="review"
        title="Operator feedback"
        description="Verdicts people gave on decisions, warnings, actions and predictions. This is how the layer is judged against what actually happened."
      />

      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} />
      ) : !data || !s ? (
        <div className="space-y-6"><PanelsSkeleton n={4} className="lg:grid-cols-4" /><RowsSkeleton /></div>
      ) : (
        <div className="space-y-8">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Rate
              label="Decision accuracy"
              value={s.decision.accuracy}
              n={s.decision.correct + s.decision.incorrect}
              parts={`${s.decision.correct} correct · ${s.decision.incorrect} incorrect`}
              what="Share of judged decisions marked correct. Only judged decisions count."
            />
            <Rate
              label="Warning precision"
              value={s.warning.precision}
              n={s.warning.useful + s.warning.not_useful + s.warning.false_positive}
              parts={`${s.warning.useful} useful · ${s.warning.not_useful} not useful · ${s.warning.false_positive} false positive`}
              what="Share of judged warnings marked useful."
            />
            <Panel title="Actions">
              <p className="num text-2xl leading-none">{s.action.useful} <span className="text-sm text-muted-foreground">useful</span></p>
              <p className="num mt-1 text-2xl leading-none">{s.action.not_useful} <span className="text-sm text-muted-foreground">not useful</span></p>
            </Panel>
            <Panel title="Predictions">
              <p className="num text-2xl leading-none">{s.prediction.correct} <span className="text-sm text-muted-foreground">held up</span></p>
              <p className="num mt-1 text-2xl leading-none">{s.prediction.incorrect} <span className="text-sm text-muted-foreground">did not</span></p>
            </Panel>
          </div>

          {data.items.length === 0 ? (
            <EmptyState title="No feedback yet" body="Use the Correct / Useful buttons on decisions, warnings, actions and forecasts. Verdicts show up here." />
          ) : (
            <>
              <div className="overflow-x-auto rounded-lg border bg-card">
                <table className="w-full min-w-[720px] text-sm">
                  <caption className="sr-only">Feedback log</caption>
                  <thead>
                    <tr className="border-b hairline text-left text-xs text-muted-foreground">
                      <th className="px-4 py-2.5 font-normal">When</th>
                      <th className="px-2 py-2.5 font-normal">Target</th>
                      <th className="px-2 py-2.5 font-normal">Verdict</th>
                      <th className="px-2 py-2.5 font-normal">Note / outcome</th>
                      <th className="px-4 py-2.5 font-normal">By</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map((f) => {
                      const href = targetHref(f);
                      return (
                        <tr key={f.id} className="border-b hairline align-top last:border-0">
                          <td className="num px-4 py-2.5 text-xs text-muted-foreground">{fmtDate(f.created_at)}</td>
                          <td className="px-2 py-2.5">
                            <span className="eyebrow block">{f.target_type}</span>
                            {href ? <Link href={href} className="break-all font-mono text-xs text-primary hover:underline">{f.target_id}</Link> : <span className="break-all font-mono text-xs">{f.target_id}</span>}
                          </td>
                          <td className="px-2 py-2.5">{humanize(f.verdict)}</td>
                          <td className="px-2 py-2.5 text-ink-2">
                            {f.note && <p>{f.note}</p>}
                            {f.outcome && <p className="text-xs"><span className="text-muted-foreground">outcome:</span> {f.outcome}</p>}
                            {!f.note && !f.outcome && <span className="text-muted-foreground">—</span>}
                          </td>
                          <td className="px-4 py-2.5 text-xs text-muted-foreground">{f.actor}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
            </>
          )}
        </div>
      )}
    </div>
  );
}
