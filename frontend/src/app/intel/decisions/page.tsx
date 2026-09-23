"use client";

import { Ban, ChevronRight } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";

import { fmtDate } from "@/components/jev/admin/ui";
import { ConfidenceBadge, CONFIDENCE_EXPLAIN } from "@/components/jev/intel/badges";
import { OptionScores } from "@/components/jev/intel/charts";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, Pagination, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { qs } from "@/lib/api";
import type { ConfidenceKind, DecisionRecord, Page } from "@/lib/intel-types";

const LIMIT = 30;

export default function DecisionsPage() {
  const [offset, setOffset] = useState(0);
  const { data, error, mutate } = useSWR<Page<DecisionRecord>>(`/intel/decisions${qs({ limit: LIMIT, offset })}`);

  return (
    <div>
      <PageHeader
        eyebrow="decide"
        title="Decision log"
        description="Fixed questions with declared answers and a versioned policy. Arithmetic happens in code; the policy only weighs the evidence. When data is thin, JEV abstains."
      />

      <dl className="mb-6 grid grid-cols-1 gap-3 text-xs sm:grid-cols-3">
        {(Object.keys(CONFIDENCE_EXPLAIN) as ConfidenceKind[]).map((k) => (
          <div key={k} className="rounded border hairline px-3 py-2.5">
            <dt className="eyebrow">{k}</dt>
            <dd className="mt-1 leading-relaxed text-ink-2">{CONFIDENCE_EXPLAIN[k].replace(/^\w+: /, "")}</dd>
          </div>
        ))}
      </dl>

      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} />
      ) : !data ? (
        <RowsSkeleton rows={6} />
      ) : data.items.length === 0 ? (
        <EmptyState title="No decisions yet" body="Decisions are recorded on every pipeline run. Run it from the overview." />
      ) : (
        <>
          <ul className="divide-y hairline rounded-lg border bg-card">
            {data.items.map((d) => (
              <li key={d.db_id}>
                <Link href={`/intel/decisions/${d.db_id}`} className="grid grid-cols-1 gap-x-6 gap-y-3 px-4 py-4 hover:bg-accent/40 md:grid-cols-[minmax(0,1fr)_260px_auto]">
                  <div className="min-w-0">
                    <p className="eyebrow">{d.key.replaceAll("_", " ")} · {d.policy_version} · as of {fmtDate(d.as_of).slice(0, 10)}</p>
                    <p className="mt-1">{d.question}</p>
                    <p className="mt-1.5 flex flex-wrap items-center gap-2 text-sm">
                      {d.abstained ? (
                        <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                          <Ban className="size-3.5" aria-hidden /> Abstained{d.fallback_reason ? ` — ${d.fallback_reason}` : ""}
                        </span>
                      ) : (
                        <>
                          <span>Answer: <strong className="font-semibold">{d.answer}</strong></span>
                          <ConfidenceBadge value={d.confidence} kind={d.confidence_kind} />
                        </>
                      )}
                      {(d.feedback.correct > 0 || d.feedback.incorrect > 0) && (
                        <span className="num text-xs text-muted-foreground">feedback {d.feedback.correct} correct · {d.feedback.incorrect} incorrect</span>
                      )}
                    </p>
                  </div>
                  <OptionScores scores={d.option_scores} answer={d.abstained ? null : d.answer} className="self-center" />
                  <ChevronRight className="hidden size-4 self-center text-muted-foreground md:block" aria-hidden />
                </Link>
              </li>
            ))}
          </ul>
          <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
        </>
      )}
    </div>
  );
}
