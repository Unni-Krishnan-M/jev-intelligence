"use client";

import { Ban, ChevronRight, Layers, OctagonAlert, X } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";
import useSWR from "swr";

import { fmtDate } from "@/components/jev/admin/ui";
import { ConfidenceBadge, CONFIDENCE_EXPLAIN } from "@/components/jev/intel/badges";
import { OptionScores, ScoreRange } from "@/components/jev/intel/charts";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, Pagination, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { api, qs } from "@/lib/api";
import { fmtAnswer, groupByBatch, humanize } from "@/lib/intel";
import type { ConfidenceKind, DecisionBatch, DecisionBatchList, DecisionRecord, Page } from "@/lib/intel-types";

const LIMIT = 30;

/** Batches of every run on this page, keyed by batch id. A missing endpoint just means no metadata. */
function useBatches(runIds: string[]) {
  const { data } = useSWR<Map<string, DecisionBatch>>(runIds.length ? ["intel:batches", ...runIds] : null, async (key: string[]) => {
    const lists = await Promise.all(
      key.slice(1).map((id) => api<DecisionBatchList>(`/intel/decisions/batches${qs({ run_id: id })}`).then((r) => r.items ?? []).catch(() => [] as DecisionBatch[])),
    );
    return new Map(lists.flat().map((b) => [b.id, b]));
  });
  return data ?? new Map<string, DecisionBatch>();
}

function DecisionRow({ d }: { d: DecisionRecord }) {
  const score = d.kind === "score" && d.scale;
  const numeric = typeof d.answer === "number" ? d.answer : null;
  return (
    <Link href={`/intel/decisions/${d.db_id}`} className="grid grid-cols-1 gap-x-6 gap-y-3 px-4 py-4 hover:bg-accent/40 md:grid-cols-[minmax(0,1fr)_260px_auto]">
      <div className="min-w-0">
        <p className="eyebrow">{humanize(d.key)} · {d.policy_version} · as of {fmtDate(d.as_of).slice(0, 10)}</p>
        <p className="mt-1">{d.question}</p>
        <p className="mt-1.5 flex flex-wrap items-center gap-2 text-sm">
          {d.abstained ? (
            <span className="inline-flex items-center gap-1.5 text-muted-foreground">
              <Ban className="size-3.5" aria-hidden /> Abstained{d.fallback_reason ? ` — ${d.fallback_reason}` : ""}
            </span>
          ) : (
            <>
              <span>Answer: <strong className="font-semibold">{fmtAnswer(d)}</strong></span>
              <ConfidenceBadge value={d.confidence} kind={d.confidence_kind} interval={d.answer_interval} />
            </>
          )}
          {(d.feedback.correct > 0 || d.feedback.incorrect > 0) && (
            <span className="num text-xs text-muted-foreground">feedback {d.feedback.correct} correct · {d.feedback.incorrect} incorrect</span>
          )}
        </p>
      </div>
      {score && d.scale ? (
        <ScoreRange value={d.abstained ? null : numeric} interval={d.abstained ? null : d.answer_interval} scale={d.scale} coverage={d.confidence_kind === "interval" ? d.confidence : null} className="self-center" />
      ) : (
        <OptionScores scores={d.option_scores} answer={d.abstained ? null : String(d.answer)} className="self-center" />
      )}
      <ChevronRight className="hidden size-4 self-center text-muted-foreground md:block" aria-hidden />
    </Link>
  );
}

function BatchHeader({ batchId, batch, n }: { batchId: string; batch: DecisionBatch | undefined; n: number }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-2 border-b hairline bg-muted/30 px-4 py-3">
      <div className="min-w-0">
        <p className="eyebrow inline-flex items-center gap-1.5">
          <Layers className="size-3.5" aria-hidden /> Batch · {batch ? humanize(batch.name) : batchId} · {n} answered together
        </p>
        {batch && <p className="mt-1 text-sm">{batch.question}</p>}
        {batch?.status === "failed" && (
          <p className="mt-1 inline-flex items-center gap-1.5 text-sm">
            <OctagonAlert className="size-3.5 text-destructive" aria-hidden /> Batch failed{batch.failure_reason ? `: ${batch.failure_reason}` : ""}; each question recorded its own abstention.
          </p>
        )}
        <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
          {batch ? (
            <>
              shared state {batch.state_hash.slice(0, 12)} · {batch.keys.map(humanize).join(", ")}
              {batch.n_abstained ? ` · ${batch.n_abstained} abstained` : ""}
            </>
          ) : "batch details not available on this API version"}
        </p>
      </div>
      <Link href={`/intel/decisions${qs({ batch_id: batchId })}`} className="shrink-0 text-xs text-primary hover:underline">Only this batch →</Link>
    </div>
  );
}

function DecisionLog() {
  const batchId = useSearchParams().get("batch_id");
  // keyed by the filter, so paging restarts when it changes
  return <DecisionList key={batchId ?? ""} batchId={batchId} />;
}

function DecisionList({ batchId }: { batchId: string | null }) {
  const [offset, setOffset] = useState(0);
  const { data, error, mutate } = useSWR<Page<DecisionRecord>>(`/intel/decisions${qs({ batch_id: batchId, limit: LIMIT, offset })}`);
  // an API without the batch filter ignores it, so the filter is applied here as well
  const items = (data?.items ?? []).filter((d) => !batchId || d.batch_id === batchId);
  const runIds = Array.from(new Set(items.filter((d) => d.batch_id).map((d) => d.run_id)));
  const batches = useBatches(runIds);

  return (
    <>
      {batchId && (
        <div className="mb-4 flex flex-wrap items-center gap-3 rounded border hairline px-3 py-2 text-sm">
          <Layers className="size-4 text-muted-foreground" aria-hidden />
          <span className="min-w-0 break-all">Batch <span className="font-mono text-xs">{batchId}</span></span>
          <Button asChild variant="ghost" size="sm" className="ml-auto h-7 px-2 text-xs">
            <Link href="/intel/decisions"><X aria-hidden /> Show all decisions</Link>
          </Button>
        </div>
      )}
      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} />
      ) : !data ? (
        <RowsSkeleton rows={6} />
      ) : items.length === 0 ? (
        batchId ? (
          <EmptyState title="No decisions in this batch" body="The batch id matches nothing in the log." action={<Button asChild variant="outline"><Link href="/intel/decisions">Show all decisions</Link></Button>} />
        ) : (
          <EmptyState title="No decisions yet" body="Decisions are recorded on every pipeline run. Run it from the overview." />
        )
      ) : (
        <>
          <ul className="divide-y hairline rounded-lg border bg-card">
            {groupByBatch(items).map((g) =>
              g.batchId ? (
                <li key={g.key}>
                  <BatchHeader batchId={g.batchId} batch={batches.get(g.batchId)} n={g.items.length} />
                  <ul className="divide-y hairline border-l-2 border-primary/40">
                    {g.items.map((d) => <li key={d.db_id}><DecisionRow d={d} /></li>)}
                  </ul>
                </li>
              ) : (
                <li key={g.key}><DecisionRow d={g.items[0]} /></li>
              ),
            )}
          </ul>
          <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
        </>
      )}
    </>
  );
}

export default function DecisionsPage() {
  return (
    <div>
      <PageHeader
        eyebrow="decide"
        title="Decision log"
        description="Fixed questions with declared answers and a versioned policy. Arithmetic happens in code; the policy only weighs the evidence. When data is thin, JEV abstains. Questions asked together share one hashed input state."
      />

      <dl className="mb-6 grid grid-cols-1 gap-3 text-xs sm:grid-cols-2 xl:grid-cols-4">
        {(Object.keys(CONFIDENCE_EXPLAIN) as ConfidenceKind[]).map((k) => (
          <div key={k} className="rounded border hairline px-3 py-2.5">
            <dt className="eyebrow">{k}</dt>
            <dd className="mt-1 leading-relaxed text-ink-2">{CONFIDENCE_EXPLAIN[k].replace(/^\w+: /, "")}</dd>
          </div>
        ))}
      </dl>

      <Suspense fallback={<RowsSkeleton rows={6} />}>
        <DecisionLog />
      </Suspense>
    </div>
  );
}
