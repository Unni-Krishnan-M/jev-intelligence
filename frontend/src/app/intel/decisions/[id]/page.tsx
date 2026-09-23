"use client";

import { ArrowLeft, Ban, Layers } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import useSWR from "swr";

import { fmtDate, Panel, SpecRows } from "@/components/jev/admin/ui";
import { ConfidenceBadge, CONFIDENCE_EXPLAIN } from "@/components/jev/intel/badges";
import { OptionScores, ScoreRange } from "@/components/jev/intel/charts";
import { EvidenceList } from "@/components/jev/intel/evidence-list";
import { FeedbackButtons } from "@/components/jev/intel/feedback-buttons";
import { IntelError } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, qs } from "@/lib/api";
import { fmtAnswer, fmtValue, humanize } from "@/lib/intel";
import type { DecisionBatchList, DecisionRecord, Page, RunList } from "@/lib/intel-types";

/** Flatten the policy's input snapshot into label/value rows (one level of nesting). */
function stateRows(state: Record<string, unknown>) {
  const rows: { label: string; value: string }[] = [];
  for (const [k, v] of Object.entries(state ?? {})) {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      for (const [k2, v2] of Object.entries(v as Record<string, unknown>)) rows.push({ label: `${humanize(k)} · ${humanize(k2)}`, value: fmtValue(v2, k2) });
    } else {
      rows.push({ label: humanize(k), value: fmtValue(v, k) });
    }
  }
  return rows;
}

/** What this decision cannot tell you, derived from its own fields. */
function limitations(d: DecisionRecord): string[] {
  const out: string[] = [];
  if (d.abstained) out.push(`JEV abstained: ${d.fallback_reason ?? "the evidence was insufficient"}. No answer is better than a guess here.`);
  if (d.confidence_kind === "margin") out.push("Confidence is an evidence margin between the top two options, not a probability of being right.");
  if (d.confidence_kind === "rule") out.push("Confidence comes from a deterministic threshold rule; it records that the rule fired, not how likely the answer is to be right.");
  if (d.confidence_kind === "probability") out.push("Confidence is an estimated probability from the data up to as-of; it is not a guarantee.");
  if (d.confidence_kind === "interval") out.push("Confidence is the nominal coverage of the interval around the answer: the range should contain the outcome that often. It is not a probability that the answer itself is right.");
  if (d.kind === "score") out.push("The answer is a recommended number on a bounded scale; values outside the scale are never proposed.");
  out.push(`Only data at or before ${fmtDate(d.as_of)} was used; anything later is invisible to this decision.`);
  if (d.feedback.correct + d.feedback.incorrect === 0) out.push("No operator has judged this decision yet, so its track record is unknown.");
  return out;
}

/** The other questions answered in the same call, against the same hashed state. */
function BatchPanel({ d }: { d: DecisionRecord }) {
  const batchId = d.batch_id as string;
  const batches = useSWR<DecisionBatchList>(`/intel/decisions/batches${qs({ run_id: d.run_id })}`, { shouldRetryOnError: false });
  const siblings = useSWR<Page<DecisionRecord>>(`/intel/decisions${qs({ batch_id: batchId, run_id: d.run_id, limit: 50 })}`);
  const batch = batches.data?.items.find((b) => b.id === batchId);
  // an API without the batch filter returns everything, so filter here too
  const members = (siblings.data?.items ?? []).filter((x) => x.batch_id === batchId && x.run_id === d.run_id);
  return (
    <Panel title={<span className="inline-flex items-center gap-1.5"><Layers className="size-3.5" aria-hidden /> Answered together</span>}>
      <p className="text-sm">{batch ? batch.question : <>Part of batch <span className="break-all font-mono text-xs">{batchId}</span>.</>}</p>
      <div className="mt-3">
        <SpecRows
          rows={[
            { label: "Batch", value: batch ? humanize(batch.name) : batchId },
            { label: "Shared state", value: batch ? <span title={batch.state_hash}>{batch.state_hash.slice(0, 12)}</span> : batches.error ? "not available" : "…" },
            ...(batch?.state_keys?.length ? [{ label: "State covers", value: <span title={batch.state_keys.join(", ")}>{batch.state_keys.map(humanize).join(", ")}</span> }] : []),
            ...(batch?.status ? [{ label: "Call", value: batch.status === "ok" ? `ok${batch.n_abstained ? ` · ${batch.n_abstained} abstained` : ""}` : `failed${batch.failure_reason ? ` · ${batch.failure_reason}` : ""}` }] : []),
            ...(batch ? Object.entries(batch.policy_versions).map(([k, v]) => ({ label: `Policy · ${humanize(k)}`, value: v })) : []),
          ]}
        />
      </div>
      <p className="eyebrow mb-1.5 mt-4">Questions in this call</p>
      {!siblings.data && !siblings.error ? (
        <Skeleton className="h-16 w-full" />
      ) : members.length ? (
        <ul className="divide-y hairline text-sm">
          {members.map((m) => (
            <li key={m.db_id} className="flex items-baseline justify-between gap-3 py-1.5">
              {m.db_id === d.db_id ? (
                <span className="min-w-0">{m.question} <span className="text-xs text-muted-foreground">(this one)</span></span>
              ) : (
                <Link href={`/intel/decisions/${m.db_id}`} className="min-w-0 hover:underline">{m.question}</Link>
              )}
              <span className="num shrink-0 text-xs text-ink-2">{m.abstained ? "abstained" : fmtAnswer(m)}</span>
            </li>
          ))}
        </ul>
      ) : batch ? (
        <p className="text-sm text-ink-2">{batch.keys.map(humanize).join(", ")}</p>
      ) : (
        <p className="text-sm text-muted-foreground">The other members could not be listed.</p>
      )}
      <Link href={`/intel/decisions${qs({ batch_id: batchId })}`} className="mt-3 inline-block text-xs text-primary hover:underline">Batch in the log →</Link>
    </Panel>
  );
}

export default function DecisionDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data: d, error, mutate } = useSWR<DecisionRecord>(id ? `/intel/decisions/${encodeURIComponent(id)}` : null);
  const runs = useSWR<RunList>(d ? "/intel/runs" : null);
  const run = runs.data?.items.find((r) => r.run_id === d?.run_id);
  const modelVersion = run?.model_version ?? (d?.entity_type === "model" ? d.entity : null);

  const back = (
    <Link href="/intel/decisions" className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
      <ArrowLeft className="size-4" aria-hidden /> Decision log
    </Link>
  );

  if (error instanceof ApiError && error.status === 404) return <div>{back}<EmptyState title="Decision not found" body={`No decision with id ${id}.`} /></div>;
  if (error) return <div>{back}<IntelError error={error} retry={() => mutate()} runBacked={false} /></div>;
  if (!d) {
    return (
      <div className="space-y-4">
        {back}
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-12 w-3/4" />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2"><Skeleton className="h-56" /><Skeleton className="h-56" /></div>
      </div>
    );
  }

  const rows = stateRows(d.state);

  return (
    <div>
      {back}
      <header className="mb-8 border-b hairline pb-6">
        <p className="eyebrow">Decision · {humanize(d.key)} · {d.kind}</p>
        <h1 className="font-display mt-2 text-[34px] leading-[1.02] tracking-tight text-balance sm:text-[44px]">{d.question}</h1>
        <div className="mt-4 flex flex-wrap items-center gap-3">
          {d.abstained ? (
            <p className="inline-flex items-center gap-2 text-lg">
              <Ban className="size-5 text-muted-foreground" aria-hidden />
              Abstained
              <span className="text-sm text-muted-foreground">{d.fallback_reason ?? "insufficient evidence"}</span>
            </p>
          ) : (
            <>
              <p className="text-lg">Answer: <strong className="font-semibold">{fmtAnswer(d)}</strong></p>
              <ConfidenceBadge value={d.confidence} kind={d.confidence_kind} interval={d.answer_interval} />
            </>
          )}
        </div>
        <p className="mt-2 max-w-2xl text-xs text-muted-foreground">{CONFIDENCE_EXPLAIN[d.confidence_kind]}</p>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          {d.kind === "score" && d.scale ? (
            <Panel title="Answer on its scale">
              <ScoreRange
                value={d.abstained || typeof d.answer !== "number" ? null : d.answer}
                interval={d.abstained ? null : d.answer_interval}
                scale={d.scale}
                coverage={d.confidence_kind === "interval" ? d.confidence : null}
              />
              <div className="mt-4">
                <SpecRows
                  rows={[
                    { label: "Answer", value: d.abstained ? "abstained" : fmtAnswer(d) },
                    { label: "Interval", value: d.answer_interval ? `${fmtValue(d.answer_interval[0])} – ${fmtValue(d.answer_interval[1])}` : "—" },
                    { label: "Nominal coverage", value: d.confidence_kind === "interval" && d.confidence !== null ? `${(d.confidence * 100).toFixed(0)} %` : "—" },
                    { label: "Scale", value: `${fmtValue(d.scale.min)} – ${fmtValue(d.scale.max)}` },
                    { label: "Unit", value: <span title={d.scale.unit}>{d.scale.unit || "—"}</span> },
                  ]}
                />
              </div>
              <p className="mt-3 text-xs text-muted-foreground">The shaded range is the interval the confidence refers to; the tick is the recommended value.</p>
            </Panel>
          ) : (
            <Panel title="Option scores">
              <OptionScores scores={d.option_scores} answer={d.abstained ? null : String(d.answer)} />
              <p className="mt-3 text-xs text-muted-foreground">Scores are the policy&apos;s weighing of each option ({d.options.join(" / ")}). They are not probabilities unless the confidence kind says so.</p>
            </Panel>
          )}

          <section aria-labelledby="why-heading" className="rounded-lg border bg-card p-4 sm:p-5">
            <h2 id="why-heading" className="font-display text-2xl">Why this decision</h2>
            <div className="mt-4 grid grid-cols-1 gap-6 xl:grid-cols-2">
              <div>
                <p className="eyebrow mb-2">Rationale</p>
                {d.rationale.length ? (
                  <ol className="list-decimal space-y-1.5 pl-5 text-sm leading-relaxed marker:font-mono marker:text-xs marker:text-muted-foreground">
                    {d.rationale.map((r, i) => <li key={i}>{r}</li>)}
                  </ol>
                ) : (
                  <p className="text-sm text-muted-foreground">No rationale recorded.</p>
                )}
                <p className="eyebrow mb-1 mt-6">State the policy saw</p>
                {rows.length ? <SpecRows rows={rows} /> : <p className="text-sm text-muted-foreground">Empty snapshot.</p>}
              </div>
              <div>
                <p className="eyebrow mb-2">Evidence</p>
                <EvidenceList items={d.evidence} />
                <p className="eyebrow mb-2 mt-6">Limitations</p>
                <ul className="space-y-1.5 text-sm text-ink-2">
                  {limitations(d).map((l) => <li key={l} className="border-l-2 border-rule pl-3">{l}</li>)}
                </ul>
              </div>
            </div>
          </section>
        </div>

        <div className="space-y-4">
          <Panel title="Provenance">
            <SpecRows
              rows={[
                { label: "Policy", value: d.policy_version },
                { label: "Spec", value: d.spec_id },
                { label: "Model version", value: modelVersion ?? (runs.data ? "— (none in this run)" : "…") },
                { label: "As of", value: fmtDate(d.as_of) },
                { label: "Recorded", value: fmtDate(d.created_at) },
                { label: "Run", value: <span title={d.run_id}>{d.run_id.slice(0, 8)}</span> },
                { label: "Subject", value: `${d.entity_type} ${d.entity}` },
                { label: "Decision id", value: <span title={d.id}>{d.id}</span> },
                ...(d.batch_id ? [{ label: "Batch", value: <Link href={`/intel/decisions${qs({ batch_id: d.batch_id })}`} className="text-primary hover:underline" title={d.batch_id}>{d.batch_id}</Link> }] : []),
              ]}
            />
          </Panel>
          {d.batch_id && <BatchPanel d={d} />}
          <Panel title="Was it right?">
            <p className="num mb-3 text-xs text-muted-foreground">so far: {d.feedback.correct} correct · {d.feedback.incorrect} incorrect</p>
            <FeedbackButtons targetType="decision" targetId={d.id} verdicts={["correct", "incorrect"]} noteField="outcome" notePlaceholder="Outcome, if known (optional)" />
          </Panel>
        </div>
      </div>
    </div>
  );
}
