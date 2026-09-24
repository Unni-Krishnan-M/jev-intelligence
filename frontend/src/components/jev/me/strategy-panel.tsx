"use client";

import { Ban } from "lucide-react";

import { SpecRows } from "@/components/jev/admin/ui";
import { ConfidenceBadge, CONFIDENCE_EXPLAIN } from "@/components/jev/intel/badges";
import { OptionScores } from "@/components/jev/intel/charts";
import { EvidenceList } from "@/components/jev/intel/evidence-list";
import { MeFeedback } from "@/components/jev/me/me-feedback";
import { STRATEGY_MEANING, strategyLabel } from "@/lib/decisions";
import { fmtValue, humanize } from "@/lib/intel";
import type { Decision, RecommendationStrategy } from "@/lib/intel-types";

function stateRows(state: Record<string, unknown>) {
  const rows: { label: string; value: string }[] = [];
  for (const [k, v] of Object.entries(state ?? {})) {
    if (v && typeof v === "object" && !Array.isArray(v)) {
      for (const [k2, v2] of Object.entries(v as Record<string, unknown>)) rows.push({ label: `${humanize(k)} · ${humanize(k2)}`, value: fmtValue(v2, k2) });
    } else rows.push({ label: humanize(k), value: fmtValue(v, k) });
  }
  return rows;
}

/** The recommendation_strategy decision: answer, option scores, confidence kind, why, and feedback. */
export function StrategyPanel({ d }: { d: Decision }) {
  const answer = d.abstained ? null : d.answer;
  const meaning = typeof answer === "string" && answer in STRATEGY_MEANING ? STRATEGY_MEANING[answer as RecommendationStrategy] : null;
  const labelled = Object.fromEntries(Object.entries(d.option_scores ?? {}).map(([k, v]) => [strategyLabel(k), v]));
  const rows = stateRows(d.state);

  return (
    <div id={d.id} className="scroll-mt-24">
      <p className="eyebrow">{d.question}</p>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        {d.abstained ? (
          <p className="inline-flex items-center gap-2 text-2xl">
            <Ban className="size-5 text-muted-foreground" aria-hidden />
            <span className="font-display">Abstained</span>
          </p>
        ) : (
          <p className="font-display text-[34px] leading-none">{strategyLabel(answer)}</p>
        )}
        {!d.abstained && <ConfidenceBadge value={d.confidence} kind={d.confidence_kind} />}
      </div>
      {d.abstained ? (
        <p className="mt-2 text-sm text-ink-2">
          JEV did not choose: {d.fallback_reason ?? "the evidence was insufficient"}. Your list falls back to the standard strategy; no answer is better than a guess here.
        </p>
      ) : (
        meaning && <p className="mt-2 text-sm text-ink-2">{meaning}</p>
      )}
      {d.fallback_reason && !d.abstained && <p className="mt-1 text-xs text-muted-foreground">Fallback: {d.fallback_reason}</p>}
      <p className="mt-2 max-w-2xl text-xs text-muted-foreground">{CONFIDENCE_EXPLAIN[d.confidence_kind]}</p>

      <div className="mt-6 grid grid-cols-1 gap-8 lg:grid-cols-2">
        <div>
          <p className="eyebrow mb-2">Option scores</p>
          {Object.keys(labelled).length ? (
            <OptionScores scores={labelled} answer={typeof answer === "string" ? strategyLabel(answer) : null} />
          ) : (
            <p className="text-sm text-muted-foreground">No option scores recorded.</p>
          )}
          <p className="mt-2 text-xs text-muted-foreground">The policy&apos;s weighing of each option. Not probabilities unless the confidence kind says so.</p>

          <p className="eyebrow mb-2 mt-6">Rationale</p>
          {d.rationale.length ? (
            <ol className="list-decimal space-y-1.5 pl-5 text-sm leading-relaxed marker:font-mono marker:text-xs marker:text-muted-foreground">
              {d.rationale.map((r, i) => <li key={i}>{r}</li>)}
            </ol>
          ) : (
            <p className="text-sm text-muted-foreground">No rationale recorded.</p>
          )}
        </div>
        <div>
          <p className="eyebrow mb-1">State the policy saw</p>
          {rows.length ? <SpecRows rows={rows} /> : <p className="text-sm text-muted-foreground">Empty snapshot.</p>}
          <p className="eyebrow mb-2 mt-6">Evidence</p>
          <EvidenceList items={d.evidence} linkRefs={false} />
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t hairline pt-4">
        <p className="font-mono text-xs text-muted-foreground">
          {d.id} · policy {d.policy_version}
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-muted-foreground">Does this strategy suit you?</span>
          <MeFeedback targetType="strategy" targetId={d.id} />
        </div>
      </div>
    </div>
  );
}
