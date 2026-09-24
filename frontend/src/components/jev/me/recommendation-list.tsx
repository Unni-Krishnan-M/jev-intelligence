"use client";

import { CornerDownRight, Sparkle } from "lucide-react";
import Link from "next/link";

import { EvidenceDisclosure } from "@/components/jev/intel/evidence-list";
import { MeFeedback } from "@/components/jev/me/me-feedback";
import { Poster } from "@/components/jev/poster";
import { fmtValue } from "@/lib/intel";
import type { EngineRecommendation } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

function movieId(r: EngineRecommendation): number | null {
  const n = typeof r.item_id === "number" ? r.item_id : Number(r.item_id);
  return Number.isInteger(n) && n > 0 ? n : null;
}

function title(r: EngineRecommendation): string {
  return r.title?.trim() || `Item ${r.item_id}`;
}

/**
 * Recommendations downstream of the strategy decision: a typeset cover (from the title; the
 * response carries no genres), the reason, a link to the decision it came from, the evidence and
 * accept / reject.
 */
export function RecommendationCards({ items, strategyId }: { items: EngineRecommendation[]; strategyId: string | null }) {
  return (
    <ol className="grid grid-cols-1 gap-x-6 gap-y-6 sm:grid-cols-2 xl:grid-cols-3">
      {items.map((r) => {
        const id = movieId(r);
        const cover = <Poster movie={{ id: id ?? 1, title: title(r), year: null, genres: [] }} />;
        return (
          <li key={`${r.item_id}-${r.rank}`} className="grid grid-cols-[72px_minmax(0,1fr)] gap-4 border-t hairline pt-4">
            {id ? <Link href={`/movies/${id}`} aria-label={title(r)}>{cover}</Link> : cover}
            <div className="min-w-0">
              <p className="num text-xs text-muted-foreground">No. {r.rank} · score {fmtValue(r.score)}</p>
              {id ? (
                <Link href={`/movies/${id}`} className="font-display mt-0.5 block text-[22px] leading-tight hover:text-primary">{title(r)}</Link>
              ) : (
                <p className="font-display mt-0.5 text-[22px] leading-tight">{title(r)}</p>
              )}
              <p className="mt-1 text-sm text-ink-2">{r.reason}</p>
              {r.decision_id ? (
                <a
                  href={r.decision_id === strategyId ? `#${r.decision_id}` : `#decision`}
                  className="mt-1.5 inline-flex max-w-full items-center gap-1 break-all font-mono text-[11px] text-primary hover:underline"
                >
                  <CornerDownRight className="size-3 shrink-0" aria-hidden /> from {r.decision_id}
                </a>
              ) : (
                <p className="mt-1.5 text-[11px] text-muted-foreground">No decision recorded for this item.</p>
              )}
              <EvidenceDisclosure items={r.evidence} className="mt-1.5" linkRefs={false} />
              <MeFeedback targetType="recommendation" targetId={String(r.item_id)} size="xs" className="mt-2" />
            </div>
          </li>
        );
      })}
    </ol>
  );
}

/** A compact ranked list for scenario comparisons; items not in the baseline list are marked. */
export function RecommendationRanks({ items, baseline, className }: { items: EngineRecommendation[]; baseline?: Set<string>; className?: string }) {
  if (!items.length) return <p className={cn("text-sm text-muted-foreground", className)}>No recommendations.</p>;
  return (
    <ol className={cn("divide-y hairline text-sm", className)}>
      {items.map((r) => {
        const fresh = baseline && !baseline.has(String(r.item_id));
        const id = movieId(r);
        return (
          <li key={`${r.item_id}-${r.rank}`} className="grid grid-cols-[2rem_minmax(0,1fr)_auto] items-baseline gap-2 py-1.5">
            <span className="num text-right text-xs text-muted-foreground">{r.rank}</span>
            {id ? <Link href={`/movies/${id}`} className="truncate hover:underline">{title(r)}</Link> : <span className="truncate">{title(r)}</span>}
            {fresh ? (
              <span className="inline-flex items-center gap-1 text-[11px] text-ink-2"><Sparkle className="size-3 text-primary" aria-hidden /> new</span>
            ) : (
              <span className="text-[11px] text-muted-foreground">{baseline ? "in baseline" : ""}</span>
            )}
          </li>
        );
      })}
    </ol>
  );
}
