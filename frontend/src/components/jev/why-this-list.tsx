"use client";

import { ArrowRight, Ban, CircleDashed, GitCompareArrows, Minus } from "lucide-react";
import Link from "next/link";

import { ConfidenceBadge } from "@/components/jev/intel/badges";
import { strategyLabel } from "@/lib/decisions";
import type { RecIntelligence } from "@/lib/types";

/**
 * "Why this list": the recommendation-strategy decision the list was produced under (GET
 * /recommendations → intelligence, docs/platform.md §8). Renders nothing on APIs without it.
 */
export function WhyThisList({ intel }: { intel: RecIntelligence | null | undefined }) {
  if (!intel) return null;
  const drift =
    intel.drift_detected === true
      ? { Icon: GitCompareArrows, label: "Taste drift detected" }
      : intel.drift_detected === false
        ? { Icon: Minus, label: "No taste drift detected" }
        : { Icon: CircleDashed, label: "Drift not tested" };
  const n = intel.evidence?.length ?? 0;
  const abstained = intel.abstained ?? intel.strategy === null;
  const served = intel.served_strategy ?? intel.strategy ?? "standard";
  return (
    <section aria-label="Why this list" className="mb-6 grid gap-x-6 gap-y-3 rounded-lg border hairline px-4 py-3 md:grid-cols-[auto_minmax(0,1fr)_auto] md:items-center">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="eyebrow">Why this list</span>
        <span className="text-sm">
          Served as <strong className="font-semibold">{strategyLabel(served)}</strong>
        </span>
        {abstained ? (
          <span className="inline-flex items-center gap-1.5 text-sm text-muted-foreground">
            <Ban className="size-3.5" aria-hidden /> JEV abstained, standard fallback
          </span>
        ) : (
          <ConfidenceBadge value={intel.confidence} kind={intel.confidence_kind} />
        )}
        <span className="inline-flex items-center gap-1.5 text-sm text-ink-2">
          <drift.Icon className="size-4 text-muted-foreground" aria-hidden /> {drift.label}
        </span>
      </div>
      <p className="min-w-0 text-sm text-muted-foreground">
        {intel.summary ?? "The strategy decision recorded no summary."}
        {n > 0 && <span className="num"> · {n} evidence item{n === 1 ? "" : "s"}</span>}
        {intel.policy_version && <span className="font-mono text-xs"> · {intel.policy_version}</span>}
      </p>
      <Link href="/me/intelligence" className="inline-flex shrink-0 items-center gap-1 text-sm text-primary hover:underline">
        My intelligence <ArrowRight className="size-3.5" aria-hidden />
      </Link>
    </section>
  );
}
