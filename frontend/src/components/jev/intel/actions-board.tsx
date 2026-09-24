"use client";

import { EffortBadge } from "@/components/jev/intel/badges";
import Link from "@/components/jev/intel/domain-context";
import { EvidenceDisclosure } from "@/components/jev/intel/evidence-list";
import { FeedbackButtons } from "@/components/jev/intel/feedback-buttons";
import { SectionHeader } from "@/components/jev/states";
import { sourceHref } from "@/lib/intel";
import type { Action } from "@/lib/intel-types";

/** The ranked action plan: P1 → P3 groups of action cards, each with its evidence and source. */
const PRIORITIES: { p: Action["priority"]; title: string; kicker: string }[] = [
  { p: "P1", title: "Do first", kicker: "P1" },
  { p: "P2", title: "Plan this cycle", kicker: "P2" },
  { p: "P3", title: "When there is room", kicker: "P3" },
];

export function ActionCard({ a }: { a: Action }) {
  const src = sourceHref(a.source);
  return (
    <article id={a.id} className="scroll-mt-24 rounded-lg border bg-card p-4 sm:p-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <h3 className="font-display min-w-0 text-2xl leading-tight">{a.title}</h3>
        <span className="flex items-center gap-2">
          <EffortBadge effort={a.effort} />
          <span className="num text-xs text-muted-foreground" title="Priority score">score {a.priority_score.toFixed(2)}</span>
        </span>
      </header>
      <dl className="mt-4 grid grid-cols-1 gap-x-8 gap-y-3 text-sm md:grid-cols-2">
        <div><dt className="eyebrow">Why</dt><dd className="mt-1 leading-relaxed">{a.reason}</dd></div>
        <div><dt className="eyebrow">Expected impact</dt><dd className="mt-1 leading-relaxed">{a.expected_impact}</dd></div>
        <div><dt className="eyebrow">Risk of acting</dt><dd className="mt-1 leading-relaxed text-ink-2">{a.risk}</dd></div>
        <div>
          <dt className="eyebrow">Next step</dt>
          <dd className="mt-1 border-l-2 border-primary pl-3 leading-relaxed">{a.next_step}</dd>
        </div>
      </dl>
      <div className="mt-4 flex flex-wrap items-end justify-between gap-4 border-t hairline pt-3">
        <div className="min-w-0 space-y-1">
          <EvidenceDisclosure items={a.evidence} />
          {a.source?.id && (
            <p className="text-xs text-muted-foreground">
              From {a.source.type}{" "}
              {src ? <Link href={src} className="font-mono text-primary hover:underline">{a.source.id}</Link> : <span className="font-mono">{a.source.id}</span>}
            </p>
          )}
        </div>
        <FeedbackButtons targetType="action" targetId={a.id} verdicts={["useful", "not_useful"]} />
      </div>
    </article>
  );
}

export function ActionsBoard({ items }: { items: Action[] }) {
  return (
    <div className="space-y-10">
      {PRIORITIES.map(({ p, title, kicker }) => {
        const group = items.filter((a) => a.priority === p).sort((a, b) => b.priority_score - a.priority_score);
        if (!group.length) return null;
        return (
          <section key={p} aria-labelledby={`h-${p}`}>
            <SectionHeader id={`h-${p}`} kicker={`${kicker} · ${group.length} item${group.length === 1 ? "" : "s"}`} title={title} />
            <div className="space-y-4">{group.map((a) => <ActionCard key={a.id} a={a} />)}</div>
          </section>
        );
      })}
    </div>
  );
}
