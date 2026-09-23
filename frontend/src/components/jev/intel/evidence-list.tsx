import { ArrowUpRight } from "lucide-react";
import Link from "next/link";

import { fmtValue, refHref } from "@/lib/intel";
import type { Evidence } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

const KIND_LABEL: Record<Evidence["kind"], string> = {
  metric: "metric",
  series: "series",
  record: "record",
  test: "test",
  model: "model",
};

/** Evidence rows: kind, label, value, detail; a ref links to the chart or object it came from. */
export function EvidenceList({ items, className, empty = "No evidence recorded." }: { items: Evidence[] | null | undefined; className?: string; empty?: string }) {
  if (!items?.length) return <p className={cn("text-sm text-muted-foreground", className)}>{empty}</p>;
  return (
    <ul className={cn("divide-y hairline border-y hairline text-sm", className)}>
      {items.map((e, i) => {
        const href = refHref(e.ref);
        return (
          <li key={`${e.label}-${i}`} className="grid grid-cols-1 gap-x-4 gap-y-0.5 py-2 sm:grid-cols-[72px_1fr_auto]">
            <span className="eyebrow pt-0.5">{KIND_LABEL[e.kind] ?? e.kind}</span>
            <div className="min-w-0">
              <p className="text-foreground">{e.label}</p>
              {e.detail && <p className="mt-0.5 text-xs text-muted-foreground">{e.detail}</p>}
              {e.ref && (
                href ? (
                  <Link href={href} className="mt-0.5 inline-flex items-center gap-0.5 break-all font-mono text-xs text-primary hover:underline">
                    {e.ref}
                    <ArrowUpRight className="size-3" aria-hidden />
                  </Link>
                ) : (
                  <p className="mt-0.5 break-all font-mono text-xs text-muted-foreground">{e.ref}</p>
                )
              )}
            </div>
            {e.value !== null && e.value !== undefined && <span className="num text-right text-ink-2 sm:pt-0.5">{fmtValue(e.value)}</span>}
          </li>
        );
      })}
    </ul>
  );
}

/** Native disclosure for evidence under a row: keyboard-accessible, no state to manage. */
export function EvidenceDisclosure({ items, label = "Evidence", className }: { items: Evidence[] | null | undefined; label?: string; className?: string }) {
  const n = items?.length ?? 0;
  return (
    <details className={cn("group", className)}>
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
        <span className="transition-transform group-open:rotate-90" aria-hidden>›</span>
        {label} <span className="num">({n})</span>
      </summary>
      <EvidenceList items={items} className="mt-2" />
    </details>
  );
}
