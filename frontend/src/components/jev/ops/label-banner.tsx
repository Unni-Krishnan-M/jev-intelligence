import { FlaskConical, Radio } from "lucide-react";

import { labelBanner } from "@/lib/ops";
import { cn } from "@/lib/utils";

/** The results' data-source label, shown above everything: live traffic, or an offline replay in capitals. */
export function LabelBanner({ dataSource, label }: { dataSource: string | null | undefined; label: string | null | undefined }) {
  const b = labelBanner(dataSource, label);
  const Icon = b.kind === "replay" ? FlaskConical : Radio;
  return (
    <div role="note" aria-label="Data source" className={cn("flex items-start gap-3 rounded-lg border px-4 py-3", b.kind === "replay" ? "border-foreground/40 bg-muted" : "border-primary/50 bg-primary/5")}>
      <Icon className={cn("mt-0.5 size-5 shrink-0", b.kind === "replay" ? "text-foreground" : "text-primary")} aria-hidden />
      <div className="min-w-0">
        <p className={cn("text-sm", b.kind === "replay" ? "font-mono font-semibold tracking-wide" : "font-medium")}>{b.title}</p>
        <p className="mt-0.5 text-xs text-ink-2">{b.body}</p>
      </div>
    </div>
  );
}

