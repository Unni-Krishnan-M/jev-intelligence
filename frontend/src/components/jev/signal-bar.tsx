import { SIGNAL_META, SIGNALS } from "@/lib/signals";
import type { SignalName, SignalValue } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Stacked bar of each signal's contribution to the hybrid score. Segments are separated by a 2px
 * surface gap, and colour follows the fixed signal order. Always shipped with a legend or a table,
 * so colour is never the only channel.
 */
export function SignalBar({ signals, className, height = 6 }: { signals: Record<SignalName, SignalValue>; className?: string; height?: number }) {
  const total = SIGNALS.reduce((s, k) => s + Math.max(signals[k]?.contribution ?? 0, 0), 0) || 1;
  return (
    <div className={cn("flex w-full gap-[2px] overflow-hidden rounded-full", className)} style={{ height }} role="presentation">
      {SIGNALS.map((k) => {
        const v = Math.max(signals[k]?.contribution ?? 0, 0);
        if (v <= 0) return null;
        return <span key={k} style={{ width: `${(v / total) * 100}%`, background: SIGNAL_META[k].color }} title={`${SIGNAL_META[k].short}: ${v.toFixed(3)}`} />;
      })}
    </div>
  );
}

export function WeightBar({ weights, className }: { weights: Record<SignalName, number>; className?: string }) {
  return (
    <div className={cn("space-y-2", className)}>
      <div className="flex h-2.5 w-full gap-[2px] overflow-hidden rounded-full">
        {SIGNALS.map((k) => (weights[k] ?? 0) > 0 && (
          <span key={k} style={{ width: `${(weights[k] ?? 0) * 100}%`, background: SIGNAL_META[k].color }} />
        ))}
      </div>
      <ul className="flex flex-wrap gap-x-4 gap-y-1">
        {SIGNALS.map((k) => (
          <li key={k} className="flex items-center gap-1.5 text-xs text-ink-2">
            <span className="size-2 rounded-full" style={{ background: SIGNAL_META[k].color }} aria-hidden />
            {SIGNAL_META[k].short}
            <span className="num text-muted-foreground">{((weights[k] ?? 0) * 100).toFixed(0)}%</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
