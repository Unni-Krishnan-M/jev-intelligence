"use client";

import { Info } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { SIGNAL_META, SIGNALS } from "@/lib/signals";
import type { RecItem } from "@/lib/types";

import { RecConfidence } from "./rec-confidence";
import { SignalBar } from "./signal-bar";

/** "Why this?" — the exact numbers the ranker used, plus the reasons derived from them. */
export function WhyDialog({ item, modelVersion, trigger }: { item: RecItem; modelVersion?: string; trigger?: React.ReactNode }) {
  const rows = SIGNALS.map((k) => ({ k, ...item.signals[k] })).sort((a, b) => b.contribution - a.contribution);
  return (
    <Dialog>
      <DialogTrigger asChild>
        {trigger ?? (
          <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs text-muted-foreground">
            <Info className="size-3.5" /> Why this?
          </Button>
        )}
      </DialogTrigger>
      <DialogContent className="sm:max-w-xl">
        <DialogHeader>
          <p className="eyebrow">Rank {item.rank} · score <span className="num">{item.score.toFixed(3)}</span></p>
          <DialogTitle className="font-display text-3xl font-normal leading-tight">{item.title}</DialogTitle>
          <DialogDescription className="text-[15px] text-foreground">{item.reason}</DialogDescription>
        </DialogHeader>
        {item.secondary_reasons.length > 0 && (
          <ul className="-mt-1 space-y-1 text-sm text-ink-2">
            {item.secondary_reasons.map((r) => <li key={r}>— {r}</li>)}
          </ul>
        )}
        <RecConfidence item={item} />
        <div className="mt-2">
          <p className="eyebrow mb-2">How the hybrid score was built</p>
          <SignalBar signals={item.signals} height={10} />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b hairline text-left text-xs text-muted-foreground">
                <th className="py-2 pr-2 font-normal">Signal</th>
                <th className="py-2 pr-2 text-right font-normal">raw</th>
                <th className="py-2 pr-2 text-right font-normal">norm</th>
                <th className="py-2 pr-2 text-right font-normal">weight</th>
                <th className="py-2 text-right font-normal">contrib.</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.k} className="border-b hairline last:border-0">
                  <td className="py-2 pr-2">
                    <span className="flex items-center gap-2">
                      <span className="size-2 rounded-full" style={{ background: SIGNAL_META[r.k].color }} aria-hidden />
                      <span>{SIGNAL_META[r.k].label}</span>
                      <span className="hidden text-xs text-muted-foreground sm:inline">{SIGNAL_META[r.k].model}</span>
                    </span>
                  </td>
                  <td className="num py-2 pr-2 text-right text-muted-foreground">{r.raw.toFixed(3)}</td>
                  <td className="num py-2 pr-2 text-right">{r.normalized.toFixed(2)}</td>
                  <td className="num py-2 pr-2 text-right">{r.weight.toFixed(2)}</td>
                  <td className="num py-2 text-right">{r.contribution.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs leading-relaxed text-muted-foreground">
          Each signal is min-max normalised across this request&apos;s candidates, then weighted. Weights adapt to how much
          JEV knows about you. The reason names the strongest personal signal that has concrete evidence behind it.
          {modelVersion ? <> Model <span className="num">{modelVersion}</span>.</> : null}
        </p>
      </DialogContent>
    </Dialog>
  );
}
