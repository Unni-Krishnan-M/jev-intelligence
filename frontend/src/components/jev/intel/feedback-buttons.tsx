"use client";

import { Check, Flag, ThumbsDown, ThumbsUp, X } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { useIntelDomain } from "@/components/jev/intel/domain-context";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api, errorMessage } from "@/lib/api";
import { humanize, useIntelRevalidate } from "@/lib/intel";
import type { FeedbackIn, FeedbackRecord, FeedbackTarget, Verdict } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

const VERDICT_META: Record<Verdict, { label: string; Icon: typeof Check }> = {
  correct: { label: "Correct", Icon: Check },
  incorrect: { label: "Incorrect", Icon: X },
  useful: { label: "Useful", Icon: ThumbsUp },
  not_useful: { label: "Not useful", Icon: ThumbsDown },
  false_positive: { label: "False positive", Icon: Flag },
};

/**
 * Operator verdict on a decision, warning, action or prediction (POST /intel/feedback).
 * Buttons disable as soon as one is pressed; they come back only if the request fails.
 */
export function FeedbackButtons({
  targetType,
  targetId,
  verdicts,
  noteField,
  notePlaceholder,
  className,
}: {
  targetType: FeedbackTarget;
  targetId: string;
  verdicts: Verdict[];
  /** optional free-text field sent with the verdict */
  noteField?: "note" | "outcome";
  notePlaceholder?: string;
  className?: string;
}) {
  const [sent, setSent] = useState<Verdict | null>(null);
  const [note, setNote] = useState("");
  const revalidate = useIntelRevalidate();
  const { q: dq } = useIntelDomain();

  async function send(verdict: Verdict) {
    setSent(verdict);
    const text = note.trim() || null;
    const body: FeedbackIn = {
      target_type: targetType,
      target_id: targetId,
      verdict,
      note: noteField === "note" ? text : null,
      outcome: noteField === "outcome" ? text : null,
    };
    try {
      await api<FeedbackRecord>(dq("/intel/feedback"), { json: body });
      toast.success(`Recorded: ${VERDICT_META[verdict].label.toLowerCase()}`);
      void revalidate();
    } catch (e) {
      setSent(null);
      toast.error(errorMessage(e));
    }
  }

  return (
    <div className={cn("space-y-2", className)}>
      {noteField && !sent && (
        <Input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          maxLength={noteField === "outcome" ? 500 : 1000}
          placeholder={notePlaceholder ?? (noteField === "outcome" ? "What actually happened? (optional)" : "Note (optional)")}
          aria-label={noteField === "outcome" ? "Outcome note" : "Feedback note"}
          className="h-8 max-w-md text-sm"
        />
      )}
      <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={`Feedback on this ${humanize(targetType)}`}>
        {verdicts.map((v) => {
          const m = VERDICT_META[v];
          const chosen = sent === v;
          return (
            <Button
              key={v}
              type="button"
              size="sm"
              variant={chosen ? "secondary" : "outline"}
              disabled={sent !== null}
              aria-pressed={chosen}
              onClick={() => send(v)}
              className={cn(chosen && "disabled:opacity-100")}
            >
              <m.Icon aria-hidden />
              {m.label}
            </Button>
          );
        })}
        {sent && <span className="text-xs text-muted-foreground">Thanks — recorded.</span>}
      </div>
    </div>
  );
}
