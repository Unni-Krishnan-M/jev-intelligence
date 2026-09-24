"use client";

import { ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

import { Button } from "@/components/ui/button";
import { api, errorMessage } from "@/lib/api";
import { isNotDeployed } from "@/lib/intel";
import type { MeFeedbackIn, MeFeedbackList, MeFeedbackRecord, MeFeedbackTarget, MeVerdict } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

const META: Record<MeVerdict, { label: string; Icon: typeof ThumbsUp }> = {
  accepted: { label: "Accept", Icon: ThumbsUp },
  rejected: { label: "Reject", Icon: ThumbsDown },
};

export const ME_FEEDBACK_KEY = "/me/intelligence/feedback?limit=200";

/** The caller's saved verdict on one target (newest first, so the first match wins). */
export function savedVerdict(list: MeFeedbackRecord[] | undefined, targetType: MeFeedbackTarget, targetId: string): MeVerdict | null {
  return list?.find((r) => r.target_type === targetType && r.target_id === targetId)?.verdict ?? null;
}

/**
 * Accept / reject on the strategy decision or one recommendation (POST /me/intelligence/feedback).
 * The saved verdict comes from GET /me/intelligence/feedback and is shown pressed; choosing the
 * other one replaces it (the API keeps one row per target).
 */
export function MeFeedback({ targetType, targetId, className, size = "sm" }: { targetType: MeFeedbackTarget; targetId: string; className?: string; size?: "sm" | "xs" }) {
  const list = useSWR<MeFeedbackList>(ME_FEEDBACK_KEY, { shouldRetryOnError: false, revalidateOnFocus: false });
  const [pending, setPending] = useState<MeVerdict | null>(null);
  const saved = savedVerdict(list.data?.items, targetType, targetId);
  const current = pending ?? saved;

  async function send(verdict: MeVerdict) {
    if (verdict === saved || pending) return;
    setPending(verdict);
    const body: MeFeedbackIn = { target_type: targetType, target_id: targetId, verdict, note: null };
    try {
      await api<MeFeedbackRecord>("/me/intelligence/feedback", { json: body });
      await list.mutate();
      toast.success(verdict === "accepted" ? "Recorded: accepted" : "Recorded: rejected");
    } catch (e) {
      toast.error(isNotDeployed(e) ? "Feedback is not available on this API yet." : errorMessage(e));
    } finally {
      setPending(null);
    }
  }

  return (
    <div className={cn("flex flex-wrap items-center gap-1.5", className)} role="group" aria-label={`Your verdict on this ${targetType}`}>
      {(Object.keys(META) as MeVerdict[]).map((v) => {
        const m = META[v];
        const chosen = current === v;
        return (
          <Button
            key={v}
            type="button"
            size="sm"
            variant={chosen ? "secondary" : "outline"}
            disabled={pending !== null}
            aria-pressed={chosen}
            onClick={() => send(v)}
            className={cn(size === "xs" && "h-7 px-2 text-xs", chosen && "border-primary/60 disabled:opacity-100")}
          >
            <m.Icon aria-hidden />
            {chosen && !pending ? (v === "accepted" ? "Accepted" : "Rejected") : m.label}
          </Button>
        );
      })}
    </div>
  );
}
