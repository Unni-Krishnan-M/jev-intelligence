"use client";

import { useId, useState } from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { reasonProblem } from "@/lib/ops";

/**
 * A confirm step for every operation that changes what is served. With `reason`, a written reason
 * is required (force-promote, rollback) and is passed to onConfirm; the confirm button stays
 * disabled until it is valid.
 */
export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  destructive = false,
  reason = false,
  reasonLabel = "Reason (recorded in the audit log)",
  children,
  onConfirm,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: React.ReactNode;
  confirmLabel: string;
  destructive?: boolean;
  reason?: boolean;
  reasonLabel?: string;
  children?: React.ReactNode;
  onConfirm: (reason: string | null) => Promise<void> | void;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [touched, setTouched] = useState(false);
  const id = useId();
  const problem = reason ? reasonProblem(text) : null;

  async function confirm() {
    setTouched(true);
    if (problem) return;
    setBusy(true);
    try {
      await onConfirm(reason ? text.trim() : null);
      setText("");
      setTouched(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        if (busy) return;
        if (!o) {
          setText("");
          setTouched(false);
        }
        onOpenChange(o);
      }}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="font-display text-2xl font-normal">{title}</DialogTitle>
          <DialogDescription asChild>
            <div className="text-sm leading-relaxed text-ink-2">{description}</div>
          </DialogDescription>
        </DialogHeader>
        {children}
        {reason && (
          <div className="space-y-1.5">
            <Label htmlFor={`${id}-reason`} className="text-xs font-normal text-muted-foreground">{reasonLabel}</Label>
            <textarea
              id={`${id}-reason`}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onBlur={() => setTouched(true)}
              rows={3}
              maxLength={500}
              required
              aria-invalid={touched && Boolean(problem)}
              aria-describedby={`${id}-reason-help`}
              className="w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 aria-[invalid=true]:border-destructive"
              placeholder="e.g. incident 42: negative feedback spike after the last promotion"
            />
            <p id={`${id}-reason-help`} className={touched && problem ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>
              {touched && problem ? problem : `${text.trim().length} / 500 · required`}
            </p>
          </div>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>Cancel</Button>
          <Button variant={destructive ? "destructive" : "default"} onClick={confirm} disabled={busy || (reason && Boolean(problem))}>
            {busy ? "Working…" : confirmLabel}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
