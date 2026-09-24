"use client";

import {
  Archive,
  Ban,
  CheckCircle2,
  CircleDashed,
  CircleDot,
  Clock,
  FlaskConical,
  Hourglass,
  Loader2,
  MinusCircle,
  OctagonAlert,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Square,
  Stamp,
  XCircle,
} from "lucide-react";

import {
  EXPERIMENT_STATUS_LABEL,
  GATE_VERDICT_LABEL,
  gateStatusMeta,
  JOB_STATUS_LABEL,
  LAG_LABEL,
  MODEL_STATE_LABEL,
  conclusionWording,
  type GateVerdict,
  type LagLevel,
} from "@/lib/ops";
import type { Conclusion, ExperimentStatus, GateStatus, JobStatus, ModelState } from "@/lib/ops-types";
import { cn } from "@/lib/utils";

/*
 * Status in the operations pages is always an icon + a word; colour only tints the icon, exactly
 * like the console's badges (components/jev/intel/badges.tsx).
 */

export const CHIP = "inline-flex shrink-0 items-center gap-1.5 whitespace-nowrap rounded border hairline px-1.5 py-0.5 text-xs text-foreground";

type Icon = typeof CheckCircle2;

function Chip({ Icon, label, color, spin, className, title }: { Icon: Icon; label: string; color?: string; spin?: boolean; className?: string; title?: string }) {
  return (
    <span className={cn(CHIP, className)} title={title}>
      <Icon className={cn("size-3.5", !color && "text-muted-foreground", spin && "motion-safe:animate-spin")} style={color ? { color } : undefined} aria-hidden />
      {label}
    </span>
  );
}

const TONE_COLOR = { good: "var(--status-good)", bad: "var(--status-critical)", neutral: undefined } as const;

export function GateStatusBadge({ status, className }: { status: GateStatus | string | null | undefined; className?: string }) {
  const m = gateStatusMeta(status);
  const Icon = m.tone === "good" ? CheckCircle2 : m.tone === "bad" ? XCircle : status === "skipped" ? MinusCircle : CircleDashed;
  return <Chip Icon={Icon} label={m.label} color={TONE_COLOR[m.tone]} className={className} />;
}

const VERDICT: Record<GateVerdict, { Icon: Icon; color?: string }> = {
  pass: { Icon: ShieldCheck, color: "var(--status-good)" },
  fail: { Icon: ShieldAlert, color: "var(--status-critical)" },
  not_evaluated: { Icon: CircleDashed },
  stale_pass: { Icon: Clock, color: "var(--status-serious)" },
  stale_fail: { Icon: ShieldAlert, color: "var(--status-critical)" },
};

export function GateVerdictBadge({ verdict, className }: { verdict: GateVerdict; className?: string }) {
  const m = VERDICT[verdict];
  const title = verdict.startsWith("stale") ? "Gated against a model that no longer serves: evaluate again before promoting." : undefined;
  return <Chip Icon={m.Icon} label={GATE_VERDICT_LABEL[verdict]} color={m.color} className={className} title={title} />;
}

const STATE: Record<ModelState, { Icon: Icon; color?: string }> = {
  active: { Icon: Radio, color: "var(--primary)" },
  candidate: { Icon: CircleDot },
  rejected: { Icon: Ban, color: "var(--status-critical)" },
  retired: { Icon: Archive },
};

export function ModelStateBadge({ state, className }: { state: ModelState; className?: string }) {
  const m = STATE[state] ?? STATE.retired;
  return <Chip Icon={m.Icon} label={MODEL_STATE_LABEL[state] ?? state} color={m.color} className={cn(state === "active" && "border-primary/60", className)} />;
}

const JOB: Record<JobStatus, { Icon: Icon; color?: string; spin?: boolean }> = {
  queued: { Icon: Hourglass },
  running: { Icon: Loader2, color: "var(--primary)", spin: true },
  succeeded: { Icon: CheckCircle2, color: "var(--status-good)" },
  failed: { Icon: XCircle, color: "var(--status-critical)" },
};

export function JobStatusBadge({ status, className }: { status: JobStatus; className?: string }) {
  const m = JOB[status] ?? JOB.queued;
  return <Chip Icon={m.Icon} label={JOB_STATUS_LABEL[status] ?? status} color={m.color} spin={m.spin} className={className} />;
}

const EXP: Record<ExperimentStatus, { Icon: Icon; color?: string }> = {
  draft: { Icon: CircleDashed },
  running: { Icon: Play, color: "var(--primary)" },
  paused: { Icon: Pause, color: "var(--status-warning)" },
  stopped: { Icon: Square },
  concluded: { Icon: Stamp },
};

export function ExperimentStatusBadge({ status, className }: { status: ExperimentStatus; className?: string }) {
  const m = EXP[status] ?? EXP.draft;
  return <Chip Icon={m.Icon} label={EXPERIMENT_STATUS_LABEL[status] ?? status} color={m.color} className={cn(status === "running" && "border-primary/60", className)} />;
}

const LAG: Record<LagLevel, { Icon: Icon; color?: string }> = {
  ok: { Icon: CheckCircle2, color: "var(--status-good)" },
  watch: { Icon: Clock, color: "var(--status-warning)" },
  stale: { Icon: OctagonAlert, color: "var(--status-serious)" },
  none: { Icon: CircleDashed },
};

export function LagBadge({ level, className }: { level: LagLevel; className?: string }) {
  const m = LAG[level];
  return <Chip Icon={m.Icon} label={LAG_LABEL[level]} color={m.color} className={className} />;
}

export function RunModeBadge({ mode, className }: { mode: "live" | "replay" | string | null | undefined; className?: string }) {
  if (mode === "replay") return <Chip Icon={RotateCcw} label="Replay" className={className} title="Run for an explicit as-of date: a historical replay, not the live state" />;
  if (mode === "live") return <Chip Icon={Radio} label="Live" color="var(--primary)" className={className} title="Run on the latest data" />;
  return null;
}

export function GuardrailBadge({ breached, className }: { breached: boolean; className?: string }) {
  return breached ? (
    <Chip Icon={ShieldAlert} label="Breached" color="var(--status-critical)" className={className} />
  ) : (
    <Chip Icon={ShieldCheck} label="Holding" color="var(--status-good)" className={className} />
  );
}

export function ConclusionBadge({ conclusion, className }: { conclusion: Pick<Conclusion, "decision" | "winner">; className?: string }) {
  const w = conclusionWording(conclusion);
  const Icon = w.tone === "good" ? CheckCircle2 : w.tone === "bad" ? XCircle : MinusCircle;
  return <Chip Icon={Icon} label={w.title} color={TONE_COLOR[w.tone]} className={className} />;
}

export function DataSourceBadge({ kind, className }: { kind: "live" | "replay"; className?: string }) {
  return kind === "replay" ? <Chip Icon={FlaskConical} label="Offline replay" className={className} /> : <Chip Icon={Radio} label="Live traffic" color="var(--primary)" className={className} />;
}
