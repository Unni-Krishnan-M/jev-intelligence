"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

import { fmtDate, Panel, SpecRows } from "@/components/jev/admin/ui";
import { ConfidenceBadge, SeverityBadge, WARNING_STATUS_META, WarningStatusBadge } from "@/components/jev/intel/badges";
import { EvidenceList } from "@/components/jev/intel/evidence-list";
import { FeedbackButtons } from "@/components/jev/intel/feedback-buttons";
import { IntelError } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError, errorMessage } from "@/lib/api";
import { fmtValue, sourceHref, TRANSITIONS, useIntelRevalidate } from "@/lib/intel";
import type { IntelWarning, WarningStatus } from "@/lib/intel-types";

const ACTION_LABEL: Record<WarningStatus, string> = {
  new: "Reopen",
  acknowledged: "Acknowledge",
  investigating: "Start investigating",
  resolved: "Resolve",
  dismissed: "Dismiss",
};

function Lifecycle({ w, onChanged }: { w: IntelWarning; onChanged: () => Promise<unknown> }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState<WarningStatus | null>(null);
  const next = TRANSITIONS[w.status] ?? [];

  async function move(to: WarningStatus) {
    setBusy(to);
    try {
      await api<IntelWarning>(`/intel/warnings/${w.id}`, { method: "PATCH", json: { status: to, note: note.trim() || null } });
      toast.success(`Warning ${WARNING_STATUS_META[to].label.toLowerCase()}`);
      setNote("");
      await onChanged();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(null);
    }
  }

  if (!next.length) {
    return <p className="text-sm text-muted-foreground">{WARNING_STATUS_META[w.status].label} is a final state. If the condition fires again, a new warning opens and links back to this one.</p>;
  }
  return (
    <div className="space-y-3">
      <Input value={note} onChange={(e) => setNote(e.target.value)} maxLength={1000} placeholder="Note for the audit trail (optional)" aria-label="Status change note" />
      <div className="flex flex-wrap gap-2">
        {next.map((to) => {
          const m = WARNING_STATUS_META[to];
          return (
            <Button key={to} size="sm" variant={to === "dismissed" ? "outline" : to === next[0] ? "default" : "secondary"} disabled={busy !== null} onClick={() => move(to)}>
              <m.Icon aria-hidden /> {busy === to ? "Saving…" : ACTION_LABEL[to]}
            </Button>
          );
        })}
      </div>
    </div>
  );
}

export default function WarningDetailPage() {
  const { id } = useParams<{ id: string }>();
  const { data: w, error, mutate } = useSWR<IntelWarning>(id ? `/intel/warnings/${encodeURIComponent(id)}` : null);
  const revalidate = useIntelRevalidate();

  const back = (
    <Link href="/intel/warnings" className="mb-6 inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground">
      <ArrowLeft className="size-4" aria-hidden /> Triage queue
    </Link>
  );

  if (error instanceof ApiError && error.status === 404) {
    return <div>{back}<EmptyState title="Warning not found" body={`No warning with id ${id}.`} /></div>;
  }
  if (error) return <div>{back}<IntelError error={error} retry={() => mutate()} runBacked={false} /></div>;
  if (!w) {
    return (
      <div className="space-y-4">
        {back}
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-12 w-3/4" />
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2"><Skeleton className="h-48" /><Skeleton className="h-48" /></div>
      </div>
    );
  }

  const tr = w.trigger ?? {};
  const src = sourceHref(w.source);
  const history = [...(w.history ?? [])].sort((a, b) => (a.at < b.at ? 1 : -1));

  return (
    <div>
      {back}
      <header className="mb-8 border-b hairline pb-6">
        <p className="eyebrow break-all">Early warning · #{w.id} · {w.key}</p>
        <h1 className="font-display mt-2 text-[34px] leading-[1.02] tracking-tight text-balance sm:text-[44px]">{w.title}</h1>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <SeverityBadge severity={w.severity} prefix="severity" />
          <WarningStatusBadge status={w.status} />
          <ConfidenceBadge value={w.confidence} kind={w.confidence_kind} />
          <span className="num text-xs text-muted-foreground">seen ×{w.occurrences}</span>
        </div>
        <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-ink-2">{w.description}</p>
      </header>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-4">
          <Panel title="Trigger condition">
            {tr.condition && <p className="mb-3 text-sm">{tr.condition}</p>}
            <SpecRows
              rows={[
                { label: "Rule", value: tr.rule ?? "—" },
                { label: "Observed", value: fmtValue(tr.observed) },
                { label: "Threshold", value: fmtValue(tr.threshold) },
              ]}
            />
          </Panel>
          <Panel title="Recommended action">
            <p className="text-sm leading-relaxed">{w.recommended_action}</p>
            {src && (
              <p className="mt-3 text-xs text-muted-foreground">
                Raised by {w.source.type ?? "source"} <Link href={src} className="font-mono text-primary hover:underline">{w.source.id}</Link>
              </p>
            )}
          </Panel>
          <Panel title="Evidence">
            <EvidenceList items={w.evidence} />
          </Panel>
        </div>

        <div className="space-y-4">
          <Panel title="Move this warning">
            <Lifecycle w={w} onChanged={async () => { await mutate(); await revalidate(); }} />
          </Panel>
          <Panel title="Was this warning worth raising?">
            <FeedbackButtons targetType="warning" targetId={String(w.id)} verdicts={["useful", "not_useful", "false_positive"]} noteField="note" />
          </Panel>
          <Panel title="Occurrences">
            <SpecRows
              rows={[
                { label: "First seen", value: fmtDate(w.detected_at) },
                { label: "Last seen", value: fmtDate(w.last_seen_at) },
                { label: "Updated", value: fmtDate(w.updated_at) },
                { label: "Occurrences", value: w.occurrences },
                { label: "First run", value: <span title={w.first_seen_run_id}>{w.first_seen_run_id.slice(0, 8)}</span> },
                { label: "Last run", value: <span title={w.last_seen_run_id}>{w.last_seen_run_id.slice(0, 8)}</span> },
                { label: "Reopened from", value: w.reopened_from ? <Link href={`/intel/warnings/${w.reopened_from}`} className="text-primary hover:underline">#{w.reopened_from}</Link> : "—" },
              ]}
            />
          </Panel>
          <Panel title="Status history">
            {!history.length ? (
              <p className="text-sm text-muted-foreground">No status changes recorded.</p>
            ) : (
              <ol className="relative ml-1.5 space-y-4 border-l hairline pl-4">
                {history.map((h, i) => {
                  const m = WARNING_STATUS_META[h.to_status];
                  return (
                    <li key={`${h.at}-${i}`} className="relative">
                      <span className="absolute -left-[23px] top-0.5 grid size-3.5 place-items-center rounded-full bg-card" aria-hidden>
                        <m.Icon className="size-3.5 text-muted-foreground" />
                      </span>
                      <p className="text-sm">
                        {h.from_status ? <>{WARNING_STATUS_META[h.from_status].label} → </> : null}
                        <span className="font-medium">{m.label}</span>
                      </p>
                      <p className="font-mono text-xs text-muted-foreground">{fmtDate(h.at)} · {h.actor}</p>
                      {h.note && <p className="mt-1 text-sm text-ink-2">{h.note}</p>}
                    </li>
                  );
                })}
              </ol>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
