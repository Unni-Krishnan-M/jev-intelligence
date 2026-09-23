"use client";

/*
 * Audit log. Structure follows common audit-trail viewers (filters for action and actor in one
 * row above the table; columns time UTC · action · actor · target · request id; failures set
 * apart by icon and word, not colour; the raw detail one click away; an empty state that says
 * the filters matched nothing). Visual style stays JEV's.
 */

import { CheckCircle2, Info, LogIn, Search, X, XCircle } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import useSWR from "swr";

import { fmtDate } from "@/components/jev/admin/ui";
import { PageHeader } from "@/components/jev/intel/page-header";
import { FilterRow, IntelError, Pagination, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { qs } from "@/lib/api";
import { fmtValue, ownerHref } from "@/lib/intel";
import type { AuditEntry, AuditPage } from "@/lib/intel-types";

const LIMIT = 50;
/** The actions the API writes (docs/intelligence.md 9.3). */
const ACTIONS = ["all", "intel.run", "warning.transition", "feedback.create", "scenario.save", "model.activate", "auth.login.success", "auth.login.failure", "auth.register"];

function ActionChip({ action }: { action: string }) {
  const failed = action.endsWith(".failure");
  const Icon = failed ? XCircle : action.startsWith("auth.") ? LogIn : action.endsWith(".success") ? CheckCircle2 : Info;
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap rounded border hairline px-1.5 py-0.5 font-mono text-xs">
      <Icon className={failed ? "size-3.5 text-destructive" : "size-3.5 text-muted-foreground"} aria-hidden />
      {action}
      {failed && <span className="sr-only"> (failed)</span>}
    </span>
  );
}

function targetHref(e: AuditEntry): string | null {
  if (!e.target_type || !e.target_id) return null;
  if (e.target_type === "model") return "/admin/models";
  if (["signal", "trend", "anomaly", "forecast", "risk", "decision", "warning", "action"].includes(e.target_type)) return ownerHref(e.target_type, e.target_id);
  return null;
}

function Detail({ detail }: { detail: AuditEntry["detail"] }) {
  const entries = Object.entries(detail ?? {});
  if (!entries.length) return <span className="text-muted-foreground">—</span>;
  return (
    <details className="group">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
        <span className="transition-transform group-open:rotate-90" aria-hidden>›</span> {entries.length} field{entries.length === 1 ? "" : "s"}
      </summary>
      <dl className="mt-1.5 space-y-0.5 text-xs">
        {entries.map(([k, v]) => (
          <div key={k} className="flex gap-2">
            <dt className="shrink-0 text-muted-foreground">{k}</dt>
            <dd className="num min-w-0 break-all">{typeof v === "object" && v !== null && !Array.isArray(v) ? JSON.stringify(v) : fmtValue(v, k)}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}

export default function AuditPage() {
  const [action, setAction] = useState("all");
  const [actorText, setActorText] = useState("");
  const [actor, setActor] = useState("");
  const [targetType, setTargetType] = useState("");
  const [offset, setOffset] = useState(0);

  useEffect(() => {
    const t = setTimeout(() => {
      setActor(actorText.trim());
      setOffset(0);
    }, 300);
    return () => clearTimeout(t);
  }, [actorText]);

  const { data, error, mutate, isValidating } = useSWR<AuditPage>(`/admin/audit${qs({ action: action === "all" ? null : action, actor, target_type: targetType, limit: LIMIT, offset })}`, { keepPreviousData: true });
  const filtered = action !== "all" || Boolean(actor) || Boolean(targetType);

  return (
    <div>
      <PageHeader
        eyebrow="trace"
        title="Audit log"
        description="Who did what, when: pipeline runs, warning transitions, feedback, saved scenarios, model activations and sign-ins. Secrets and passwords are never recorded."
      />

      <div className="mb-5 flex flex-col gap-2.5">
        <form role="search" onSubmit={(e) => { e.preventDefault(); setActor(actorText.trim()); setOffset(0); }} className="relative max-w-sm">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden />
          <Input type="search" value={actorText} onChange={(e) => setActorText(e.target.value)} placeholder="Actor (email or system)" aria-label="Filter by actor" className="h-9 pl-8" />
        </form>
        <FilterRow label="Action" value={action} onChange={(v) => { setAction(v); setOffset(0); }} options={ACTIONS.map((a) => ({ value: a, label: a === "all" ? "All" : a }))} />
        {targetType && (
          <p className="flex items-center gap-2 text-xs">
            <span className="eyebrow">Target</span>
            <Button variant="outline" size="sm" className="h-7 gap-1 px-2 text-xs" onClick={() => { setTargetType(""); setOffset(0); }} aria-label={`Clear target filter ${targetType}`}>
              {targetType} <X aria-hidden />
            </Button>
          </p>
        )}
      </div>

      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} what="the audit log (GET /admin/audit)" />
      ) : !data ? (
        <RowsSkeleton rows={8} />
      ) : data.items.length === 0 ? (
        <EmptyState title="No audit entries" body={filtered ? "Nothing matches these filters." : "Nothing has been recorded yet."} />
      ) : (
        <>
          <p className="num mb-2 text-xs text-muted-foreground">{data.total.toLocaleString()} entries · times in UTC</p>
          <div className={`overflow-x-auto rounded-lg border bg-card transition-opacity ${isValidating ? "opacity-70" : ""}`}>
            <table className="w-full min-w-[820px] text-sm">
              <caption className="sr-only">Audit log</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2.5 font-normal">Time</th>
                  <th className="px-2 py-2.5 font-normal">Action</th>
                  <th className="px-2 py-2.5 font-normal">Actor</th>
                  <th className="px-2 py-2.5 font-normal">Target</th>
                  <th className="px-2 py-2.5 font-normal">Detail</th>
                  <th className="px-4 py-2.5 font-normal">Request</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((e) => {
                  const href = targetHref(e);
                  return (
                    <tr key={e.id} className="border-b hairline align-top last:border-0">
                      <td className="num whitespace-nowrap px-4 py-2.5 text-xs text-muted-foreground">{fmtDate(e.at)}</td>
                      <td className="px-2 py-2.5"><ActionChip action={e.action} /></td>
                      <td className="px-2 py-2.5 text-xs">
                        {e.actor ? (
                          <button type="button" className="break-all text-left hover:underline" onClick={() => setActorText(e.actor ?? "")} title="Show only this actor">{e.actor}</button>
                        ) : (
                          <span className="text-muted-foreground">anonymous</span>
                        )}
                      </td>
                      <td className="px-2 py-2.5">
                        {e.target_type ? (
                          <button type="button" className="eyebrow block hover:text-foreground" onClick={() => { setTargetType(e.target_type ?? ""); setOffset(0); }} title="Show only this target type">{e.target_type}</button>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                        {e.target_id && (href ? <Link href={href} className="break-all font-mono text-xs text-primary hover:underline">{e.target_id}</Link> : <span className="break-all font-mono text-xs">{e.target_id}</span>)}
                      </td>
                      <td className="max-w-[260px] px-2 py-2.5"><Detail detail={e.detail} /></td>
                      <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground" title={e.request_id ?? undefined}>{e.request_id ? e.request_id.slice(0, 12) : "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <Pagination total={data.total} limit={data.limit ?? LIMIT} offset={offset} onChange={setOffset} />
        </>
      )}
    </div>
  );
}
