"use client";

/*
 * Model lifecycle (docs/RETRAINING_AND_MODEL_GOVERNANCE.md): what serves and what served before
 * (rollback behind a reason), every version with its state and gate, the per-gate evidence (CI
 * against the non-inferiority margin, staleness), the lineage of a version, retrain and evaluate
 * jobs (polled while one runs) and the training snapshots.
 */

import { FileSearch, Hammer, History, Package, RotateCcw, ShieldCheck, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import useSWR, { useSWRConfig } from "swr";

import { fmtDate, Panel, SpecRows } from "@/components/jev/admin/ui";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, PanelsSkeleton, RowsSkeleton } from "@/components/jev/intel/states";
import { GateStatusBadge, GateVerdictBadge, JobStatusBadge, ModelStateBadge } from "@/components/jev/ops/badges";
import { ConfirmDialog } from "@/components/jev/ops/confirm-dialog";
import { IntervalKey, IntervalPlot, intervalDomain } from "@/components/jev/ops/interval-plot";
import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError, errorMessage } from "@/lib/api";
import { fmtValue } from "@/lib/intel";
import { anyJobActive, GATE_LABEL, gateCounts, gateVerdict, isGateStale, nonInferiority, orderedGates, promoteControls, stepSummary } from "@/lib/ops";
import type { GovernedModel, GovernedModelList, ModelLineage, PromotionResult, SnapshotOut, TrainingJobOut } from "@/lib/ops-types";
import { cn } from "@/lib/utils";

const HEAD = ["ndcg@10", "recall@10", "coverage@10"] as const;
const short = (v: string | null | undefined) => (v ? v.replace(/^jev-/, "") : "—");

function useRevalidateGovernance() {
  const { mutate } = useSWRConfig();
  return () => mutate((k) => typeof k === "string" && k.startsWith("/governance"));
}

function failToast(e: unknown) {
  const blockers = e instanceof ApiError && Array.isArray(e.details) ? (e.details as string[]) : [];
  toast.error(errorMessage(e), blockers.length ? { description: blockers.join(" · ") } : undefined);
}

// ---- active / previous + rollback -----------------------------------------------------------

function ServingPanel({ list }: { list: GovernedModelList }) {
  const [open, setOpen] = useState(false);
  const revalidate = useRevalidateGovernance();
  const active = list.items.find((m) => m.version === list.active);
  const previous = list.items.find((m) => m.version === list.previous);

  async function rollback(reason: string | null) {
    try {
      const r = await api<PromotionResult>("/governance/models/rollback", { json: { reason } });
      toast.success(`Rolled back to ${r.version}`);
      setOpen(false);
      await revalidate();
    } catch (e) {
      failToast(e);
    }
  }

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Panel title="Serving now">
        {active ? (
          <>
            <p className="font-display break-all text-2xl leading-tight">{active.version}</p>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <ModelStateBadge state="active" />
              {active.forced && <span className="text-xs text-muted-foreground">force-promoted: {active.force_reason}</span>}
            </div>
            <SpecRows
              rows={[
                { label: "Promoted", value: active.promoted_at ? `${fmtDate(active.promoted_at)} · ${active.promoted_by ?? "—"}` : "before governance" },
                ...HEAD.map((k) => ({ label: `${k} (training test)`, value: fmtValue(active.metrics[k]) })),
                { label: "Snapshot", value: active.snapshot_id ?? "—" },
              ]}
            />
          </>
        ) : (
          <p className="text-sm text-muted-foreground">{list.active ? `${list.active} (not in the governance table)` : "No model is active. Promote a candidate to start serving."}</p>
        )}
      </Panel>
      <Panel title="Rollback target" action={
        <Button size="sm" variant="outline" disabled={!list.previous} onClick={() => setOpen(true)}><RotateCcw aria-hidden /> Roll back…</Button>
      }>
        {list.previous ? (
          <>
            <p className="font-display break-all text-2xl leading-tight">{list.previous}</p>
            <div className="mt-2">{previous && <ModelStateBadge state={previous.state} />}</div>
            <p className="mt-3 text-sm text-ink-2">It served before the active model, so it is restored without a new gate. A second rollback walks further back.</p>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Nothing to roll back to: no version served before the active one under governance.</p>
        )}
      </Panel>
      <ConfirmDialog
        open={open}
        onOpenChange={setOpen}
        title={`Roll back to ${short(list.previous)}?`}
        description={<>The active model <span className="font-mono">{short(list.active)}</span> stops serving and is retired; <span className="font-mono">{short(list.previous)}</span> is loaded and swapped in, and recommendation caches are cleared. Audited as model.rollback.</>}
        confirmLabel="Roll back"
        destructive
        reason
        onConfirm={rollback}
      />
    </div>
  );
}

// ---- promote / evaluate ---------------------------------------------------------------------

function PromoteButtons({ m, busyJob }: { m: GovernedModel; busyJob: boolean }) {
  const [mode, setMode] = useState<"promote" | "force" | null>(null);
  const revalidate = useRevalidateGovernance();
  const c = promoteControls(m);

  async function promote(reason: string | null) {
    try {
      const r = await api<PromotionResult>(`/governance/models/${encodeURIComponent(m.version)}/promote`, { json: mode === "force" ? { force: true, reason } : {} });
      toast.success(`${r.version} is now serving${r.forced ? " (forced)" : ""}`);
      setMode(null);
      await revalidate();
    } catch (e) {
      failToast(e);
    }
  }
  async function evaluate() {
    try {
      await api<TrainingJobOut>(`/governance/models/${encodeURIComponent(m.version)}/evaluate`, { json: { quick: true } });
      toast.success(`Evaluating ${short(m.version)} against the active model`);
      await revalidate();
    } catch (e) {
      failToast(e);
    }
  }

  if (m.is_active) return <span className="text-xs text-muted-foreground">serving</span>;
  return (
    <div className="flex flex-wrap justify-end gap-1.5">
      <Button size="sm" variant="outline" onClick={evaluate} disabled={busyJob} title={busyJob ? "A job is running" : "Gate this version against the active model"}>
        <ShieldCheck aria-hidden /> Evaluate
      </Button>
      {c.canPromote ? (
        <Button size="sm" onClick={() => setMode("promote")}><Upload aria-hidden /> Promote</Button>
      ) : (
        <>
          <Button size="sm" disabled title={c.why.join("; ")} aria-describedby={`why-${m.version}`}><Upload aria-hidden /> Promote</Button>
          <span id={`why-${m.version}`} className="sr-only">Blocked: {c.why.join("; ")}</span>
          {c.canForce && <Button size="sm" variant="ghost" className="text-destructive" onClick={() => setMode("force")}>Force…</Button>}
        </>
      )}
      <ConfirmDialog
        open={mode !== null}
        onOpenChange={(o) => !o && setMode(null)}
        title={mode === "force" ? `Force-promote ${short(m.version)}?` : `Promote ${short(m.version)}?`}
        description={
          mode === "force" ? (
            <>
              <p>This bypasses the gate (never the load check). It is blocked because:</p>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-xs">{c.why.map((w) => <li key={w}>{w}</li>)}</ul>
              <p className="mt-2">The reason is stored on the version and in the audit log.</p>
            </>
          ) : (
            "The gate passed against the model serving now. The version is loaded, swapped in under a lock, and recommendation caches are cleared."
          )
        }
        confirmLabel={mode === "force" ? "Force-promote" : "Promote"}
        destructive={mode === "force"}
        reason={mode === "force"}
        onConfirm={promote}
      />
    </div>
  );
}

// ---- gate panel -------------------------------------------------------------------------------

function GatePanel({ m, active }: { m: GovernedModel; active: string | null }) {
  const gates = orderedGates(m.gates);
  const ni = gates.map(([, g]) => nonInferiority(g)).filter((x) => x !== null);
  const domain = intervalDomain(ni.map((x) => ({ diff: x.diff, lo: x.lo, hi: x.hi })), Math.max(0, ...ni.map((x) => x.margin)));
  const level = ni[0]?.level ?? null;
  const stale = isGateStale(m, active);
  return (
    <Panel title={<>Gate · <span className="font-mono normal-case tracking-normal">{short(m.version)}</span></>} action={<GateVerdictBadge verdict={gateVerdict(m, active)} />}>
      {!gates.length ? (
        <p className="text-sm text-muted-foreground">This version has not been gated. Evaluate it to compare it with the active model on the candidate&apos;s frozen split.</p>
      ) : (
        <>
          <p className="text-sm text-ink-2">
            Gated against <span className="font-mono text-xs">{m.gated_against ?? "no model (bootstrap)"}</span>
            {m.gated_at ? ` on ${fmtDate(m.gated_at)} UTC` : ""}.
            {stale && <strong className="ml-1 font-semibold">That model no longer serves, so this result is stale: evaluate again before promoting.</strong>}
          </p>
          <ul className="mt-4 divide-y hairline">
            {gates.map(([name, g]) => {
              const n = nonInferiority(g);
              const other = Object.entries(g.numbers).filter(([k]) => !["metric", "ci_level"].includes(k) && !(n && ["diff", "ci_lo", "ci_hi", "margin", "p_value", "n_users", "candidate", "incumbent"].includes(k)));
              return (
                <li key={name} className="py-3 first:pt-0 last:pb-0">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <p className="text-sm font-medium">{GATE_LABEL[name] ?? name}</p>
                    <GateStatusBadge status={g.status} />
                  </div>
                  {n && (
                    <div className="mt-2 space-y-1">
                      <IntervalPlot row={{ diff: n.diff, lo: n.lo, hi: n.hi }} domain={domain} margin={n.margin} format={(v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(4)} label={`${name} candidate − incumbent`} highlight={g.status === "fail"} />
                      <p className="num text-xs text-ink-2">
                        Δ {n.diff >= 0 ? "+" : "−"}{Math.abs(n.diff).toFixed(4)} · CI [{n.lo.toFixed(4)}, {n.hi.toFixed(4)}] vs −{n.margin}
                        {n.p !== null ? ` · p ${n.p.toFixed(3)}` : ""}{n.n !== null ? ` · ${n.n} users` : ""}
                      </p>
                    </div>
                  )}
                  {n && typeof g.numbers.candidate === "number" && (
                    <p className="num mt-1 text-xs text-muted-foreground">candidate {fmtValue(g.numbers.candidate)} · incumbent {fmtValue(g.numbers.incumbent)}</p>
                  )}
                  {other.length > 0 && (
                    <p className="num mt-1 break-words text-xs text-muted-foreground">{other.map(([k, v]) => `${k.replaceAll("_", " ")} ${fmtValue(v, k)}`).join(" · ")}</p>
                  )}
                  {g.reason && <p className="mt-1 text-xs text-destructive">{g.reason}</p>}
                </li>
              );
            })}
          </ul>
          {ni.length > 0 && <div className="mt-4"><IntervalKey margin level={level} /></div>}
          <p className="mt-2 text-xs text-muted-foreground">A non-inferiority gate passes when the CI&apos;s lower bound stays right of the dashed margin: the candidate may be slightly worse, never meaningfully worse.</p>
        </>
      )}
    </Panel>
  );
}

// ---- lineage drawer ---------------------------------------------------------------------------

function LineageDrawer({ version, onClose }: { version: string | null; onClose: () => void }) {
  const { data, error, mutate } = useSWR<ModelLineage>(version ? `/governance/models/${encodeURIComponent(version)}/lineage` : null);
  const lin = (data?.lineage ?? {}) as Record<string, unknown>;
  return (
    <Sheet open={version !== null} onOpenChange={(o) => !o && onClose()}>
      <SheetContent className="w-full overflow-y-auto sm:max-w-xl">
        <SheetHeader>
          <SheetTitle className="font-display break-all text-2xl font-normal">{version}</SheetTitle>
          <SheetDescription>Everything needed to reproduce and audit this version.</SheetDescription>
        </SheetHeader>
        <div className="space-y-6 px-4 pb-8">
          {error ? (
            <IntelError error={error} retry={() => mutate()} runBacked={false} />
          ) : !data ? (
            <Skeleton className="h-80 w-full" />
          ) : (
            <>
              <section>
                <p className="eyebrow mb-1">Reproduce</p>
                <SpecRows
                  rows={[
                    { label: "State", value: <ModelStateBadge state={data.state} /> },
                    { label: "Config hash", value: data.config_hash ?? "—" },
                    { label: "Config", value: <span title={String(lin.config_path ?? "")}>{String(lin.config_path ?? "—").split("/").pop()}</span> },
                    { label: "Git commit", value: data.git_commit ?? "—" },
                    { label: "Seed", value: data.seed ?? "—" },
                    { label: "jev_ml", value: String(lin.jev_ml_version ?? "—") },
                    { label: "Quick", value: lin.quick === undefined ? "—" : String(lin.quick) },
                    { label: "Experiment run", value: data.experiment_run ?? "—" },
                    { label: "Retrain decision", value: data.decision_id ?? "— (manual)" },
                    { label: "Incumbent at training", value: String(lin.incumbent_at_training ?? "—") },
                  ]}
                />
              </section>
              <section>
                <p className="eyebrow mb-1">Snapshot</p>
                {data.snapshot ? (
                  <>
                    <SpecRows
                      rows={[
                        { label: "Id", value: data.snapshot.snapshot_id },
                        { label: "Content hash", value: <span title={data.snapshot.content_hash}>{data.snapshot.content_hash.slice(0, 16)}</span> },
                        { label: "Cut-off", value: data.snapshot.cutoff ? fmtDate(data.snapshot.cutoff) : "—" },
                        { label: "Semantics", value: data.snapshot.semantics_version },
                        ...Object.entries(data.snapshot.row_counts).map(([k, v]) => ({ label: `rows · ${k}`, value: fmtValue(v) })),
                      ]}
                    />
                    {data.snapshot.manifest && (
                      <details className="mt-2">
                        <summary className="cursor-pointer text-xs text-muted-foreground">Manifest</summary>
                        <pre className="mt-2 max-h-72 overflow-auto rounded border hairline p-2 font-mono text-[11px] leading-snug">{JSON.stringify(data.snapshot.manifest, null, 2)}</pre>
                      </details>
                    )}
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">{data.snapshot_id ? `${data.snapshot_id}: the snapshot row is not in this database.` : "Trained before snapshots existed."}</p>
                )}
              </section>
              <section>
                <p className="eyebrow mb-1">Job</p>
                {data.job ? (
                  <>
                    <p className="flex flex-wrap items-center gap-2 text-sm"><JobStatusBadge status={data.job.status} /> {data.job.kind} · {data.job.trigger} · {data.job.requested_by}</p>
                    <ol className="mt-2 space-y-1 text-xs">
                      {data.job.steps.map((s, i) => <li key={i} className="flex gap-2"><span className="eyebrow w-20 shrink-0">{s.step}</span><span className="num min-w-0 break-all text-ink-2">{stepSummary(s)}</span></li>)}
                    </ol>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">No job recorded in this database; the gate result, if any, was read from the model&apos;s files.</p>
                )}
              </section>
              <section>
                <p className="eyebrow mb-1">Promotion</p>
                <SpecRows
                  rows={[
                    { label: "Promoted", value: data.promoted_at ? `${fmtDate(data.promoted_at)} · ${data.promoted_by ?? "—"}` : "never" },
                    { label: "Forced", value: data.forced ? `yes · ${data.force_reason ?? ""}` : "no" },
                    { label: "Gate", value: data.gate ? `${data.gate.passed ? "passed" : "failed"} vs ${data.gate.incumbent ?? "none"}` : "not gated" },
                  ]}
                />
              </section>
              <section>
                <p className="eyebrow mb-1">Registry history</p>
                {data.history.length ? (
                  <ol className="space-y-1.5 text-xs">
                    {data.history.map((h, i) => (
                      <li key={i} className="border-l-2 border-rule pl-3">
                        <span className="num text-muted-foreground">{fmtDate(h.at)}</span> · <strong className="font-medium">{h.action}</strong>
                        {h.version && h.version !== version ? ` → ${h.version}` : ""}
                        {h.reason && <span className="block text-ink-2">{String(h.reason)}</span>}
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="text-sm text-muted-foreground">No registry actions recorded.</p>
                )}
              </section>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

// ---- jobs and snapshots -----------------------------------------------------------------------

function JobsPanel({ jobs, error, retry }: { jobs: TrainingJobOut[] | undefined; error: unknown; retry: () => void }) {
  const [busy, setBusy] = useState(false);
  const revalidate = useRevalidateGovernance();
  const running = anyJobActive(jobs);

  async function retrain() {
    setBusy(true);
    try {
      const j = await api<TrainingJobOut>("/governance/retrain", { json: { quick: true } });
      toast.success(`Retrain queued · job ${j.job_id.slice(0, 8)}`);
      await revalidate();
    } catch (e) {
      failToast(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel title="Jobs" action={
      <Button size="sm" onClick={retrain} disabled={busy || running} title={running ? "One job at a time" : "Snapshot, train (no tuning), gate against the active model; never activates"}>
        <Hammer aria-hidden /> Retrain (quick)
      </Button>
    }>
      {running && <p className="mb-3 text-xs text-muted-foreground" aria-live="polite">A job is running; this list refreshes every 3 s.</p>}
      {error ? (
        <IntelError error={error} retry={retry} runBacked={false} what="training jobs (GET /governance/jobs)" since="Phase 2" />
      ) : !jobs ? (
        <RowsSkeleton rows={3} />
      ) : !jobs.length ? (
        <p className="text-sm text-muted-foreground">No retrain or evaluate job has run on this database.</p>
      ) : (
        <ul className="divide-y hairline">
          {jobs.map((j) => (
            <li key={j.job_id} className="py-3 first:pt-0 last:pb-0">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <p className="text-sm"><span className="font-medium">{j.kind}</span> <span className="text-muted-foreground">· {j.trigger} · {j.quick ? "quick" : "full"} · {j.requested_by}</span></p>
                <JobStatusBadge status={j.status} />
              </div>
              <p className="num mt-0.5 break-all text-xs text-muted-foreground">
                {j.job_id.slice(0, 8)} · {fmtDate(j.created_at)}{j.duration_ms !== null ? ` · ${(j.duration_ms / 1000).toFixed(1)} s` : ""}
                {j.model_version ? ` · ${short(j.model_version)}` : ""}{j.gate_passed !== null ? ` · gate ${j.gate_passed ? "passed" : "failed"}` : ""}{j.promoted ? " · promoted" : ""}
              </p>
              {j.steps.length > 0 && (
                <ol className="mt-1.5 space-y-0.5 text-xs">
                  {j.steps.map((s, i) => <li key={i} className="flex gap-2"><span className="eyebrow w-20 shrink-0">{s.step}</span><span className="num min-w-0 break-all text-ink-2">{stepSummary(s)}</span></li>)}
                </ol>
              )}
              {j.error && <p className="mt-1 line-clamp-3 break-all font-mono text-xs text-destructive" title={j.error}>{j.error}</p>}
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function SnapshotsPanel() {
  const { data, error, mutate } = useSWR<SnapshotOut[]>("/governance/snapshots?limit=20");
  const [busy, setBusy] = useState(false);
  async function exportNow() {
    setBusy(true);
    try {
      const s = await api<SnapshotOut>("/governance/snapshots", { method: "POST" });
      toast.success(`Snapshot ${s.snapshot_id}`);
      await mutate();
    } catch (e) {
      failToast(e);
    } finally {
      setBusy(false);
    }
  }
  return (
    <Panel title="Training snapshots" action={<Button size="sm" variant="outline" onClick={exportNow} disabled={busy}><Package aria-hidden /> Export now</Button>}>
      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} />
      ) : !data ? (
        <RowsSkeleton rows={2} />
      ) : !data.length ? (
        <p className="text-sm text-muted-foreground">No snapshots yet. A retrain exports one; the same data always gives the same id.</p>
      ) : (
        <ul className="divide-y hairline">
          {data.map((s) => (
            <li key={s.snapshot_id} className="py-2.5 first:pt-0 last:pb-0">
              <p className="break-all font-mono text-sm">{s.snapshot_id}</p>
              <p className="num text-xs text-muted-foreground">
                {fmtDate(s.created_at)} · {s.created_by} · cut-off {s.cutoff ? fmtDate(s.cutoff) : "—"} ·{" "}
                {Object.entries(s.row_counts).map(([k, v]) => `${fmtValue(v)} ${k}`).join(" · ")}
              </p>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

// ---- page -------------------------------------------------------------------------------------

export default function ModelLifecyclePage() {
  const jobs = useSWR<TrainingJobOut[]>("/governance/jobs?limit=10", { refreshInterval: (d) => (anyJobActive(d) ? 3000 : 30000) });
  const running = anyJobActive(jobs.data);
  const models = useSWR<GovernedModelList>("/governance/models", { refreshInterval: running ? 5000 : 0 });
  const [selected, setSelected] = useState<string | null>(null);
  const [lineage, setLineage] = useState<string | null>(null);
  const list = models.data;
  const focus = useMemo(() => {
    if (!list) return null;
    return list.items.find((m) => m.version === selected) ?? list.items.find((m) => !m.is_active && m.gate_passed !== null) ?? list.items[0] ?? null;
  }, [list, selected]);

  return (
    <div>
      <PageHeader
        eyebrow="operations"
        title="Model lifecycle"
        description="Candidates are gated against the model serving now; only a version that passed is promotable. Every promotion, rejection and rollback is audited."
      />
      {models.error ? (
        <IntelError error={models.error} retry={() => models.mutate()} runBacked={false} what="model governance (GET /governance/models)" since="Phase 2" />
      ) : !list ? (
        <div className="space-y-4"><PanelsSkeleton n={2} className="lg:grid-cols-2" /><RowsSkeleton rows={4} /></div>
      ) : (
        <div className="space-y-8">
          <ServingPanel list={list} />

          <section aria-labelledby="versions-h">
            <h2 id="versions-h" className="eyebrow mb-3">Versions</h2>
            {!list.items.length ? (
              <EmptyState title="No model versions" body="Train a model (Retrain below, or scripts/retrain.py run --quick) to register a candidate." />
            ) : (
              <div className="relative overflow-x-auto rounded-lg border bg-card">
                <table className="w-full min-w-[980px] text-sm">
                  <caption className="sr-only">Model versions with state, training metrics, gate result and actions</caption>
                  <thead>
                    <tr className="border-b hairline text-left text-xs text-muted-foreground">
                      <th className="px-4 py-2.5 font-normal">Version</th>
                      <th className="px-2 py-2.5 font-normal">State</th>
                      {HEAD.map((k) => <th key={k} className="px-2 py-2.5 text-right font-normal">{k}</th>)}
                      <th className="px-2 py-2.5 font-normal">Gate</th>
                      <th className="px-2 py-2.5 font-normal">Details</th>
                      <th className="px-4 py-2.5 text-right font-normal">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.items.map((m) => {
                      const c = gateCounts(m.gates);
                      const total = c.pass + c.fail + c.skipped;
                      return (
                        <tr key={m.version} className={cn("border-b hairline align-top last:border-0", focus?.version === m.version && "bg-primary/5")}>
                          <td className="px-4 py-2.5">
                            <span className="font-mono text-xs" title={m.version}>{m.version}</span>
                            <span className="num block text-xs text-muted-foreground">{m.trained_at ? `trained ${fmtDate(m.trained_at)}` : "—"}</span>
                          </td>
                          <td className="px-2 py-2.5"><ModelStateBadge state={m.state} /></td>
                          {HEAD.map((k) => <td key={k} className="num px-2 py-2.5 text-right">{m.metrics[k] !== undefined ? m.metrics[k].toFixed(4) : "—"}</td>)}
                          <td className="px-2 py-2.5">
                            <GateVerdictBadge verdict={gateVerdict(m, list.active)} />
                            {total > 0 && <span className="num mt-1 block text-xs text-muted-foreground">{c.pass} pass · {c.fail} fail · {c.skipped} skipped</span>}
                          </td>
                          <td className="px-2 py-2.5">
                            <div className="flex flex-col items-start gap-1">
                              <button type="button" onClick={() => setSelected(m.version)} className="inline-flex items-center gap-1 text-xs text-primary hover:underline" aria-pressed={focus?.version === m.version}>
                                <FileSearch className="size-3.5" aria-hidden /> Gates
                              </button>
                              <button type="button" onClick={() => setLineage(m.version)} className="inline-flex items-center gap-1 text-xs text-primary hover:underline">
                                <History className="size-3.5" aria-hidden /> Lineage
                              </button>
                            </div>
                          </td>
                          <td className="px-4 py-2.5"><PromoteButtons m={m} busyJob={running} /></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            <p className="mt-2 text-xs text-muted-foreground">Metrics are each version&apos;s own training-time test scores (headline @10); the gate compares recipes on one frozen split instead.</p>
          </section>

          <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
            {focus ? <GatePanel m={focus} active={list.active} /> : <div />}
            <div className="space-y-4">
              <JobsPanel jobs={jobs.data} error={jobs.error} retry={() => jobs.mutate()} />
              <SnapshotsPanel />
            </div>
          </div>
        </div>
      )}
      <LineageDrawer version={lineage} onClose={() => setLineage(null)} />
    </div>
  );
}
