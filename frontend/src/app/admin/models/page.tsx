"use client";

import { useState } from "react";
import { toast } from "sonner";
import useSWR, { useSWRConfig } from "swr";

import { SeriesLine } from "@/components/jev/admin/charts";
import { fmtDate, fmtNum, Panel, SpecRows } from "@/components/jev/admin/ui";
import { WeightBar } from "@/components/jev/signal-bar";
import { Container, EmptyState, ErrorState, PageHeader } from "@/components/jev/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { api, errorMessage } from "@/lib/api";
import type { ModelVersion, ModelVersionDetail, SignalName } from "@/lib/types";
import { cn } from "@/lib/utils";

const HEAD = ["ndcg@10", "recall@10", "precision@10", "hit_rate@10"] as const;
const SIGNAL_KEYS: SignalName[] = ["content", "collaborative", "latent", "popularity", "preference", "recency"];

function fmtValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function ModelDetail({ id, onActivated }: { id: number; onActivated: () => void }) {
  const { data, error, mutate } = useSWR<ModelVersionDetail>(`/models/${id}`);
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);

  if (error) return <ErrorState error={error} retry={() => mutate()} />;
  if (!data) return <Skeleton className="h-96 w-full" />;

  const m = data.manifest;
  const hybrid = (m.hybrid_config ?? {}) as Record<string, unknown>;
  const weights = (hybrid.weights ?? {}) as Record<string, number>;
  const total = SIGNAL_KEYS.reduce((s, k) => s + (weights[k] ?? 0), 0) || 1;
  const normWeights = Object.fromEntries(SIGNAL_KEYS.map((k) => [k, (weights[k] ?? 0) / total])) as Record<SignalName, number>;
  const loss = (m.als_loss_history ?? []).map((v, i) => ({ iteration: i + 1, loss: v }));

  async function activate() {
    setBusy(true);
    try {
      await api(`/models/${id}/activate`, { method: "POST" });
      toast.success(`Activated ${data?.version}`);
      setConfirm(false);
      onActivated();
      await mutate();
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      <Panel>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="eyebrow">Model version</p>
            <h2 className="font-display mt-1 break-all text-3xl leading-tight">{data.version}</h2>
            <p className="mt-1 font-mono text-xs text-muted-foreground">
              trained {fmtDate(data.trained_at)} · seed {data.training_seed} · {(data.artifact_bytes / 1e6).toFixed(1)} MB
            </p>
          </div>
          {data.is_active ? (
            <Badge variant="outline" className="border-primary/50 text-primary">Serving</Badge>
          ) : (
            <Button onClick={() => setConfirm(true)}>Activate</Button>
          )}
        </div>
        <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
          {HEAD.map((k) => (
            <div key={k}>
              <dt className="eyebrow">{k.replace("_", " ")}</dt>
              <dd className="num mt-0.5 text-lg">{fmtNum(data.metrics[k])}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-3 text-xs text-muted-foreground">Hybrid model on the held-out test split (fit on train+val).</p>
      </Panel>

      <Panel title="Hybrid blend (base weights)">
        <WeightBar weights={normWeights} />
        <SpecRows
          rows={[
            "behavioral_ramp", "cold_start_boost", "content_support_damping", "diversity_lambda", "normalization",
            "candidates_per_source", "quality_floor",
          ]
            .filter((k) => k in hybrid)
            .map((k) => ({ label: k.replaceAll("_", " "), value: fmtValue(hybrid[k]) }))}
        />
      </Panel>

      {loss.length > 0 && (
        <Panel title="ALS training objective">
          <SeriesLine data={loss} xKey="iteration" yKey="loss" name="loss / nnz" />
          <p className="mt-2 text-xs text-muted-foreground">
            Final weighted loss <span className="num">{loss[loss.length - 1].loss.toFixed(5)}</span> after {loss.length} iterations.
          </p>
        </Panel>
      )}

      {m.components && (
        <Panel title="Component parameters">
          <div className="grid gap-x-8 md:grid-cols-2">
            {Object.entries(m.components).map(([name, params]) => (
              <div key={name} className="mb-3">
                <p className="mb-1 text-sm font-medium">{name}</p>
                <SpecRows
                  rows={Object.entries(params)
                    .filter(([, v]) => typeof v !== "object" || v === null)
                    .map(([k, v]) => ({ label: k.replaceAll("_", " "), value: fmtValue(v) }))}
                />
              </div>
            ))}
          </div>
        </Panel>
      )}

      <Panel title="Provenance">
        <SpecRows
          rows={[
            { label: "Dataset version", value: data.dataset_version },
            { label: "Trained on rows", value: m.trained_on_rows?.toLocaleString() ?? "—" },
            { label: "Catalogue items", value: m.n_items?.toLocaleString() ?? "—" },
            { label: "Git commit", value: m.git_commit ?? "—" },
            { label: "Python", value: m.python ?? "—" },
            { label: "Artifact dir", value: data.artifact_path },
          ]}
        />
        {m.artifact_files && (
          <details className="mt-3">
            <summary className="cursor-pointer text-sm text-muted-foreground">{m.artifact_files.length} artifact files</summary>
            <ul className="mt-2 grid gap-1 font-mono text-xs text-ink-2 sm:grid-cols-2">
              {m.artifact_files.map((f) => <li key={f} className="truncate">{f}</li>)}
            </ul>
          </details>
        )}
        {m.error && <p className="mt-2 text-sm text-destructive">{m.error}</p>}
      </Panel>

      <Dialog open={confirm} onOpenChange={setConfirm}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="font-display text-2xl font-normal">Activate this model?</DialogTitle>
            <DialogDescription>
              <span className="num break-all">{data.version}</span> is loaded and validated first. It then replaces the
              serving model for every user, and cached recommendations are cleared.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirm(false)} disabled={busy}>Cancel</Button>
            <Button onClick={activate} disabled={busy}>{busy ? "Activating…" : "Activate"}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

export default function ModelsPage() {
  const { data, error, mutate } = useSWR<ModelVersion[]>("/models");
  const { mutate: globalMutate } = useSWRConfig();
  const [selected, setSelected] = useState<number | null>(null);
  const current = selected ?? data?.find((m) => m.is_active)?.id ?? data?.[0]?.id ?? null;

  return (
    <Container>
      <PageHeader eyebrow="Admin · model registry" title="Model versions">
        Every training run registers a version with its config, dataset version, seed, metrics and artifacts. Pick
        one to inspect it, then activate it to serve.
      </PageHeader>

      {error ? (
        <ErrorState error={error} retry={() => mutate()} />
      ) : !data ? (
        <Skeleton className="h-48 w-full" />
      ) : data.length === 0 ? (
        <EmptyState title="No models registered" body="Run `uv run python scripts/train_models.py` to train and register one." />
      ) : (
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full min-w-[640px] text-sm">
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2.5 font-normal">Version</th>
                  <th className="px-2 py-2.5 font-normal">Trained</th>
                  {HEAD.map((k) => <th key={k} className="px-2 py-2.5 text-right font-normal">{k.replace("_", " ")}</th>)}
                  <th className="px-4 py-2.5 text-right font-normal">Size</th>
                </tr>
              </thead>
              <tbody>
                {data.map((m) => (
                  <tr
                    key={m.id}
                    onClick={() => setSelected(m.id)}
                    className={cn("cursor-pointer border-b hairline last:border-0 hover:bg-accent/50", current === m.id && "bg-accent/70")}
                  >
                    <td className="px-4 py-3">
                      <button className="text-left" onClick={() => setSelected(m.id)} aria-pressed={current === m.id}>
                        <span className="num block max-w-[15rem] truncate">{m.version}</span>
                        <span className="text-xs text-muted-foreground">seed {m.training_seed} · {m.dataset_version.slice(0, 24)}</span>
                      </button>
                      {m.is_active && <Badge variant="outline" className="mt-1 border-primary/50 text-primary">Active</Badge>}
                    </td>
                    <td className="num px-2 py-3 text-xs text-muted-foreground">{fmtDate(m.trained_at)}</td>
                    {HEAD.map((k) => <td key={k} className="num px-2 py-3 text-right">{fmtNum(m.metrics[k])}</td>)}
                    <td className="num px-4 py-3 text-right text-muted-foreground">{(m.artifact_bytes / 1e6).toFixed(1)} MB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {current !== null && (
            <ModelDetail
              key={current}
              id={current}
              onActivated={() => {
                void mutate();
                void globalMutate((k) => typeof k === "string" && (k.startsWith("/recommendations") || k.startsWith("admin:")));
              }}
            />
          )}
        </div>
      )}
    </Container>
  );
}
