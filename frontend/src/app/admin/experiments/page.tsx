"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";

import { MetricBars, MODEL_COLOR, MODEL_LABEL, MODEL_ORDER, MultiLine, SeriesLine } from "@/components/jev/admin/charts";
import { fmtDate, fmtNum, Panel, SpecRows } from "@/components/jev/admin/ui";
import { Container, EmptyState, ErrorState, PageHeader, SectionHeader } from "@/components/jev/states";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { Experiment, ExperimentDetail, Metric } from "@/lib/types";
import { cn } from "@/lib/utils";

type Protocol = "test" | "cold_start";

const COLUMNS: { metric: string; label: string; digits: number }[] = [
  { metric: "precision", label: "Precision", digits: 4 },
  { metric: "recall", label: "Recall", digits: 4 },
  { metric: "f1", label: "F1", digits: 4 },
  { metric: "ndcg", label: "NDCG", digits: 4 },
  { metric: "map", label: "MAP", digits: 4 },
  { metric: "hit_rate", label: "HitRate", digits: 4 },
  { metric: "coverage", label: "Coverage", digits: 4 },
  { metric: "diversity", label: "Diversity", digits: 4 },
  { metric: "novelty", label: "Novelty", digits: 2 },
];
const PANELS = [
  { metric: "ndcg", title: "NDCG@10" },
  { metric: "recall", title: "Recall@10" },
  { metric: "precision", title: "Precision@10" },
  { metric: "hit_rate", title: "HitRate@10" },
  { metric: "coverage", title: "Coverage@10" },
];

type Lookup = (model: string, metric: string, k: number) => number | undefined;

function makeLookup(metrics: Metric[], protocol: Protocol): Lookup {
  const map = new Map<string, number>();
  for (const m of metrics) if (m.protocol === protocol) map.set(`${m.model_name}|${m.metric}|${m.k}`, m.value);
  return (model, metric, k) => map.get(`${model}|${metric}|${k}`);
}

function ComparisonTable({ lookup, models }: { lookup: Lookup; models: string[] }) {
  const best = Object.fromEntries(
    COLUMNS.map((c) => [c.metric, Math.max(...models.map((m) => lookup(m, c.metric, 10) ?? -Infinity))]),
  );
  return (
    <div className="overflow-x-auto rounded-lg border bg-card">
      <table className="w-full min-w-[760px] text-sm">
        <caption className="sr-only">Model comparison at K=10</caption>
        <thead>
          <tr className="border-b hairline text-left text-xs text-muted-foreground">
            <th className="px-4 py-2.5 font-normal">Model</th>
            {COLUMNS.map((c) => <th key={c.metric} className="px-2 py-2.5 text-right font-normal">{c.label}@10</th>)}
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m} className={cn("border-b hairline last:border-0", m === "hybrid" && "bg-accent/40")}>
              <td className="px-4 py-2.5">
                <span className="flex items-center gap-2">
                  <span className="size-2 rounded-full" style={{ background: MODEL_COLOR[m] }} aria-hidden />
                  {MODEL_LABEL[m] ?? m}
                </span>
              </td>
              {COLUMNS.map((c) => {
                const v = lookup(m, c.metric, 10);
                const isBest = v !== undefined && v === best[c.metric];
                return (
                  <td key={c.metric} className={cn("num px-2 py-2.5 text-right", isBest ? "font-semibold text-foreground" : "text-ink-2")}>
                    {fmtNum(v, c.digits)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExperimentView({ id }: { id: number }) {
  const { data, error, mutate } = useSWR<ExperimentDetail>(`/experiments/${id}`);
  const [protocol, setProtocol] = useState<Protocol>("test");
  const lookup = useMemo(() => (data ? makeLookup(data.metrics, protocol) : () => undefined), [data, protocol]);

  if (error) return <ErrorState error={error} retry={() => mutate()} />;
  if (!data) return <Skeleton className="h-[480px] w-full" />;

  const present = new Set(data.metrics.filter((m) => m.protocol === protocol).map((m) => m.model_name));
  const models = MODEL_ORDER.filter((m) => present.has(m)) as string[];
  const hasCold = data.metrics.some((m) => m.protocol === "cold_start");
  const ks = data.summary.evaluation?.ks ?? [5, 10, 20];
  const lineData = ks.map((k) => ({ k, ...Object.fromEntries(models.map((m) => [m, lookup(m, "ndcg", k) ?? 0])) }));
  const s = data.summary;
  const split = data.split as Record<string, number | string>;
  const loss = (s.als_loss_history ?? []).map((v, i) => ({ iteration: i + 1, loss: v }));
  const nUsers = protocol === "test" ? s.n_eval_users?.test : s.n_eval_users?.cold_start;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div className="min-w-0">
          <p className="eyebrow">Experiment · {fmtDate(data.created_at)}</p>
          <h2 className="font-display mt-1 break-all text-3xl leading-tight">{data.run_id}</h2>
          <p className="mt-1 font-mono text-xs text-muted-foreground">
            model {data.model_version ?? "— (evaluation only)"} · {nUsers ?? "?"} evaluated users
          </p>
        </div>
        {hasCold && (
          <ToggleGroup
            type="single"
            value={protocol}
            onValueChange={(v) => v && setProtocol(v as Protocol)}
            variant="outline"
            aria-label="Evaluation protocol"
          >
            <ToggleGroupItem value="test" className="px-3 text-xs">Test split</ToggleGroupItem>
            <ToggleGroupItem value="cold_start" className="px-3 text-xs">
              Cold start ({s.evaluation?.cold_start_profile_size ?? 3} interactions)
            </ToggleGroupItem>
          </ToggleGroup>
        )}
      </div>

      <section aria-labelledby="cmp-heading">
        <SectionHeader
          id="cmp-heading"
          kicker={protocol === "test" ? "held-out ratings ≥ 4, K = 10" : "profiles truncated to first interactions, K = 10"}
          title="All models, one protocol"
        />
        <ComparisonTable lookup={lookup} models={models} />
        <p className="mt-2 text-xs text-muted-foreground">
          Bold marks the best value per column. Coverage, diversity and novelty are beyond-accuracy metrics, so random
          is expected to win them.
        </p>
      </section>

      <section aria-label="Metric charts">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {PANELS.map((p) => (
            <MetricBars key={p.metric} title={p.title} rows={models.map((m) => ({ model: m, value: lookup(m, p.metric, 10) ?? 0 }))} />
          ))}
          <Panel title="NDCG by cut-off K">
            <MultiLine
              data={lineData}
              xKey="k"
              xFormat={(v) => `K=${v}`}
              series={models.map((m) => ({ key: m, label: MODEL_LABEL[m] ?? m, color: MODEL_COLOR[m] }))}
              height={200}
            />
          </Panel>
        </div>
      </section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Run summary">
          <SpecRows
            rows={[
              { label: "Split strategy", value: String(split.strategy ?? "—") },
              { label: "Train / val / test rows", value: `${split.train_rows ?? "?"} / ${split.val_rows ?? "?"} / ${split.test_rows ?? "?"}` },
              { label: "Evaluated users (test / cold)", value: `${s.n_eval_users?.test ?? "—"} / ${s.n_eval_users?.cold_start ?? "—"}` },
              { label: "Relevance threshold", value: s.evaluation?.relevance_threshold !== undefined ? `rating ≥ ${s.evaluation.relevance_threshold}` : "—" },
              { label: "Selection metric", value: s.selection_metric ?? (s.quick ? "— (quick run, no tuning)" : "—") },
              {
                label: "Tuning trials",
                value: s.tuning_trials && Object.keys(s.tuning_trials).length
                  ? Object.entries(s.tuning_trials).map(([k, v]) => `${k} ${v}`).join(" · ")
                  : "none",
              },
              { label: "Runtime", value: s.seconds_total !== undefined ? `${s.seconds_total.toFixed(0)} s` : "—" },
              { label: "Seed", value: data.training_seed },
              { label: "Dataset", value: data.dataset_version },
            ]}
          />
        </Panel>
        {loss.length > 0 ? (
          <Panel title="ALS training objective (production fit)">
            <SeriesLine data={loss} xKey="iteration" yKey="loss" name="loss / nnz" />
          </Panel>
        ) : (
          <Panel title="ALS training objective"><p className="text-sm text-muted-foreground">No loss history recorded.</p></Panel>
        )}
      </div>
    </div>
  );
}

export default function ExperimentsPage() {
  const { data, error, mutate } = useSWR<Experiment[]>("/experiments");
  const [selected, setSelected] = useState<number | null>(null);
  const current = selected ?? data?.[0]?.id ?? null;

  return (
    <Container>
      <PageHeader eyebrow="Admin · experiments" title="Evaluation runs">
        Every model is scored with one protocol: a per-user temporal split, relevance = held-out rating ≥ 4, and the
        same users, exclusions and K for all. The numbers below are read from the saved run files.
      </PageHeader>

      {error ? (
        <ErrorState error={error} retry={() => mutate()} />
      ) : !data ? (
        <Skeleton className="h-40 w-full" />
      ) : data.length === 0 ? (
        <EmptyState title="No experiments yet" body="Run `uv run python scripts/train_models.py` to produce one." />
      ) : (
        <div className="space-y-10">
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full min-w-[640px] text-sm">
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2.5 font-normal">Run</th>
                  <th className="px-2 py-2.5 font-normal">Created</th>
                  <th className="px-2 py-2.5 font-normal">Model version</th>
                  <th className="px-2 py-2.5 text-right font-normal">Hybrid NDCG@10</th>
                  <th className="px-4 py-2.5 text-right font-normal">Hybrid Recall@10</th>
                </tr>
              </thead>
              <tbody>
                {data.map((e) => (
                  <tr key={e.id} onClick={() => setSelected(e.id)}
                    className={cn("cursor-pointer border-b hairline last:border-0 hover:bg-accent/50", current === e.id && "bg-accent/70")}>
                    <td className="px-4 py-2.5">
                      <button className="num text-left" onClick={() => setSelected(e.id)} aria-pressed={current === e.id}>{e.run_id}</button>
                    </td>
                    <td className="num px-2 py-2.5 text-xs text-muted-foreground">{fmtDate(e.created_at)}</td>
                    <td className="num px-2 py-2.5 text-xs text-ink-2">{e.model_version ?? "—"}</td>
                    <td className="num px-2 py-2.5 text-right">{fmtNum(e.headline["ndcg@10"])}</td>
                    <td className="num px-4 py-2.5 text-right">{fmtNum(e.headline["recall@10"])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {current !== null && <ExperimentView key={current} id={current} />}
        </div>
      )}
    </Container>
  );
}
