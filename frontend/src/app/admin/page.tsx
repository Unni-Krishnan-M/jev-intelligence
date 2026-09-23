"use client";

import Link from "next/link";
import useSWR from "swr";

import { DailyBars } from "@/components/jev/admin/charts";
import { fmtDate, fmtNum, Panel, SpecRows, StatTile, Status } from "@/components/jev/admin/ui";
import { Container, EmptyState, ErrorState, PageHeader } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { api, ApiError } from "@/lib/api";
import type { ModelVersion } from "@/lib/types";

interface Health {
  status: string;
  version: string;
  database: string;
  cache: { backend: string; status: string };
}
interface MlHealth {
  status: string;
  error?: string;
  model_version?: string;
  n_items?: number;
  als_factors?: number;
  content_features?: number;
  itemknn_nnz?: number;
  load_seconds?: number;
  dataset_version?: string;
  self_check?: { recommendations: number };
}
interface Stats {
  users: number;
  ratings: number;
  watches: number;
  recommendations_served: number;
  feedback: Record<string, number>;
  recommendations_per_day: { date: string; count: number }[];
  reason_codes: { code: string; count: number }[];
}

// Health endpoints answer 503 with a JSON body when degraded; show that body instead of an error.
async function tolerant<T>(path: string): Promise<T> {
  try {
    return await api<T>(path);
  } catch (e) {
    if (e instanceof ApiError && e.status === 503) {
      return { status: "degraded", error: e.message } as T;
    }
    throw e;
  }
}

const FEEDBACK_LABEL: Record<string, string> = {
  like: "Good pick",
  dislike: "Dislike",
  not_interested: "Not for me",
  clicked: "Opened",
};

export default function AdminOverview() {
  const health = useSWR<Health>("admin:health", () => tolerant<Health>("/health"), { refreshInterval: 30000 });
  const ml = useSWR<MlHealth>("admin:health-ml", () => tolerant<MlHealth>("/health/ml"), { refreshInterval: 30000 });
  const stats = useSWR<Stats>("/admin/stats");
  const models = useSWR<ModelVersion[]>("/models");
  const active = models.data?.find((m) => m.is_active);

  return (
    <Container>
      <PageHeader eyebrow="Admin · overview" title="Control room">
        Live health of the API, database, cache and loaded model, plus how recommendations are being served and received.
      </PageHeader>

      <div className="mb-8 flex flex-wrap items-center justify-between gap-3 rounded-lg border bg-card px-4 py-3">
        <p className="text-sm">
          <span className="eyebrow mr-2">Intelligence</span>
          Signals, early warnings, decisions and action plans built on the platform&apos;s data.
        </p>
        <Button asChild variant="outline" size="sm"><Link href="/intel">Open the console</Link></Button>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Panel title="System health">
          {health.error ? (
            <ErrorState error={health.error} retry={() => health.mutate()} />
          ) : !health.data ? (
            <Skeleton className="h-32 w-full" />
          ) : (
            <SpecRows
              rows={[
                { label: "API", value: <Status ok={health.data.status === "ok"} label={`${health.data.status} · v${health.data.version ?? "?"}`} /> },
                { label: "Database", value: <Status ok={health.data.database === "ok"} label={health.data.database ?? "unknown"} /> },
                {
                  label: "Cache",
                  value: <Status ok={health.data.cache?.status === "ok"} label={`${health.data.cache?.backend ?? "?"} · ${health.data.cache?.status ?? "?"}`} />,
                },
              ]}
            />
          )}
        </Panel>

        <Panel title="Recommendation model" className="lg:col-span-2">
          {ml.error ? (
            <ErrorState error={ml.error} retry={() => ml.mutate()} />
          ) : !ml.data ? (
            <Skeleton className="h-32 w-full" />
          ) : ml.data.status !== "ok" ? (
            <div className="space-y-2 text-sm">
              <Status ok={false} label={ml.data.status} />
              <p className="text-muted-foreground">{ml.data.error ?? "Model not loaded."}</p>
            </div>
          ) : (
            <div className="grid gap-x-8 sm:grid-cols-2">
              <SpecRows
                rows={[
                  { label: "Status", value: <Status ok label={`ok · self-check ${ml.data.self_check?.recommendations ?? 0} recs`} /> },
                  { label: "Version", value: ml.data.model_version },
                  { label: "Dataset", value: ml.data.dataset_version },
                  { label: "Load time", value: `${fmtNum(ml.data.load_seconds, 3)} s` },
                ]}
              />
              <SpecRows
                rows={[
                  { label: "Catalogue items", value: ml.data.n_items?.toLocaleString() },
                  { label: "ALS factors", value: ml.data.als_factors },
                  { label: "Content features", value: ml.data.content_features?.toLocaleString() },
                  { label: "Item-kNN non-zeros", value: ml.data.itemknn_nnz?.toLocaleString() },
                ]}
              />
            </div>
          )}
        </Panel>
      </div>

      <div className="mt-8 grid grid-cols-2 gap-4 lg:grid-cols-4">
        {stats.error ? (
          <ErrorState className="col-span-full" error={stats.error} retry={() => stats.mutate()} />
        ) : !stats.data ? (
          Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-24" />)
        ) : (
          <>
            <StatTile label="Users" value={stats.data.users.toLocaleString()} />
            <StatTile label="Ratings" value={stats.data.ratings.toLocaleString()} hint="Stored in JEV (not MovieLens)" />
            <StatTile label="Watches" value={stats.data.watches.toLocaleString()} />
            <StatTile label="Recs served" value={stats.data.recommendations_served.toLocaleString()} hint="Every served item is logged with its reason" />
          </>
        )}
      </div>

      {stats.data && (
        <div className="mt-4 grid gap-4 lg:grid-cols-3">
          <Panel title="Recommendations served per day (14 days)" className="lg:col-span-2">
            {stats.data.recommendations_per_day.length === 0 ? (
              <EmptyState title="Nothing served yet" body="Recommendations appear here once users open their programme." />
            ) : (
              <>
                <DailyBars data={stats.data.recommendations_per_day} />
                <table className="sr-only">
                  <caption>Recommendations per day</caption>
                  <tbody>
                    {stats.data.recommendations_per_day.map((d) => (
                      <tr key={d.date}><td>{d.date}</td><td>{d.count}</td></tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </Panel>
          <div className="space-y-4">
            <Panel title="Feedback">
              {Object.keys(stats.data.feedback).length === 0 ? (
                <p className="text-sm text-muted-foreground">No feedback yet.</p>
              ) : (
                <SpecRows
                  rows={Object.entries(stats.data.feedback)
                    .sort((a, b) => b[1] - a[1])
                    .map(([k, v]) => ({ label: FEEDBACK_LABEL[k] ?? k, value: v.toLocaleString() }))}
                />
              )}
            </Panel>
            <Panel title="Explanation types served">
              {stats.data.reason_codes.length === 0 ? (
                <p className="text-sm text-muted-foreground">No explanations logged yet.</p>
              ) : (
                <SpecRows rows={stats.data.reason_codes.map((r) => ({ label: r.code.replaceAll("_", " "), value: r.count.toLocaleString() }))} />
              )}
            </Panel>
          </div>
        </div>
      )}

      <div className="mt-8">
        <Panel
          title="Active model"
          action={<Button asChild variant="outline" size="sm"><Link href="/admin/models">All versions</Link></Button>}
        >
          {models.error ? (
            <ErrorState error={models.error} retry={() => models.mutate()} />
          ) : !models.data ? (
            <Skeleton className="h-20 w-full" />
          ) : !active ? (
            <EmptyState title="No active model" body="Train a model with scripts/train_models.py, then activate it." />
          ) : (
            <div className="flex flex-wrap items-end justify-between gap-6">
              <div>
                <p className="font-display text-2xl leading-tight">{active.version}</p>
                <p className="mt-1 font-mono text-xs text-muted-foreground">
                  trained {fmtDate(active.trained_at)} · seed {active.training_seed} · {active.dataset_version}
                </p>
              </div>
              <dl className="grid grid-cols-2 gap-x-8 gap-y-2 sm:grid-cols-4">
                {(["ndcg@10", "recall@10", "precision@10", "hit_rate@10"] as const).map((k) => (
                  <div key={k}>
                    <dt className="eyebrow">{k.replace("_", " ")}</dt>
                    <dd className="num mt-0.5 text-lg">{fmtNum(active.metrics[k])}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </Panel>
      </div>
    </Container>
  );
}
