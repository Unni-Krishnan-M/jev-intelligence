"use client";

import { Play, Plus, Save, X } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

import { fmtDate, Panel } from "@/components/jev/admin/ui";
import { SLOT_COLORS, TableView, TimeSeriesChart } from "@/components/jev/intel/charts";
import { CapabilityNotice, useIntelDomain } from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { FilterRow, IntelError, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState, SectionHeader } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { api, errorMessage, qs } from "@/lib/api";
import { fmtSeriesValue, fmtSigned, isNoRun, seriesLabel } from "@/lib/intel";
import type {
  Page,
  PredictionsResponse,
  SavedScenario,
  ScenarioInput,
  ScenarioKind,
  ScenarioOutput,
  ScenarioRequest,
  ScenarioResponse,
  ScenarioSpec,
  Trend,
} from "@/lib/intel-types";

const MAX_SCENARIOS = 6;
const KIND_LABEL: Record<ScenarioKind, string> = {
  continue: "Trend continues",
  slow: "Trend slows (× m)",
  reverse: "Trend reverses (× m)",
  shock: "Level shock (± %)",
};
// starting presets for the builder (inputs, not results)
const PRESETS: ScenarioSpec[] = [
  { name: "Trend continues", kind: "continue" },
  { name: "Trend slows", kind: "slow", trend_multiplier: 0.5 },
  { name: "Trend reverses", kind: "reverse", trend_multiplier: -1 },
  { name: "Shock −20 %", kind: "shock", level_shift_pct: -20, shock_month: 1 },
];

function withDefaults(kind: ScenarioKind, prev: ScenarioSpec): ScenarioSpec {
  const base = { name: prev.name, kind };
  if (kind === "slow") return { ...base, trend_multiplier: prev.kind === "slow" ? prev.trend_multiplier : 0.5 };
  if (kind === "reverse") return { ...base, trend_multiplier: prev.kind === "reverse" ? prev.trend_multiplier : -1 };
  if (kind === "shock") return { ...base, level_shift_pct: prev.level_shift_pct ?? -20, shock_month: prev.shock_month ?? 1 };
  return base;
}

/** Only the parameters that belong to the scenario kind are sent. */
function clean(s: ScenarioSpec): ScenarioSpec {
  const out: ScenarioSpec = { name: s.name.trim(), kind: s.kind };
  if (s.kind === "slow" || s.kind === "reverse") out.trend_multiplier = Number(s.trend_multiplier ?? 0);
  if (s.kind === "shock") {
    out.level_shift_pct = Number(s.level_shift_pct ?? 0);
    out.shock_month = Math.max(1, Math.round(Number(s.shock_month ?? 1)));
  }
  return out;
}

function NumField({ label, value, onChange, step = 0.1, min, max }: { label: string; value: number | undefined; onChange: (v: number) => void; step?: number; min?: number; max?: number }) {
  return (
    <label className="flex min-w-0 flex-col gap-1 text-xs text-muted-foreground">
      {label}
      <Input type="number" inputMode="decimal" step={step} min={min} max={max} value={value ?? ""} onChange={(e) => onChange(e.target.valueAsNumber)} className="num h-8 w-full" />
    </label>
  );
}

function ScenarioRow({ s, index, onChange, onRemove, canRemove }: { s: ScenarioSpec; index: number; onChange: (s: ScenarioSpec) => void; onRemove: () => void; canRemove: boolean }) {
  return (
    <li className="grid grid-cols-[12px_minmax(0,1fr)_auto] items-start gap-x-3 gap-y-2 border-b hairline py-3 last:border-0">
      <span className="mt-2.5 size-2.5 rounded-full" style={{ background: SLOT_COLORS[index] }} aria-hidden />
      <div className="grid grid-cols-1 min-w-0 gap-2 sm:grid-cols-[minmax(0,1fr)_190px] lg:grid-cols-[minmax(0,1fr)_190px_minmax(0,1.2fr)]">
        <label className="flex min-w-0 flex-col gap-1 text-xs text-muted-foreground">
          Name
          <Input value={s.name} maxLength={60} onChange={(e) => onChange({ ...s, name: e.target.value })} className="h-8" />
        </label>
        <label className="flex min-w-0 flex-col gap-1 text-xs text-muted-foreground">
          Kind
          <Select value={s.kind} onValueChange={(v) => onChange(withDefaults(v as ScenarioKind, s))}>
            <SelectTrigger className="h-8 w-full" aria-label={`Scenario ${index + 1} kind`}><SelectValue /></SelectTrigger>
            <SelectContent>
              {(Object.keys(KIND_LABEL) as ScenarioKind[]).map((k) => <SelectItem key={k} value={k}>{KIND_LABEL[k]}</SelectItem>)}
            </SelectContent>
          </Select>
        </label>
        <div className="grid grid-cols-2 gap-2 sm:col-span-2 lg:col-span-1">
          {(s.kind === "slow" || s.kind === "reverse") && (
            <NumField label="Trend multiplier m" value={s.trend_multiplier} onChange={(v) => onChange({ ...s, trend_multiplier: v })} min={-3} max={3} />
          )}
          {s.kind === "shock" && (
            <>
              <NumField label="Level shift %" value={s.level_shift_pct} onChange={(v) => onChange({ ...s, level_shift_pct: v })} step={5} min={-100} max={300} />
              <NumField label="From month" value={s.shock_month} onChange={(v) => onChange({ ...s, shock_month: v })} step={1} min={1} max={36} />
            </>
          )}
          {s.kind === "continue" && <p className="col-span-2 self-end pb-1.5 text-xs text-muted-foreground">Keeps the fitted trend as it is.</p>}
        </div>
      </div>
      <Button variant="ghost" size="icon-sm" onClick={onRemove} disabled={!canRemove} aria-label={`Remove scenario ${index + 1}`} className="mt-5">
        <X aria-hidden />
      </Button>
    </li>
  );
}

function ResultView({ out, names }: { out: ScenarioOutput; names: string[] }) {
  const [fit, setFit] = useState<"paths" | "interval">("paths");
  const metric = out.series_id.split(":")[0];
  const color = (name: string) => SLOT_COLORS[Math.max(0, names.indexOf(name)) % SLOT_COLORS.length];
  const byName = new Map(out.scenarios.map((s) => [s.name, s]));
  const comparison = [...out.comparison].sort((a, b) => a.rank - b.rank);
  return (
    <div className="space-y-4">
      <Panel
        title={`${seriesLabel(out.series_id)} · ${out.model} · as of ${fmtDate(out.as_of).slice(0, 10)}`}
        action={
          <FilterRow
            label="Scale"
            value={fit}
            onChange={(v) => setFit(v as "paths" | "interval")}
            options={[{ value: "paths", label: "Fit paths" }, { value: "interval", label: "Full interval" }]}
          />
        }
      >
        <TimeSeriesChart
          history={out.history.slice(-36)}
          band={out.baseline.points}
          bandLabel="Baseline 80 % interval"
          bandColor="var(--muted-foreground)"
          historyColor="var(--foreground)"
          means={[
            { key: "baseline", label: "Baseline forecast", color: "var(--muted-foreground)", dashed: true, points: out.baseline.points },
            ...out.scenarios.map((s, i) => ({ key: `s${i}`, label: s.name, color: color(s.name), points: s.points })),
          ]}
          metric={metric}
          height={320}
          fitToLines={fit === "paths"}
        />
        <TableView
          caption="Scenario means by month"
          head={["month", "baseline", "lo 80", "hi 80", ...out.scenarios.map((s) => s.name)]}
          rows={out.baseline.points.map((p, j) => [
            p.t.slice(0, 7),
            fmtSeriesValue(p.mean, metric),
            fmtSeriesValue(p.lo80, metric),
            fmtSeriesValue(p.hi80, metric),
            ...out.scenarios.map((s) => fmtSeriesValue(s.points[j]?.mean, metric)),
          ])}
        />
        <p className="mt-3 text-xs text-muted-foreground">{out.uncertainty_note}</p>
      </Panel>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <Panel title="Comparison over the horizon">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[460px] text-sm">
              <caption className="sr-only">Scenario totals against the baseline</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-2 font-normal">#</th>
                  <th className="px-2 py-2 font-normal">Scenario</th>
                  <th className="px-2 py-2 text-right font-normal">Total</th>
                  <th className="px-2 py-2 text-right font-normal">vs baseline</th>
                  <th className="py-2 pl-2 text-right font-normal">End level</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b hairline text-ink-2">
                  <td className="py-2 pr-2" />
                  <td className="px-2 py-2">
                    <span className="flex items-center gap-2"><span className="w-3 border-t-2 border-dashed border-muted-foreground" aria-hidden />Baseline</span>
                  </td>
                  <td className="num px-2 py-2 text-right">{fmtSeriesValue(out.baseline.total, metric)}</td>
                  <td className="num px-2 py-2 text-right">—</td>
                  <td className="num py-2 pl-2 text-right">{fmtSeriesValue(out.baseline.points[out.baseline.points.length - 1]?.mean, metric)}</td>
                </tr>
                {comparison.map((c) => (
                  <tr key={c.name} className="border-b hairline last:border-0">
                    <td className="num py-2 pr-2 text-xs text-muted-foreground">{c.rank}</td>
                    <td className="px-2 py-2">
                      <span className="flex items-center gap-2"><span className="size-2 rounded-full" style={{ background: color(c.name) }} aria-hidden />{c.name}</span>
                    </td>
                    <td className="num px-2 py-2 text-right">{fmtSeriesValue(c.total, metric)}</td>
                    <td className="num px-2 py-2 text-right">
                      {fmtSigned(c.delta_pct, 1, " %")}
                      {byName.get(c.name) && <span className="block text-xs text-muted-foreground">{fmtSigned(byName.get(c.name)?.delta_vs_baseline, metric === "share" || metric === "rating" ? 3 : 0)}</span>}
                    </td>
                    <td className="num py-2 pl-2 text-right">{fmtSeriesValue(c.end_level, metric)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {metric !== "volume" && metric !== "active_users" && (
            <p className="mt-2 text-xs text-muted-foreground">For shares and ratings a “total” sums monthly values; compare scenarios with each other rather than reading it as a count.</p>
          )}
        </Panel>
        <Panel title="Assumptions">
          <ul className="space-y-3 text-sm">
            {out.scenarios.map((s) => (
              <li key={s.name}>
                <p className="flex items-center gap-2"><span className="size-2 rounded-full" style={{ background: color(s.name) }} aria-hidden />{s.name}</p>
                <ul className="mt-1 list-disc space-y-0.5 pl-8 text-xs text-ink-2">
                  {s.assumptions.map((a) => <li key={a}>{a}</li>)}
                </ul>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  );
}

function Builder() {
  const params = useSearchParams();
  const preset = params.get("series");
  const { q: dq } = useIntelDomain();
  const trends = useSWR<Page<Trend>>(dq(`/intel/trends${qs({ limit: 100 })}`));
  const preds = useSWR<PredictionsResponse>(dq("/intel/predictions"));
  const saved = useSWR<{ items: SavedScenario[]; total: number }>(dq("/intel/scenarios"));

  const options = useMemo(() => {
    const ids = new Set<string>();
    for (const f of preds.data?.forecasts ?? []) ids.add(f.series_id);
    for (const t of trends.data?.items ?? []) ids.add(t.series_id);
    return [...ids].sort((a, b) => (a === "volume:all" ? -1 : b === "volume:all" ? 1 : a.localeCompare(b)));
  }, [preds.data, trends.data]);

  const [picked, setPicked] = useState<string | null>(preset);
  const seriesId = picked ?? options[0] ?? "";
  const [horizon, setHorizon] = useState(12);
  const [specs, setSpecs] = useState<ScenarioSpec[]>(PRESETS);
  const [result, setResult] = useState<ScenarioOutput | null>(null);
  const [resultNames, setResultNames] = useState<string[]>([]);
  const [busy, setBusy] = useState<"run" | "save" | null>(null);
  const [title, setTitle] = useState("");
  const [savedId, setSavedId] = useState<number | null>(null);

  const names = specs.map((s) => s.name.trim());
  const problem = !seriesId
    ? "Pick a series."
    : !Number.isFinite(horizon) || horizon < 1 || horizon > 36
      ? "Horizon must be 1–36 months."
      : names.some((n) => !n)
        ? "Every scenario needs a name."
        : new Set(names).size !== names.length
          ? "Scenario names must be unique."
          : null;

  const input = (): ScenarioInput => ({ series_id: seriesId, horizon_months: Math.round(horizon), as_of: null, scenarios: specs.map(clean) });

  async function submit(save: boolean) {
    if (problem) return;
    setBusy(save ? "save" : "run");
    try {
      const body: ScenarioRequest = { ...input(), save, ...(save ? { title: title.trim() || `${seriesLabel(seriesId)} · ${Math.round(horizon)} m` } : {}) };
      const out = await api<ScenarioResponse>(dq("/intel/scenarios"), { json: body });
      setResult(out);
      setResultNames(specs.map((s) => s.name.trim()));
      if (save) {
        setSavedId(out.id);
        toast.success("Scenario saved");
        await saved.mutate();
      } else {
        setSavedId(null);
      }
    } catch (e) {
      toast.error(errorMessage(e));
    } finally {
      setBusy(null);
    }
  }

  function open(s: SavedScenario) {
    setPicked(s.input.series_id);
    setHorizon(s.input.horizon_months);
    setSpecs(s.input.scenarios);
    setResult(s.output);
    setResultNames(s.input.scenarios.map((x) => x.name));
    setSavedId(s.id);
    setTitle(s.title);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  const loadingOptions = !trends.data && !preds.data && !trends.error && !preds.error;
  const noRun = isNoRun(trends.error) && isNoRun(preds.error);

  return (
    <div className="space-y-10">
      {noRun ? (
        <IntelError error={trends.error} />
      ) : (
        <Panel title="Build">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,1fr)_140px]">
            <div className="min-w-0 space-y-1">
              <Label className="text-xs font-normal text-muted-foreground" htmlFor="series">Series</Label>
              {loadingOptions ? (
                <Skeleton className="h-8 w-full" />
              ) : (
                <Select value={seriesId || undefined} onValueChange={setPicked}>
                  <SelectTrigger id="series" className="h-8 w-full"><SelectValue placeholder="No series in the latest run" /></SelectTrigger>
                  <SelectContent>
                    {options.map((id) => <SelectItem key={id} value={id}>{seriesLabel(id)}</SelectItem>)}
                    {preset && !options.includes(preset) && <SelectItem value={preset}>{seriesLabel(preset)}</SelectItem>}
                  </SelectContent>
                </Select>
              )}
            </div>
            <NumField label="Horizon (months)" value={horizon} onChange={setHorizon} step={1} min={1} max={36} />
          </div>

          <ul className="mt-4 border-t hairline">
            {specs.map((s, i) => (
              <ScenarioRow
                key={i}
                s={s}
                index={i}
                canRemove={specs.length > 1}
                onChange={(n) => setSpecs(specs.map((x, j) => (j === i ? n : x)))}
                onRemove={() => setSpecs(specs.filter((_, j) => j !== i))}
              />
            ))}
          </ul>

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <Button
              variant="outline"
              size="sm"
              disabled={specs.length >= MAX_SCENARIOS}
              onClick={() => setSpecs([...specs, { name: `Scenario ${specs.length + 1}`, kind: "continue" }])}
            >
              <Plus aria-hidden /> Add scenario <span className="num text-muted-foreground">{specs.length}/{MAX_SCENARIOS}</span>
            </Button>
            <div className="flex items-center gap-3">
              {problem && <p className="text-xs text-destructive" role="status">{problem}</p>}
              <Button onClick={() => submit(false)} disabled={Boolean(problem) || busy !== null}>
                <Play aria-hidden /> {busy === "run" ? "Running…" : "Run scenarios"}
              </Button>
            </div>
          </div>
        </Panel>
      )}

      {result && (
        <section aria-labelledby="result-heading">
          <SectionHeader
            id="result-heading"
            kicker="result"
            title="Scenarios against the baseline"
            action={
              <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
                <Input value={title} onChange={(e) => setTitle(e.target.value)} maxLength={120} placeholder="Title for this analysis" aria-label="Scenario title" className="h-8 min-w-0 flex-1 sm:w-64" />
                <Button variant="outline" size="sm" onClick={() => submit(true)} disabled={Boolean(problem) || busy !== null}>
                  <Save aria-hidden /> {busy === "save" ? "Saving…" : savedId ? "Save again" : "Save"}
                </Button>
              </div>
            }
          />
          <ResultView out={result} names={resultNames} />
        </section>
      )}

      <section aria-labelledby="saved-heading">
        <SectionHeader id="saved-heading" kicker="library" title="Saved analyses" />
        {saved.error ? (
          <IntelError error={saved.error} retry={() => saved.mutate()} runBacked={false} />
        ) : !saved.data ? (
          <RowsSkeleton rows={3} />
        ) : saved.data.items.length === 0 ? (
          <EmptyState title="Nothing saved yet" body="Run a set of scenarios, give it a title and save it to keep it here." />
        ) : (
          <ul className="divide-y hairline rounded-lg border bg-card">
            {saved.data.items.map((s) => (
              <li key={s.id} className="flex flex-wrap items-center justify-between gap-3 px-4 py-3">
                <div className="min-w-0">
                  <p className="truncate">{s.title}</p>
                  <p className="font-mono text-xs text-muted-foreground">
                    {seriesLabel(s.series_id)} · {s.input.horizon_months} m · {s.input.scenarios.length} scenarios · {fmtDate(s.created_at)}
                  </p>
                </div>
                <Button variant="outline" size="sm" onClick={() => open(s)} aria-pressed={savedId === s.id}>Open</Button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

export default function ScenariosPage() {
  const { can } = useIntelDomain();
  const cap = can("scenarios");
  return (
    <div>
      <PageHeader
        eyebrow="anticipate"
        title="What-if scenarios"
        description="Bend a series' fitted trend — continue, slow, reverse, or shock its level — and compare the paths with the baseline forecast and its 80 % band. Scenarios move the trend, not the noise."
      />
      {cap.available ? (
        <Suspense fallback={<Skeleton className="h-72 w-full" />}>
          <Builder />
        </Suspense>
      ) : (
        <CapabilityNotice cap="scenarios" />
      )}
    </div>
  );
}
