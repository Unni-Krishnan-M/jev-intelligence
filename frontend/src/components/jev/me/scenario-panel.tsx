"use client";

import { Play } from "lucide-react";
import { useState } from "react";

import { ChartLegend, TableView } from "@/components/jev/intel/charts";
import { EvidenceDisclosure } from "@/components/jev/intel/evidence-list";
import { NotDeployedState } from "@/components/jev/intel/states";
import { RecommendationRanks } from "@/components/jev/me/recommendation-list";
import { ErrorState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api } from "@/lib/api";
import { fmtPct, humanize, isNotDeployed } from "@/lib/intel";
import type { PreferenceScenarioKind, PreferenceScenarioRequest, PreferenceScenarioResponse, PreferenceScenarioResult, PreferenceScenarioSpec } from "@/lib/intel-types";
import { fmtShare, projectedRows, shareAxisMax, type ProjectedRow } from "@/lib/me-intel";
import { cn } from "@/lib/utils";

const KINDS: { kind: PreferenceScenarioKind; name: string; blurb: string }[] = [
  { kind: "continue", name: "Keeps moving", blurb: "Your recent trend carries on at its current pace." },
  { kind: "accelerate", name: "Moves faster", blurb: "The same direction, stronger by the factor you pick." },
  { kind: "reverse", name: "Swings back", blurb: "Your taste returns towards your longer history." },
];

const FACTORS = ["1.5", "2", "3"];
const K = 10;

/*
 * Projected share per genre as a dot on a shared 0..max track: the 80 % band is a wash of the
 * accent, the mean a dot with a surface ring, today's (baseline) share a neutral tick. One hue plus
 * neutral, values printed at the end of every row, and a table view underneath.
 */
function ProjectedShares({ rows, axisMax }: { rows: ProjectedRow[]; axisMax: number }) {
  const x = (v: number) => `${Math.max(0, Math.min(1, v / axisMax)) * 100}%`;
  if (!rows.length) return <p className="text-sm text-muted-foreground">No projected shares returned.</p>;
  return (
    <div>
      <ul className="space-y-2">
        {rows.map((r) => {
          const p = r.projected;
          return (
            <li key={r.category} className="grid grid-cols-[88px_minmax(0,1fr)_92px] items-center gap-2 text-xs" title={p ? `${r.category}: ${fmtShare(p.mean)} (80 % ${fmtShare(p.lo80)}–${fmtShare(p.hi80)}), baseline ${fmtShare(r.baseline)}` : r.category}>
              <span className="truncate text-foreground">{r.category}</span>
              <span className="relative h-4 border-l border-r hairline" aria-hidden>
                <span className="absolute inset-x-0 top-1/2 h-px bg-rule" />
                {p && <span className="absolute top-1 h-2 rounded-[4px] bg-primary/20" style={{ left: x(p.lo80), width: `calc(${x(p.hi80)} - ${x(p.lo80)})` }} />}
                {r.baseline !== null && <span className="absolute top-0 h-4 w-[2px] -translate-x-1/2 bg-muted-foreground" style={{ left: x(r.baseline) }} />}
                {p && <span className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary ring-2 ring-card" style={{ left: x(p.mean) }} />}
              </span>
              <span className="num text-right text-ink-2">
                {p ? fmtShare(p.mean) : "—"}
                {p && <span className="block text-[10px] text-muted-foreground">{Math.round(p.lo80 * 100)}–{Math.round(p.hi80 * 100)} %</span>}
              </span>
            </li>
          );
        })}
      </ul>
      <div className="mt-1 grid grid-cols-[88px_minmax(0,1fr)_92px] gap-2 font-mono text-[10px] text-muted-foreground">
        <span />
        <span className="flex justify-between"><span>0 %</span><span>{fmtShare(axisMax)}</span></span>
        <span />
      </div>
    </div>
  );
}

function ScenarioCard({ s, baselineShares, baselineIds }: { s: PreferenceScenarioResult; baselineShares: Record<string, number>; baselineIds: Set<string> }) {
  const rows = projectedRows(s, baselineShares);
  const axisMax = shareAxisMax(rows);
  return (
    <article className="flex min-w-0 flex-col rounded-lg border bg-card p-4 sm:p-5">
      <p className="eyebrow">{humanize(String(s.kind))}</p>
      <h3 className="font-display mt-1 text-2xl leading-tight">{s.name}</h3>
      <p className="mt-1 text-sm">
        <span className="num">{fmtPct(s.overlap_with_baseline)}</span> <span className="text-muted-foreground">of today&apos;s list stays in this one</span>
      </p>
      <div className="mt-4">
        <p className="eyebrow mb-2">Projected genre shares · 80 % band</p>
        <ProjectedShares rows={rows} axisMax={axisMax} />
        <TableView
          caption={`Projected genre shares for ${s.name}`}
          head={["genre", "baseline", "mean", "lo 80", "hi 80"]}
          rows={rows.map((r) => [r.category, fmtShare(r.baseline), fmtShare(r.projected?.mean), fmtShare(r.projected?.lo80), fmtShare(r.projected?.hi80)])}
        />
      </div>
      <div className="mt-4">
        <p className="eyebrow mb-1">Recommendations under this scenario</p>
        <RecommendationRanks items={s.recommendations} baseline={baselineIds} />
      </div>
      {s.assumptions.length > 0 && (
        <ul className="mt-4 space-y-1 border-t hairline pt-3 text-xs text-muted-foreground">
          {s.assumptions.map((a) => <li key={a}>{a}</li>)}
        </ul>
      )}
    </article>
  );
}

/** "What if my taste keeps changing?" — POST /me/intelligence/scenarios. */
export function ScenarioPanel() {
  const [kinds, setKinds] = useState<string[]>(KINDS.map((k) => k.kind));
  const [factor, setFactor] = useState("2");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PreferenceScenarioResponse | null>(null);
  const [error, setError] = useState<unknown>(null);

  async function run() {
    const scenarios: PreferenceScenarioSpec[] = KINDS.filter((k) => kinds.includes(k.kind)).map((k) => ({
      name: k.name,
      kind: k.kind,
      factor: k.kind === "accelerate" ? Number(factor) : 1,
    }));
    if (!scenarios.length) return;
    const body: PreferenceScenarioRequest = { k: K, scenarios };
    setBusy(true);
    setError(null);
    try {
      setResult(await api<PreferenceScenarioResponse>("/me/intelligence/scenarios", { json: body }));
    } catch (e) {
      setError(e);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const baselineIds = new Set((result?.baseline.recommendations ?? []).map((r) => String(r.item_id)));

  return (
    <div>
      <div className="grid gap-4 rounded-lg border bg-card/60 p-4 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
        <div>
          <p className="eyebrow mb-2">Scenarios</p>
          <ToggleGroup type="multiple" value={kinds} onValueChange={setKinds} variant="outline" size="sm" aria-label="Scenarios to project" className="flex-wrap">
            {KINDS.map((k) => <ToggleGroupItem key={k.kind} value={k.kind}>{k.name}</ToggleGroupItem>)}
          </ToggleGroup>
          <ul className="mt-2 space-y-0.5 text-xs text-muted-foreground">
            {KINDS.filter((k) => kinds.includes(k.kind)).map((k) => <li key={k.kind}><span className="text-ink-2">{k.name}:</span> {k.blurb}</li>)}
          </ul>
        </div>
        <div className="flex flex-wrap items-end gap-4">
          <div className={cn(!kinds.includes("accelerate") && "opacity-50")}>
            <p className="eyebrow mb-2">Faster by</p>
            <ToggleGroup type="single" value={factor} onValueChange={(v) => v && setFactor(v)} variant="outline" size="sm" aria-label="Acceleration factor" disabled={!kinds.includes("accelerate")}>
              {FACTORS.map((f) => <ToggleGroupItem key={f} value={f} className="num">×{f}</ToggleGroupItem>)}
            </ToggleGroup>
          </div>
          <Button onClick={run} disabled={busy || !kinds.length}>
            <Play aria-hidden /> {busy ? "Projecting…" : "Project my taste"}
          </Button>
        </div>
      </div>

      <div className="mt-6" aria-live="polite">
        {error ? (
          isNotDeployed(error) ? (
            <NotDeployedState what="preference scenarios (POST /me/intelligence/scenarios)" since="v1.2" />
          ) : (
            <ErrorState error={error} retry={run} />
          )
        ) : busy ? (
          <div className="grid gap-4 lg:grid-cols-3"><Skeleton className="h-80" /><Skeleton className="h-80" /><Skeleton className="h-80" /></div>
        ) : !result ? (
          <p className="text-sm text-muted-foreground">
            Pick the futures to compare and press “Project my taste”. JEV projects your genre trend, re-ranks the catalogue under each projection
            with the real hybrid model, and resamples your history to show how uncertain that is.
          </p>
        ) : (
          <div className="space-y-6">
            <ChartLegend
              items={[
                { label: "projected mean", color: "var(--primary)", kind: "dot" },
                { label: "80 % band", color: "var(--primary)", kind: "band" },
                { label: "today's share", color: "var(--muted-foreground)", kind: "rule" },
              ]}
            />
            {result.scenarios.length === 0 ? (
              <p className="text-sm text-muted-foreground">The API returned no scenarios.</p>
            ) : (
              <div className="grid gap-4 lg:grid-cols-3">
                {result.scenarios.map((s) => <ScenarioCard key={s.name} s={s} baselineShares={result.baseline.shares} baselineIds={baselineIds} />)}
              </div>
            )}
            <div className="grid gap-6 lg:grid-cols-2">
              <div>
                <p className="eyebrow mb-1">Baseline · today&apos;s list</p>
                <RecommendationRanks items={result.baseline.recommendations} />
              </div>
              <div className="space-y-4">
                <div className="border-l-2 border-primary/60 pl-3">
                  <p className="eyebrow mb-1">Uncertainty</p>
                  <p className="text-sm text-ink-2">{result.uncertainty_note}</p>
                </div>
                {result.assumptions.length > 0 && (
                  <div>
                    <p className="eyebrow mb-1">Assumptions</p>
                    <ul className="space-y-1 text-sm text-ink-2">{result.assumptions.map((a) => <li key={a} className="border-l-2 border-rule pl-3">{a}</li>)}</ul>
                  </div>
                )}
                <EvidenceDisclosure items={result.evidence} linkRefs={false} />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
