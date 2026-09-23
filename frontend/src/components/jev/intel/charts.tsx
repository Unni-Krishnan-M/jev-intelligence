"use client";

import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { SEVERITY_META } from "@/components/jev/intel/badges";
import { fmtMonth, fmtSeriesValue } from "@/lib/intel";
import type { CalibrationBin, ForecastPoint, Risk, SeriesPoint, Severity } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

/*
 * Chart conventions (shared with admin/charts.tsx): hairline solid grid, mono ticks in muted ink,
 * 2px lines, >= 8px markers with a 2px surface ring, fills as a ~12 % wash, legend below the plot,
 * one y-axis only. Every chart ships with a table view so no value is gated behind hover.
 * Categorical slots follow the validated --sig-* order; status colours are reserved for severity.
 */

export const SLOT_COLORS = [
  "var(--sig-content)",
  "var(--sig-collaborative)",
  "var(--sig-latent)",
  "var(--sig-popularity)",
  "var(--sig-preference)",
  "var(--sig-recency)",
];

const TICK = { fill: "var(--muted-foreground)", fontSize: 11, fontFamily: "var(--font-plex-mono)" };

interface TipEntry {
  name?: string | number;
  value?: unknown;
  color?: string;
  dataKey?: unknown;
  payload?: Record<string, unknown>;
}

function asTips(p: unknown): TipEntry[] {
  return Array.isArray(p) ? (p as TipEntry[]) : [];
}

function TipBox({ title, rows }: { title?: string; rows: { key: string; label: string; value: string; color?: string; ring?: boolean }[] }) {
  if (!rows.length) return null;
  return (
    <div className="rounded-md border bg-popover px-2.5 py-2 text-xs shadow-xl shadow-black/30">
      {title && <p className="mb-1 text-muted-foreground">{title}</p>}
      <ul className="space-y-0.5">
        {rows.map((r) => (
          <li key={r.key} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5">
              {r.color && (
                <span className={cn("size-2 rounded-full", r.ring && "border-2 bg-transparent")} style={r.ring ? { borderColor: r.color } : { background: r.color }} aria-hidden />
              )}
              <span className="text-foreground">{r.label}</span>
            </span>
            <span className="num text-foreground">{r.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export type LegendItem = { label: string; color: string; kind: "line" | "dashed" | "band" | "dot" | "ring" | "rule" };

export function ChartLegend({ items, className }: { items: LegendItem[]; className?: string }) {
  return (
    <ul className={cn("mt-3 flex flex-wrap gap-x-4 gap-y-1.5", className)} aria-label="Legend">
      {items.map((s) => (
        <li key={s.label} className="flex items-center gap-1.5 text-xs text-ink-2">
          {s.kind === "line" && <span className="h-0.5 w-4 rounded-full" style={{ background: s.color }} aria-hidden />}
          {s.kind === "dashed" && <span className="w-4 border-t-2 border-dashed" style={{ borderColor: s.color }} aria-hidden />}
          {s.kind === "rule" && <span className="h-3 w-px" style={{ background: s.color }} aria-hidden />}
          {s.kind === "band" && <span className="h-2.5 w-4 rounded-[2px]" style={{ background: s.color, opacity: 0.25 }} aria-hidden />}
          {s.kind === "dot" && <span className="size-2 rounded-full" style={{ background: s.color }} aria-hidden />}
          {s.kind === "ring" && <span className="size-2 rounded-full border-2" style={{ borderColor: s.color }} aria-hidden />}
          {s.label}
        </li>
      ))}
    </ul>
  );
}

/** Tiny trend line for list rows. Decorative (aria-hidden): the row carries the numbers. */
export function Sparkline({ values, className, height = 24 }: { values: number[]; className?: string; height?: number }) {
  if (values.length < 2) return <span className={cn("block text-xs text-muted-foreground", className)}>—</span>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * 100},${22 - ((v - min) / span) * 20}`).join(" ");
  const lastY = ((22 - ((values[values.length - 1] - min) / span) * 20) / 24) * 100;
  return (
    <span className={cn("relative block", className)} style={{ height }} aria-hidden>
      <svg viewBox="0 0 100 24" preserveAspectRatio="none" className="absolute inset-0 size-full overflow-visible">
        <polyline points={pts} fill="none" stroke="var(--ink-2)" strokeWidth={1.5} vectorEffect="non-scaling-stroke" strokeLinejoin="round" strokeLinecap="round" />
      </svg>
      <span className="absolute right-0 size-1.5 -translate-y-1/2 translate-x-1/2 rounded-full bg-primary" style={{ top: `${lastY}%` }} />
    </span>
  );
}

// ---- time series with trend window, change point, anomalies and forecast bands ---------------

export interface MeanLine {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
  points: { t: string; mean: number }[];
}

export interface Marker {
  t: string;
  v: number;
  severity: Severity;
  label: string;
}

type Row = { t: string } & Record<string, number | [number, number] | string | null | undefined>;

/**
 * History (solid), optional 80 % band (wash) and mean lines (dashed = projection) on one y-axis.
 * The last observed point seeds every projection so lines join where the forecast is issued.
 */
export function TimeSeriesChart({
  history,
  band,
  bandLabel = "80 % interval",
  bandColor = "var(--sig-content)",
  means = [],
  metric,
  window,
  changePoint,
  markers = [],
  historyLabel = "Observed",
  historyColor = "var(--sig-content)",
  height = 280,
  showLegend = true,
  fitToLines = false,
}: {
  history: SeriesPoint[] | { t: string; v: number }[];
  band?: ForecastPoint[];
  bandLabel?: string;
  bandColor?: string;
  means?: MeanLine[];
  metric: string;
  window?: { start: string; end: string } | null;
  changePoint?: { t: string; label: string } | null;
  markers?: Marker[];
  historyLabel?: string;
  historyColor?: string;
  height?: number;
  showLegend?: boolean;
  /** scale the y-axis to the lines only; a wider band is clipped at the plot edge */
  fitToLines?: boolean;
}) {
  const map = new Map<string, Row>();
  const row = (t: string) => {
    const k = t.slice(0, 10);
    let r = map.get(k);
    if (!r) {
      r = { t: k };
      map.set(k, r);
    }
    return r;
  };
  for (const p of history) {
    const r = row(p.t);
    r.v = p.v;
    if ("partial" in p && p.partial) r.partial = p.v;
  }
  const last = history.length ? history[history.length - 1] : null;
  if (band?.length) {
    if (last) row(last.t).band = [last.v, last.v];
    for (const p of band) row(p.t).band = [p.lo80, p.hi80];
  }
  for (const m of means) {
    if (last) row(last.t)[m.key] = last.v;
    for (const p of m.points) row(p.t)[m.key] = p.mean;
  }
  const markerBy = new Map(markers.map((m) => [m.t.slice(0, 10), m]));
  for (const m of markers) {
    const r = row(m.t);
    r.anomaly = m.v;
  }
  const data = [...map.values()].sort((a, b) => (a.t < b.t ? -1 : 1));
  const fmt = (v: number) => fmtSeriesValue(v, metric);
  const hasPartial = data.some((d) => typeof d.partial === "number");
  const lastT = last ? last.t.slice(0, 10) : null;
  const projecting = Boolean(band?.length || means.length);
  let yDomain: [number | string, number | string] = ["auto", "auto"];
  let clipped = false;
  if (fitToLines) {
    const vals = data.flatMap((d) => [d.v, ...means.map((m) => d[m.key])]).filter((x): x is number => typeof x === "number");
    if (vals.length) {
      const lo = Math.min(...vals);
      const hi = Math.max(...vals);
      const pad = (hi - lo || Math.abs(hi) || 1) * 0.12;
      yDomain = [Math.max(0, lo - pad), hi + pad];
      clipped = (band ?? []).some((p) => p.hi80 > hi + pad || p.lo80 < lo - pad);
    }
  }

  const labels: Record<string, string> = { v: historyLabel, band: bandLabel, partial: "Partial month", anomaly: "Anomaly", ...Object.fromEntries(means.map((m) => [m.key, m.label])) };
  const colors: Record<string, string> = { v: historyColor, band: bandColor, partial: historyColor, ...Object.fromEntries(means.map((m) => [m.key, m.color])) };

  const legend: LegendItem[] = [
    { label: historyLabel, color: historyColor, kind: "line" },
    ...(band?.length ? [{ label: bandLabel, color: bandColor, kind: "band" as const }] : []),
    ...means.map((m) => ({ label: m.label, color: m.color, kind: (m.dashed ? "dashed" : "line") as LegendItem["kind"] })),
    ...(window ? [{ label: "Trend window", color: "var(--muted-foreground)", kind: "band" as const }] : []),
    ...(changePoint ? [{ label: changePoint.label, color: "var(--foreground)", kind: "rule" as const }] : []),
    ...(markers.length ? [{ label: "Anomaly (colour = severity; see table)", color: "var(--status-serious)", kind: "dot" as const }] : []),
    ...(hasPartial ? [{ label: "Partial month (incomplete)", color: historyColor, kind: "ring" as const }] : []),
  ];

  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--rule)" strokeWidth={1} />
          <XAxis dataKey="t" tick={TICK} axisLine={{ stroke: "var(--rule)" }} tickLine={false} minTickGap={28} tickFormatter={(t: string) => t.slice(0, 7)} />
          <YAxis tick={TICK} axisLine={false} tickLine={false} width={56} domain={yDomain} allowDataOverflow={fitToLines} tickFormatter={fmt} />
          {window && <ReferenceArea x1={window.start.slice(0, 10)} x2={window.end.slice(0, 10)} fill="var(--muted-foreground)" fillOpacity={0.08} strokeOpacity={0} ifOverflow="hidden" />}
          {changePoint && (
            <ReferenceLine
              x={changePoint.t.slice(0, 10)}
              stroke="var(--foreground)"
              strokeWidth={1}
              label={{ value: changePoint.label, position: "insideTopLeft", fill: "var(--foreground)", fontSize: 11, fontFamily: "var(--font-plex-mono)" }}
            />
          )}
          {projecting && lastT && (
            <ReferenceLine
              x={lastT}
              stroke="var(--muted-foreground)"
              strokeWidth={1}
              label={{ value: "as of", position: "insideTopRight", fill: "var(--muted-foreground)", fontSize: 11, fontFamily: "var(--font-plex-mono)" }}
            />
          )}
          <Tooltip
            cursor={{ stroke: "var(--muted-foreground)", strokeWidth: 1 }}
            content={({ active, payload, label }) => {
              if (!active) return null;
              const t = String(label ?? "");
              const mk = markerBy.get(t);
              const rows = asTips(payload)
                .filter((p) => p.value !== null && p.value !== undefined && String(p.dataKey) !== "anomaly")
                .filter((p) => !(String(p.dataKey) === "partial"))
                .map((p) => {
                  const key = String(p.dataKey);
                  const val = Array.isArray(p.value) ? `${fmt(Number(p.value[0]))} – ${fmt(Number(p.value[1]))}` : fmt(Number(p.value));
                  return { key, label: labels[key] ?? key, value: val, color: colors[key] };
                });
              if (mk) rows.push({ key: "anomaly", label: `Anomaly · ${SEVERITY_META[mk.severity].label}`, value: mk.label, color: SEVERITY_META[mk.severity].color });
              const partial = payload?.some((p) => String(p.dataKey) === "partial" && p.value !== null && p.value !== undefined);
              return <TipBox title={`${fmtMonth(t)}${partial ? " · partial month" : ""}`} rows={rows} />;
            }}
          />
          {band?.length ? (
            <Area dataKey="band" stroke="none" fill={bandColor} fillOpacity={0.14} isAnimationActive={false} connectNulls activeDot={false} />
          ) : null}
          <Line dataKey="v" name={historyLabel} stroke={historyColor} strokeWidth={2} dot={false} activeDot={{ r: 4, stroke: "var(--card)", strokeWidth: 2 }} isAnimationActive={false} connectNulls />
          {means.map((m) => (
            <Line
              key={m.key}
              dataKey={m.key}
              name={m.label}
              stroke={m.color}
              strokeWidth={2}
              strokeDasharray={m.dashed ? "5 4" : undefined}
              dot={false}
              activeDot={{ r: 4, stroke: "var(--card)", strokeWidth: 2 }}
              isAnimationActive={false}
              connectNulls
            />
          ))}
          {hasPartial && (
            <Line dataKey="partial" stroke="none" dot={{ r: 4, fill: "var(--card)", stroke: historyColor, strokeWidth: 2 }} activeDot={false} isAnimationActive={false} legendType="none" />
          )}
          {markers.length > 0 && (
            <Line
              dataKey="anomaly"
              stroke="none"
              isAnimationActive={false}
              legendType="none"
              activeDot={false}
              dot={(props: unknown) => {
                const p = props as { cx?: number; cy?: number; index?: number; payload?: Row };
                const mk = p.payload ? markerBy.get(String(p.payload.t)) : undefined;
                if (p.cx === undefined || p.cy === undefined || p.cy === null || !mk) return <g key={`a-${p.index}`} />;
                return <circle key={`a-${p.index}`} cx={p.cx} cy={p.cy} r={5} fill={SEVERITY_META[mk.severity].color} stroke="var(--card)" strokeWidth={2} />;
              }}
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
      {showLegend && <ChartLegend items={legend} />}
      {clipped && <p className="mt-1.5 text-xs text-muted-foreground">The interval runs past the top of this scale; the table view has its full range.</p>}
    </div>
  );
}

/** Forecast fan chart: history + mean + 80 % band from one forecast. */
export function FanChart({ history, points, metric, height = 280 }: { history: SeriesPoint[]; points: ForecastPoint[]; metric: string; height?: number }) {
  return (
    <TimeSeriesChart
      history={history}
      band={points}
      means={[{ key: "mean", label: "Forecast mean (estimate)", color: "var(--sig-content)", dashed: true, points }]}
      metric={metric}
      height={height}
    />
  );
}

// ---- reliability diagram ----------------------------------------------------------------------

/** Predicted vs observed rate per bin, against the y = x line of perfect calibration. */
export function ReliabilityDiagram({ bins, height = 280 }: { bins: CalibrationBin[]; height?: number }) {
  const data = [...bins].sort((a, b) => a.predicted - b.predicted);
  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 12, right: 16, bottom: 16, left: 0 }}>
          <CartesianGrid stroke="var(--rule)" strokeWidth={1} />
          <XAxis
            type="number"
            dataKey="predicted"
            domain={[0, 1]}
            ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
            tick={TICK}
            axisLine={{ stroke: "var(--rule)" }}
            tickLine={false}
            label={{ value: "predicted probability", position: "insideBottom", offset: -8, fill: "var(--muted-foreground)", fontSize: 11 }}
          />
          <YAxis type="number" domain={[0, 1]} ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]} tick={TICK} axisLine={false} tickLine={false} width={40} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--muted-foreground)" strokeWidth={1} ifOverflow="hidden" />
          <Tooltip
            cursor={{ stroke: "var(--muted-foreground)", strokeWidth: 1 }}
            content={({ active, payload }) => {
              const b = asTips(payload)[0]?.payload as CalibrationBin | undefined;
              if (!active || !b) return null;
              return (
                <TipBox
                  title={`bin ${b.bin} · n = ${b.n.toLocaleString()}`}
                  rows={[
                    { key: "p", label: "Mean predicted", value: b.predicted.toFixed(3) },
                    { key: "o", label: "Observed rate", value: b.observed.toFixed(3), color: "var(--sig-content)" },
                  ]}
                />
              );
            }}
          />
          <Line dataKey="observed" stroke="var(--sig-content)" strokeWidth={2} dot={{ r: 4, fill: "var(--sig-content)", stroke: "var(--card)", strokeWidth: 2 }} activeDot={{ r: 6 }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <ChartLegend
        items={[
          { label: "Observed lapse rate per bin", color: "var(--sig-content)", kind: "line" },
          { label: "Perfect calibration (y = x)", color: "var(--muted-foreground)", kind: "line" },
        ]}
      />
    </div>
  );
}

// ---- counts per bin ---------------------------------------------------------------------------

export function CountBars({ data, height = 180, name = "decisions" }: { data: { bin: string; n: number }[]; height?: number; name?: string }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap={2}>
        <CartesianGrid vertical={false} stroke="var(--rule)" strokeWidth={1} />
        <XAxis dataKey="bin" tick={TICK} axisLine={{ stroke: "var(--rule)" }} tickLine={false} />
        <YAxis tick={TICK} axisLine={false} tickLine={false} width={32} allowDecimals={false} />
        <Tooltip
          cursor={{ fill: "var(--accent)", opacity: 0.4 }}
          content={({ active, payload, label }) =>
            active ? (
              <TipBox title={`confidence ${label}`} rows={asTips(payload).map((p) => ({ key: "n", label: name, value: String(p.value), color: "var(--sig-content)" }))} />
            ) : null
          }
        />
        <Bar dataKey="n" fill="var(--sig-content)" radius={[4, 4, 0, 0]} maxBarSize={24} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ---- likelihood × impact ----------------------------------------------------------------------

/** Risks placed by likelihood (x) and impact (y); numbers match the register below. */
export function RiskMatrix({ risks, className }: { risks: Risk[]; className?: string }) {
  const S = 320;
  const P = 36;
  const W = S - P - 12;
  const x = (v: number) => P + Math.max(0, Math.min(1, v)) * W;
  const y = (v: number) => 12 + (1 - Math.max(0, Math.min(1, v))) * W;
  const levels = Array.from(new Set(risks.map((r) => r.level)));
  return (
    <figure className={cn("w-full", className)}>
      <svg viewBox={`0 0 ${S} ${S}`} className="mx-auto block w-full max-w-[380px]" role="img" aria-label="Risk matrix: likelihood by impact. Values are listed in the register.">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={x(g)} x2={x(g)} y1={y(0)} y2={y(1)} stroke="var(--rule)" strokeWidth={1} />
            <line x1={x(0)} x2={x(1)} y1={y(g)} y2={y(g)} stroke="var(--rule)" strokeWidth={1} />
            <text x={x(g)} y={y(0) + 14} textAnchor="middle" fontSize={10} fill="var(--muted-foreground)" fontFamily="var(--font-plex-mono)">{g}</text>
            <text x={x(0) - 6} y={y(g) + 3} textAnchor="end" fontSize={10} fill="var(--muted-foreground)" fontFamily="var(--font-plex-mono)">{g}</text>
          </g>
        ))}
        <text x={x(0.5)} y={S - 2} textAnchor="middle" fontSize={11} fill="var(--ink-2)">likelihood →</text>
        <text x={10} y={y(0.5)} textAnchor="middle" fontSize={11} fill="var(--ink-2)" transform={`rotate(-90 10 ${y(0.5)})`}>impact →</text>
        {risks.map((r, i) => (
          <a key={r.id} href={`#${r.id}`} aria-label={`${i + 1}. ${r.title}: likelihood ${r.likelihood.toFixed(2)}, impact ${r.impact.toFixed(2)}, ${r.level}`}>
            <title>{`${i + 1}. ${r.title} — likelihood ${r.likelihood.toFixed(2)}, impact ${r.impact.toFixed(2)}, score ${r.score.toFixed(0)} (${r.level})`}</title>
            <circle cx={x(r.likelihood)} cy={y(r.impact)} r={12} fill="transparent" />
            <circle cx={x(r.likelihood)} cy={y(r.impact)} r={5} fill={SEVERITY_META[r.level].color} stroke="var(--card)" strokeWidth={2} />
            <text x={x(r.likelihood) + 8} y={y(r.impact) - 6} fontSize={11} fill="var(--foreground)" fontFamily="var(--font-plex-mono)">{i + 1}</text>
          </a>
        ))}
      </svg>
      <ChartLegend
        className="justify-center"
        items={(["critical", "high", "medium", "low"] as Severity[]).filter((l) => levels.includes(l)).map((l) => ({ label: `${SEVERITY_META[l].label} level`, color: SEVERITY_META[l].color, kind: "dot" }))}
      />
    </figure>
  );
}

// ---- small HTML bars --------------------------------------------------------------------------

/** One bar per option; the chosen answer in the accent, the rest in neutral ink. */
export function OptionScores({ scores, answer, className }: { scores: Record<string, number>; answer: string | null; className?: string }) {
  const entries = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, v]) => v), 1e-9);
  return (
    <ul className={cn("space-y-1.5", className)}>
      {entries.map(([opt, v]) => (
        <li key={opt} className="grid grid-cols-[minmax(64px,120px)_1fr_48px] items-center gap-2 text-sm">
          <span className={cn("truncate", opt === answer ? "text-foreground" : "text-muted-foreground")}>
            {opt}
            {opt === answer && <span className="sr-only"> (chosen)</span>}
          </span>
          <span className="h-2 overflow-hidden rounded-r-[4px] bg-transparent">
            <span className={cn("block h-full rounded-r-[4px]", opt === answer ? "bg-primary" : "bg-muted-foreground/40")} style={{ width: `${(v / max) * 100}%` }} />
          </span>
          <span className="num text-right text-xs text-ink-2">{v.toFixed(2)}</span>
        </li>
      ))}
    </ul>
  );
}

/** Horizontal bars for named durations or contributions (single series, neutral + accent max). */
export function NamedBars({ rows, format, className }: { rows: { name: string; value: number; note?: string }[]; format: (v: number) => string; className?: string }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  return (
    <ul className={cn("space-y-1.5", className)}>
      {rows.map((r) => (
        <li key={r.name} className="grid grid-cols-[minmax(72px,140px)_1fr_64px] items-center gap-2 text-sm">
          <span className="truncate text-ink-2" title={r.note ?? r.name}>{r.name}</span>
          <span className="h-2">
            <span className="block h-full rounded-r-[4px] bg-sig-content" style={{ width: `${(Math.abs(r.value) / max) * 100}%` }} />
          </span>
          <span className="num text-right text-xs">{format(r.value)}</span>
        </li>
      ))}
    </ul>
  );
}

/** A table-view twin for any chart, folded by default. */
export function TableView({ caption, head, rows }: { caption: string; head: string[]; rows: (string | number)[][] }) {
  return (
    <details className="group mt-3">
      <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
        <span className="transition-transform group-open:rotate-90" aria-hidden>›</span> Table view
      </summary>
      <div className="mt-2 max-h-72 overflow-auto rounded border hairline">
        <table className="w-full text-xs">
          <caption className="sr-only">{caption}</caption>
          <thead className="sticky top-0 bg-card">
            <tr className="border-b hairline text-left text-muted-foreground">
              {head.map((h, i) => <th key={h} className={cn("px-2.5 py-1.5 font-normal", i > 0 && "text-right")}>{h}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b hairline last:border-0">
                {r.map((c, j) => <td key={j} className={cn("num px-2.5 py-1", j > 0 && "text-right")}>{c}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
