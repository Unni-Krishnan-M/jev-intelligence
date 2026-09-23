"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { cn } from "@/lib/utils";

/** Model display names and fixed categorical slots (colour follows the model, never its rank). */
export const MODEL_ORDER = ["hybrid", "als", "itemknn", "content", "popularity", "random"] as const;
export type ModelName = (typeof MODEL_ORDER)[number];
export const MODEL_LABEL: Record<string, string> = {
  hybrid: "Hybrid",
  als: "ALS (MF)",
  itemknn: "Item-kNN (CF)",
  content: "Content (TF-IDF)",
  popularity: "Popularity",
  random: "Random",
};
export const MODEL_COLOR: Record<string, string> = {
  hybrid: "var(--sig-content)",
  als: "var(--sig-collaborative)",
  itemknn: "var(--sig-latent)",
  content: "var(--sig-popularity)",
  popularity: "var(--sig-preference)",
  random: "var(--sig-recency)",
};
export const PEER = "#5a544b";

const TICK = { fill: "var(--muted-foreground)", fontSize: 11, fontFamily: "var(--font-plex-mono)" };

interface TipEntry {
  name?: string | number;
  value?: number | string;
  color?: string;
  payload?: Record<string, unknown>;
}

// Recharts' payload type is broad; we only read name/value/color/payload.
function asTips(p: unknown): TipEntry[] | undefined {
  return Array.isArray(p) ? (p as TipEntry[]) : undefined;
}

export function ChartTooltip({
  active,
  payload,
  label,
  digits = 4,
  labelFormat,
}: {
  active?: boolean;
  payload?: TipEntry[];
  label?: string | number;
  digits?: number;
  labelFormat?: (l: string | number | undefined) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-md border bg-popover px-2.5 py-2 text-xs shadow-xl shadow-black/30">
      {label !== undefined && <p className="mb-1 text-muted-foreground">{labelFormat ? labelFormat(label) : label}</p>}
      <ul className="space-y-0.5">
        {payload.map((p, i) => (
          <li key={`${p.name}-${i}`} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5">
              <span className="size-2 rounded-full" style={{ background: p.color }} aria-hidden />
              <span className="text-foreground">{String(p.name ?? "")}</span>
            </span>
            <span className="num text-foreground">{typeof p.value === "number" ? p.value.toFixed(digits) : p.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

/** One metric across models, hybrid highlighted; values labelled on hybrid + the max. */
export function MetricBars({ title, rows, digits = 3 }: { title: string; rows: { model: string; value: number }[]; digits?: number }) {
  const max = Math.max(...rows.map((r) => r.value), 0);
  const data = rows.map((r) => ({
    ...r,
    label: MODEL_LABEL[r.model] ?? r.model,
    shown: r.model === "hybrid" || r.value === max ? r.value.toFixed(digits) : "",
  }));
  return (
    <figure className="rounded-lg border bg-card p-4">
      <figcaption className="eyebrow mb-2">{title}</figcaption>
      <ResponsiveContainer width="100%" height={rows.length * 30 + 20}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 44, bottom: 0, left: 0 }} barCategoryGap={2}>
          <CartesianGrid horizontal={false} stroke="var(--rule)" strokeWidth={1} />
          <XAxis type="number" hide domain={[0, max > 0 ? max * 1.05 : 1]} />
          <YAxis type="category" dataKey="label" width={112} tick={TICK} axisLine={false} tickLine={false} />
          <Tooltip
            cursor={{ fill: "var(--accent)", opacity: 0.4 }}
            content={({ active, payload }) => (
              <ChartTooltip
                active={active}
                digits={digits + 1}
                payload={asTips(payload)?.map((p) => ({
                  name: String(p.payload?.label ?? ""),
                  value: p.value,
                  color: MODEL_COLOR[String(p.payload?.model)] ?? PEER,
                }))}
              />
            )}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={16} isAnimationActive={false}>
            {data.map((d) => (
              <Cell key={d.model} fill={d.model === "hybrid" ? MODEL_COLOR.hybrid : PEER} />
            ))}
            <LabelList dataKey="shown" position="right" style={{ fill: "var(--foreground)", fontSize: 11, fontFamily: "var(--font-plex-mono)" }} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </figure>
  );
}

/** Single-series line (e.g. ALS objective). The title names the series, so no legend is needed. */
export function SeriesLine({
  data,
  xKey,
  yKey,
  name,
  height = 220,
  digits = 4,
  className,
}: {
  data: Record<string, number>[];
  xKey: string;
  yKey: string;
  name: string;
  height?: number;
  digits?: number;
  className?: string;
}) {
  return (
    <div className={cn("w-full", className)}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--rule)" strokeWidth={1} />
          <XAxis dataKey={xKey} tick={TICK} axisLine={{ stroke: "var(--rule)" }} tickLine={false} />
          <YAxis tick={TICK} axisLine={false} tickLine={false} width={56} domain={["auto", "auto"]} tickFormatter={(v: number) => v.toFixed(3)} />
          <Tooltip
            cursor={{ stroke: "var(--muted-foreground)", strokeWidth: 1 }}
            content={({ active, payload, label }) => (
              <ChartTooltip active={active} payload={asTips(payload)} label={label} digits={digits} labelFormat={(l) => `${xKey} ${l}`} />
            )}
          />
          <Line type="monotone" dataKey={yKey} name={name} stroke="var(--sig-content)" strokeWidth={2} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Multi-series line with legend below the plot (never covering data). */
export function MultiLine({
  data,
  xKey,
  series,
  height = 260,
  xFormat,
}: {
  data: Record<string, number | string>[];
  xKey: string;
  series: { key: string; label: string; color: string }[];
  height?: number;
  xFormat?: (v: string | number) => string;
}) {
  return (
    <div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
          <CartesianGrid vertical={false} stroke="var(--rule)" strokeWidth={1} />
          <XAxis dataKey={xKey} tick={TICK} axisLine={{ stroke: "var(--rule)" }} tickLine={false} tickFormatter={xFormat} />
          <YAxis tick={TICK} axisLine={false} tickLine={false} width={48} tickFormatter={(v: number) => v.toFixed(2)} />
          <Tooltip
            cursor={{ stroke: "var(--muted-foreground)", strokeWidth: 1 }}
            content={({ active, payload, label }) => (
              <ChartTooltip
                active={active}
                payload={asTips(payload)}
                label={label}
                labelFormat={(l) => (xFormat && l !== undefined ? xFormat(l) : String(l))}
              />
            )}
          />
          {series.map((s) => (
            <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5" aria-label="Legend">
        {series.map((s) => (
          <li key={s.key} className="flex items-center gap-1.5 text-xs text-ink-2">
            <span className="h-0.5 w-4 rounded-full" style={{ background: s.color }} aria-hidden />
            {s.label}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Single-series vertical bars (e.g. recommendations per day). */
export function DailyBars({ data, height = 200 }: { data: { date: string; count: number }[]; height?: number }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap={2}>
        <CartesianGrid vertical={false} stroke="var(--rule)" strokeWidth={1} />
        <XAxis dataKey="date" tick={TICK} axisLine={{ stroke: "var(--rule)" }} tickLine={false} tickFormatter={(d: string) => d.slice(5)} />
        <YAxis tick={TICK} axisLine={false} tickLine={false} width={40} allowDecimals={false} />
        <Tooltip
          cursor={{ fill: "var(--accent)", opacity: 0.4 }}
          content={({ active, payload, label }) => (
            <ChartTooltip
              active={active}
              digits={0}
              label={label}
              payload={asTips(payload)?.map((p) => ({ name: "served", value: p.value, color: "var(--sig-content)" }))}
            />
          )}
        />
        <Bar dataKey="count" fill="var(--sig-content)" radius={[4, 4, 0, 0]} maxBarSize={36} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}
