"use client";

import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";

import { DirectionIcon } from "@/components/jev/intel/badges";
import { Sparkline } from "@/components/jev/intel/charts";
import { PageHeader } from "@/components/jev/intel/page-header";
import { FilterRow, IntelError, Pagination, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState } from "@/components/jev/states";
import { Skeleton } from "@/components/ui/skeleton";
import { qs } from "@/lib/api";
import { fmtMonth, fmtP, fmtSigned, seriesHref, seriesLabel } from "@/lib/intel";
import type { Page, SeriesDetail, Trend } from "@/lib/intel-types";

const LIMIT = 25;

/** Sparkline for one row, read from the series endpoint (shared SWR key with the detail page). */
function RowSpark({ seriesId }: { seriesId: string }) {
  const { data, error } = useSWR<SeriesDetail>(`/intel/series/${encodeURIComponent(seriesId)}`);
  if (error) return <span className="text-xs text-muted-foreground">—</span>;
  if (!data) return <Skeleton className="h-6 w-full" />;
  return <Sparkline values={data.series.points.slice(-36).map((p) => p.v)} />;
}

export default function TrendsPage() {
  const [direction, setDirection] = useState("all");
  const [offset, setOffset] = useState(0);
  const { data, error, mutate } = useSWR<Page<Trend>>(`/intel/trends${qs({ direction: direction === "all" ? null : direction, limit: LIMIT, offset })}`);

  return (
    <div>
      <PageHeader
        eyebrow="detect"
        title="Trends"
        description="Monthly series tested for a monotonic trend (Mann–Kendall) with a Theil–Sen slope and its 95 % interval, compared with the window before."
        asOf={data?.as_of}
        runId={data?.run_id}
      />

      <div className="mb-5">
        <FilterRow
          label="Direction"
          value={direction}
          onChange={(v) => { setDirection(v); setOffset(0); }}
          options={[{ value: "all", label: "All" }, { value: "up", label: "Rising" }, { value: "down", label: "Falling" }, { value: "flat", label: "Flat" }]}
        />
      </div>

      <p className="mb-4 max-w-2xl text-xs text-muted-foreground">
        <strong className="font-medium text-ink-2">Evidence strength = 1 − p.</strong> It says how surprising the pattern would be if there were no trend.
        It is <em>not</em> the probability that the trend is real, and it says nothing about how big it is — read the slope and its interval for that.
      </p>

      {error ? (
        <IntelError error={error} retry={() => mutate()} />
      ) : !data ? (
        <RowsSkeleton rows={8} />
      ) : data.items.length === 0 ? (
        <EmptyState title="No trends" body={direction === "all" ? "The latest run found no series long enough to test." : "No trends in that direction in the latest run."} />
      ) : (
        <>
          <div className="overflow-x-auto rounded-lg border bg-card">
            <table className="w-full min-w-[820px] text-sm">
              <caption className="sr-only">Trends in the latest run</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="px-4 py-2.5 font-normal">Series</th>
                  <th className="px-2 py-2.5 font-normal">Last 36 months</th>
                  <th className="px-2 py-2.5 text-right font-normal">Change / month</th>
                  <th className="px-2 py-2.5 text-right font-normal">Slope (95 % CI)</th>
                  <th className="px-2 py-2.5 text-right font-normal">p</th>
                  <th className="px-2 py-2.5 text-right font-normal">Evidence (1−p)</th>
                  <th className="px-2 py-2.5 text-right font-normal">Recent ÷ prior</th>
                  <th className="px-4 py-2.5 font-normal">Change point</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((t) => (
                  <tr key={t.id} id={t.id} className="scroll-mt-24 border-b hairline last:border-0 hover:bg-accent/40">
                    <td className="px-4 py-2.5">
                      <Link href={seriesHref(t.series_id)} className="flex items-center gap-2 hover:underline">
                        <DirectionIcon direction={t.direction} />
                        <span>
                          {seriesLabel(t.series_id)}
                          <span className="block font-mono text-xs text-muted-foreground">
                            {t.window.start.slice(0, 7)} → {t.window.end.slice(0, 7)} · {t.window.months} m
                          </span>
                        </span>
                      </Link>
                    </td>
                    <td className="w-32 px-2 py-2.5"><RowSpark seriesId={t.series_id} /></td>
                    <td className="num px-2 py-2.5 text-right">{fmtSigned(t.change_rate_pct_per_month, 2, " %")}</td>
                    <td className="num px-2 py-2.5 text-right text-xs text-ink-2">
                      {t.slope.toPrecision(3)}
                      <span className="block text-muted-foreground">[{t.slope_ci[0].toPrecision(2)}, {t.slope_ci[1].toPrecision(2)}]</span>
                    </td>
                    <td className="num px-2 py-2.5 text-right">{fmtP(t.p_value)}</td>
                    <td className="num px-2 py-2.5 text-right">{t.evidence_strength.toFixed(3)}</td>
                    <td className="num px-2 py-2.5 text-right">{t.ratio === null ? "—" : `×${t.ratio.toFixed(2)}`}</td>
                    <td className="px-4 py-2.5 text-xs text-ink-2">
                      {t.change_point ? <>{fmtMonth(t.change_point.date)} <span className="num text-muted-foreground">p {fmtP(t.change_point.p_value)}</span></> : <span className="text-muted-foreground">none</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pagination total={data.total} limit={LIMIT} offset={offset} onChange={setOffset} />
        </>
      )}
    </div>
  );
}
