"use client";

/*
 * Recommender monitoring. The calibration block follows the usual reliability display
 * (scikit-learn's calibration guide, hollance/reliability-diagrams): a reliability diagram
 * against y = x, a per-bin table, the ECE / Brier figures beside it with the base-rate Brier
 * for scale, and the histogram of served confidences underneath so sparse bins are visible.
 * Without a calibration file the page says so instead of drawing anything.
 */

import useSWR from "swr";

import { fmtDate, fmtNum, Panel, SpecRows, StatTile } from "@/components/jev/admin/ui";
import { ActionsBoard } from "@/components/jev/intel/actions-board";
import { CountBars, DailyBars, NamedBars, ReliabilityDiagram, TableView } from "@/components/jev/intel/charts";
import Link, { CapabilityNotice, useIntelDomain } from "@/components/jev/intel/domain-context";
import { PageHeader } from "@/components/jev/intel/page-header";
import { IntelError, PanelsSkeleton, RowsSkeleton } from "@/components/jev/intel/states";
import { EmptyState, SectionHeader } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { qs } from "@/lib/api";
import { strategyLabel } from "@/lib/decisions";
import { calibrationView, fmtPct, humanize } from "@/lib/intel";
import type { Action, Page, RecommenderMonitoring } from "@/lib/intel-types";

/** Below this many served recommendations a reason code's positive rate is too noisy to rank by. */
const MIN_SERVED = 20;

/** Say plainly when the calibrated confidence barely separates hits from misses. */
function aucNote(cal: RecommenderMonitoring["calibration"]): string {
  const aucs = [cal?.headline?.auc, ...(cal?.strata ?? []).map((s) => s.test?.auc)].filter((v): v is number => typeof v === "number");
  if (!aucs.length) return "";
  const lo = Math.min(...aucs);
  const hi = Math.max(...aucs);
  const range = lo === hi ? lo.toFixed(2) : `${lo.toFixed(2)}–${hi.toFixed(2)}`;
  return hi < 0.7
    ? `Discrimination is weak (AUC ${range}; 0.5 is chance): well calibrated, but it separates hits from misses only a little.`
    : `AUC ${range} (0.5 is chance).`;
}

function Calibration({ m }: { m: RecommenderMonitoring }) {
  const v = calibrationView(m.calibration);
  if (!v) {
    return (
      <EmptyState
        title="Not calibrated"
        body={`Model ${m.model_version ?? "(none)"} has no calibration file, so members see no confidence figure. Run scripts/calibrate_recommendations.py for this model version to fit one on held-out ratings.`}
      />
    );
  }
  return (
    <div className={v.bins.length ? "grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]" : "grid grid-cols-1 gap-4 lg:grid-cols-2"}>
      {v.bins.length > 0 && (
        <Panel title="Reliability diagram">
          <ReliabilityDiagram bins={v.bins} observedLabel="Observed share rated 4★+ in the next 5 ratings" />
          <TableView
            caption="Recommendation calibration bins"
            head={["bin", "predicted", "observed", "n"]}
            rows={v.bins.map((b) => [b.bin, b.predicted.toFixed(3), b.observed.toFixed(3), b.n.toLocaleString()])}
          />
        </Panel>
      )}
      <div className={v.bins.length ? "space-y-4" : "grid grid-cols-1 gap-4 lg:col-span-2 lg:grid-cols-2"}>
        <div className="grid grid-cols-2 content-start gap-4">
          <StatTile label="ECE" value={fmtNum(v.ece, 3)} hint="expected calibration error; 0 is perfect" />
          <StatTile label="Brier" value={fmtNum(v.brier, 3)} hint={v.baseRateBrier !== null ? `base-rate forecaster ${fmtNum(v.baseRateBrier, 3)}` : "lower is better"} />
        </div>
        <Panel title="Calibration">
          <SpecRows
            rows={[
              { label: "Figures from", value: v.split },
              { label: "Method", value: v.method ?? "—" },
              { label: "Fitted on", value: v.fittedOn ?? "—" },
              { label: "Rows scored", value: v.n === null ? "—" : v.n.toLocaleString() },
            ]}
          />
          {m.calibration?.target && <p className="mt-3 text-xs leading-relaxed text-muted-foreground">Target: {m.calibration.target}.</p>}
          {!v.bins.length && <p className="mt-2 text-xs text-muted-foreground">The calibration file carries no reliability bins, so no diagram is drawn; the per-stratum figures below compare predicted and observed rates.</p>}
        </Panel>
      </div>
      {m.calibration?.strata?.length ? (
        <Panel title="Per profile-size stratum" className="lg:col-span-2">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <caption className="sr-only">Calibration per profile-size stratum, held-out test split</caption>
              <thead>
                <tr className="border-b hairline text-left text-xs text-muted-foreground">
                  <th className="py-2 pr-2 font-normal">Stratum</th>
                  <th className="px-2 py-2 font-normal">Applies to</th>
                  <th className="px-2 py-2 font-normal">Feature</th>
                  <th className="px-2 py-2 text-right font-normal">Observed</th>
                  <th className="px-2 py-2 text-right font-normal">Predicted</th>
                  <th className="px-2 py-2 text-right font-normal">ECE ↓</th>
                  <th className="px-2 py-2 text-right font-normal">Brier (base)</th>
                  <th className="py-2 pl-2 text-right font-normal">AUC</th>
                </tr>
              </thead>
              <tbody>
                {m.calibration.strata.map((st) => {
                  const t = st.test ?? null;
                  return (
                    <tr key={st.name} className="border-b hairline last:border-0">
                      <td className="py-2 pr-2">{humanize(st.name)}</td>
                      <td className="num px-2 py-2 text-xs text-ink-2">{st.applies_to ? `${st.applies_to.min}${st.applies_to.max === null || st.applies_to.max === undefined ? "+" : `–${st.applies_to.max}`} ${st.applies_to.unit}` : "—"}</td>
                      <td className="px-2 py-2 text-xs text-ink-2">{st.feature ?? "—"}</td>
                      <td className="num px-2 py-2 text-right">{fmtPct(t?.observed_rate ?? null, 2)}</td>
                      <td className="num px-2 py-2 text-right">{fmtPct(t?.mean_predicted ?? null, 2)}</td>
                      <td className="num px-2 py-2 text-right">{fmtNum(t?.ece ?? null, 4)}</td>
                      <td className="num px-2 py-2 text-right">{fmtNum(t?.brier ?? null, 4)} <span className="text-xs text-muted-foreground">({fmtNum(t?.base_rate_brier ?? null, 4)})</span></td>
                      <td className="num py-2 pl-2 text-right">{fmtNum(t?.auc ?? null, 3)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-muted-foreground">
            Held-out test figures. {aucNote(m.calibration)} A Brier close to the base-rate forecaster&apos;s means the confidence is honest on
            average but says little beyond it.
          </p>
        </Panel>
      ) : null}
      {m.calibration?.assumptions?.length ? (
        <details className="group lg:col-span-2">
          <summary className="inline-flex cursor-pointer list-none items-center gap-1 text-xs text-muted-foreground hover:text-foreground [&::-webkit-details-marker]:hidden">
            <span className="transition-transform group-open:rotate-90" aria-hidden>›</span> Assumptions ({m.calibration.assumptions.length})
          </summary>
          <ul className="mt-2 max-w-3xl space-y-2 text-sm text-ink-2">
            {m.calibration.assumptions.map((a) => <li key={a} className="border-l-2 border-rule pl-3">{a}</li>)}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

function Monitoring({ m }: { m: RecommenderMonitoring }) {
  const f = m.feedback_totals;
  const codes = [...m.reason_codes].sort((a, b) => b.served - a.served);
  const hist = m.confidence_histogram ?? [];
  const withConfidence = m.recent.filter((r) => r.confidence !== null && r.confidence !== undefined).length;

  return (
    <div className="space-y-12">
      <section aria-labelledby="rec-cal">
        <SectionHeader id="rec-cal" index={1} kicker="calibration · held-out ratings" title="Does the confidence mean what it says?" />
        <Calibration m={m} />
      </section>

      <section aria-labelledby="rec-served">
        <SectionHeader id="rec-served" index={2} kicker="traffic" title="What was served" />
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
          <StatTile label="Served" value={m.served.total.toLocaleString()} hint="recommendations recorded" />
          <StatTile label="Clicked" value={f.clicked.toLocaleString()} />
          <StatTile label="Liked" value={f.like.toLocaleString()} />
          <StatTile label="Disliked" value={f.dislike.toLocaleString()} />
          <StatTile label="Not interested" value={f.not_interested.toLocaleString()} />
        </div>
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel title="Served per day">
            {m.served.per_day.length ? (
              <>
                <DailyBars data={m.served.per_day} name="recommendations" />
                <TableView caption="Recommendations served per day" head={["date", "served"]} rows={m.served.per_day.map((d) => [d.date.slice(0, 10), d.count.toLocaleString()])} />
              </>
            ) : (
              <p className="text-sm text-muted-foreground">Nothing served yet.</p>
            )}
          </Panel>
          <Panel title="Served confidence · P(4★+ among next 5 ratings)">
            {!hist.length || hist.every((b) => b.n === 0) ? (
              <p className="text-sm text-muted-foreground">
                {m.calibration ? "No served recommendation carries a confidence yet." : "No confidences: the active model is not calibrated."}
              </p>
            ) : (
              <>
                <CountBars data={hist} name="recommendations" binLabel="confidence" tickFormatter={(b) => b.split("-")[0]} />
                <TableView caption="Served recommendations per confidence bin" head={["bin", "n"]} rows={hist.map((b) => [b.bin, b.n.toLocaleString()])} />
                <p className="mt-2 text-xs text-muted-foreground">How confident the served list was, not whether it was right; the reliability diagram answers that.</p>
              </>
            )}
          </Panel>
        </div>
      </section>

      {m.strategies && Object.keys(m.strategies).length > 0 && (
        <section aria-labelledby="rec-strategy">
          <SectionHeader id="rec-strategy" kicker="JEV decisions · recommendation strategy" title="Which strategy each list was served under" />
          <NamedBars
            rows={Object.entries(m.strategies)
              .sort((a, b) => b[1] - a[1])
              .map(([k, v]) => ({ name: k === "unrecorded" ? "unrecorded (before v1.2)" : strategyLabel(k), value: v }))}
            format={(v) => v.toLocaleString()}
          />
          <p className="mt-2 text-xs text-muted-foreground">
            Served rows per strategy. Each member&apos;s list is downstream of their recommendation-strategy decision; an abstention is served as standard.
          </p>
        </section>
      )}

      <section aria-labelledby="rec-reasons">
        <SectionHeader id="rec-reasons" index={3} kicker="explanations" title="Which reasons land?" />
        {codes.length === 0 ? (
          <p className="text-sm text-muted-foreground">No reason codes recorded yet.</p>
        ) : (
          <>
            <div className="overflow-x-auto rounded-lg border bg-card">
              <table className="w-full min-w-[720px] text-sm">
                <caption className="sr-only">Feedback per reason code</caption>
                <thead>
                  <tr className="border-b hairline text-left text-xs text-muted-foreground">
                    <th className="px-4 py-2.5 font-normal">Reason code</th>
                    <th className="px-2 py-2.5 text-right font-normal">Served</th>
                    <th className="px-2 py-2.5 text-right font-normal">Clicked</th>
                    <th className="px-2 py-2.5 text-right font-normal">Liked</th>
                    <th className="px-2 py-2.5 text-right font-normal">Disliked</th>
                    <th className="px-2 py-2.5 text-right font-normal">Not interested</th>
                    <th className="w-48 px-4 py-2.5 font-normal">Positive rate</th>
                  </tr>
                </thead>
                <tbody>
                  {codes.map((c) => {
                    const thin = c.served < MIN_SERVED;
                    return (
                      <tr key={c.code} className="border-b hairline last:border-0">
                        <td className="px-4 py-2">{humanize(c.code)}</td>
                        <td className="num px-2 py-2 text-right">{c.served.toLocaleString()}</td>
                        <td className="num px-2 py-2 text-right text-ink-2">{c.clicked.toLocaleString()}</td>
                        <td className="num px-2 py-2 text-right text-ink-2">{c.like.toLocaleString()}</td>
                        <td className="num px-2 py-2 text-right text-ink-2">{c.dislike.toLocaleString()}</td>
                        <td className="num px-2 py-2 text-right text-ink-2">{c.not_interested.toLocaleString()}</td>
                        <td className="px-4 py-2">
                          {c.positive_rate === null ? (
                            <span className="text-xs text-muted-foreground">no feedback</span>
                          ) : (
                            <span className="flex items-center gap-2">
                              <span className="h-2 flex-1 rounded-full bg-primary/15">
                                <span className="block h-full rounded-full bg-primary" style={{ width: `${Math.max(0, Math.min(1, c.positive_rate)) * 100}%` }} />
                              </span>
                              <span className="num w-12 text-right text-xs">{fmtPct(c.positive_rate)}</span>
                              {thin && <span className="text-[10px] uppercase tracking-[0.1em] text-muted-foreground" title={`fewer than ${MIN_SERVED} served`}>thin</span>}
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">Positive rate = likes ÷ (likes + dislikes + not interested); clicks are not counted. Rows marked “thin” have fewer than {MIN_SERVED} served, too few to compare.</p>
          </>
        )}
      </section>

      <section aria-labelledby="rec-recent">
        <SectionHeader id="rec-recent" index={4} kicker="log" title="Recently served" />
        {m.recent.length === 0 ? (
          <p className="text-sm text-muted-foreground">Nothing served yet.</p>
        ) : (
          <>
            <div className="overflow-x-auto rounded-lg border bg-card">
              <table className="w-full min-w-[860px] text-sm">
                <caption className="sr-only">Recently served recommendations</caption>
                <thead>
                  <tr className="border-b hairline text-left text-xs text-muted-foreground">
                    <th className="px-4 py-2.5 font-normal">When</th>
                    <th className="px-2 py-2.5 font-normal">User</th>
                    <th className="px-2 py-2.5 font-normal">Film</th>
                    <th className="px-2 py-2.5 text-right font-normal">Rank</th>
                    <th className="px-2 py-2.5 text-right font-normal">Score</th>
                    <th className="px-2 py-2.5 text-right font-normal" title="Calibrated P(rated 4★+ among the member's next 5 ratings)">Confidence</th>
                    <th className="px-2 py-2.5 font-normal">Strategy</th>
                    <th className="px-4 py-2.5 font-normal">Reason</th>
                  </tr>
                </thead>
                <tbody>
                  {m.recent.map((r) => (
                    <tr key={r.id} className="border-b hairline align-top last:border-0">
                      <td className="num whitespace-nowrap px-4 py-2 text-xs text-muted-foreground">{fmtDate(r.created_at)}</td>
                      <td className="num px-2 py-2 text-xs">{r.user_id}</td>
                      <td className="px-2 py-2"><Link href={`/movies/${r.movie_id}`} className="hover:underline">{r.title}</Link></td>
                      <td className="num px-2 py-2 text-right">{r.rank}</td>
                      <td className="num px-2 py-2 text-right text-ink-2">{r.score.toFixed(3)}</td>
                      <td className="num px-2 py-2 text-right">{r.confidence === null || r.confidence === undefined ? <span className="text-muted-foreground">—</span> : fmtPct(r.confidence, 1)}</td>
                      <td className="px-2 py-2 text-xs" title={r.decision_id ?? undefined}>{r.strategy ? strategyLabel(r.strategy) : <span className="text-muted-foreground">—</span>}</td>
                      <td className="px-4 py-2 text-xs text-ink-2">
                        <span className="eyebrow block">{humanize(r.reason_code)}</span>
                        {r.reason}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">{withConfidence} of {m.recent.length} carry a calibrated confidence; “—” means none was available when it was served.</p>
          </>
        )}
      </section>
    </div>
  );
}

/** For domains without a recommender: JEV's recommended actions, each with its evidence. */
function RecommendedActions() {
  const { q: dq } = useIntelDomain();
  const { data, error, mutate } = useSWR<Page<Action>>(dq(`/intel/actions${qs({ limit: 100 })}`));
  const items = data?.items ?? [];
  if (error) return <IntelError error={error} retry={() => mutate()} />;
  if (!data) return <RowsSkeleton rows={5} />;
  if (!items.length) return <EmptyState title="No recommended actions" body="Nothing in the latest run for this domain calls for an action." />;
  return (
    <>
      <p className="mb-6 font-mono text-xs text-muted-foreground">
        {items.length.toLocaleString()} action{items.length === 1 ? "" : "s"}{data.as_of ? ` · as of ${fmtDate(data.as_of)}` : ""}
      </p>
      <ActionsBoard items={items} />
    </>
  );
}

function RecommenderMonitoringView() {
  const { q: dq } = useIntelDomain();
  const { data, error, mutate } = useSWR<RecommenderMonitoring>(dq("/intel/recommendations"));
  return (
    <>
      {data?.model_version && <p className="-mt-4 mb-6 font-mono text-xs text-muted-foreground">model {data.model_version}</p>}
      {error ? (
        <IntelError error={error} retry={() => mutate()} runBacked={false} what="recommender monitoring (GET /intel/recommendations)" />
      ) : !data ? (
        <PanelsSkeleton n={6} />
      ) : (
        <Monitoring m={data} />
      )}
    </>
  );
}

export default function RecommendationsPage() {
  const { can, name } = useIntelDomain();
  const recommender = can("recommendation");
  const perUser = can("user_intelligence");

  return (
    <div>
      <PageHeader
        eyebrow="recommend"
        title={recommender.available ? "Recommendations" : "Recommended actions"}
        description={
          recommender.available
            ? "How the member-facing recommender is doing: its calibration, what it served and how people responded. Each member's list is produced under a JEV strategy decision."
            : `What JEV recommends doing about the ${name} situation: actions ranked by priority, downstream of its decisions, warnings and risks, each with the evidence behind it.`
        }
        action={
          recommender.available && perUser.available ? (
            <Button asChild variant="outline" size="sm"><Link href="/me/intelligence">Per-user intelligence →</Link></Button>
          ) : undefined
        }
      />
      {recommender.available ? (
        <RecommenderMonitoringView />
      ) : (
        <>
          <CapabilityNotice cap="recommendation" className="mb-6" />
          <RecommendedActions />
        </>
      )}
    </div>
  );
}
