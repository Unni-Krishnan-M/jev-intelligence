"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import useSWR from "swr";

import { DirectionIcon, Meter } from "@/components/jev/intel/badges";
import { NotDeployedState } from "@/components/jev/intel/states";
import { DriftReport } from "@/components/jev/me/drift-report";
import { PreferenceHistoryChart } from "@/components/jev/me/preference-history";
import { RecommendationCards } from "@/components/jev/me/recommendation-list";
import { ScenarioPanel } from "@/components/jev/me/scenario-panel";
import { StrategyPanel } from "@/components/jev/me/strategy-panel";
import { Container, EmptyState, ErrorState, PageHeader, SectionHeader } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/hooks/use-session";
import { ApiError } from "@/lib/api";
import { strategyLabel } from "@/lib/decisions";
import { fmtDay, humanize, isNotDeployed } from "@/lib/intel";
import type { MeIntelligence } from "@/lib/intel-types";

function Loading() {
  return (
    <div className="space-y-6" aria-busy="true" aria-label="Loading">
      <Skeleton className="h-48 w-full" />
      <Skeleton className="h-64 w-full" />
      <Skeleton className="h-40 w-full" />
    </div>
  );
}

function Report({ m }: { m: MeIntelligence }) {
  const strategy = m.strategy;
  const recs = [...m.recommendations].sort((a, b) => a.rank - b.rank);

  return (
    <div className="space-y-14">
      <section aria-labelledby="decision-h" id="decision" className="scroll-mt-24">
        <SectionHeader id="decision-h" index={1} kicker="the decision · recommendation strategy" title="How your list is being made" />
        <StrategyPanel d={strategy} />
      </section>

      <section aria-labelledby="drift-h">
        <SectionHeader id="drift-h" index={2} kicker="preference drift · historical against recent" title="Has your taste moved?" />
        <DriftReport drift={m.drift} />
      </section>

      <section aria-labelledby="history-h">
        <SectionHeader id="history-h" index={3} kicker="preference history · genre share per window" title="Where your viewing went" />
        {m.preference_history.windows.length ? (
          <PreferenceHistoryChart history={m.preference_history} />
        ) : (
          <EmptyState title="No windows yet" body="Your history is too short to split into time windows." />
        )}
      </section>

      <section aria-labelledby="recs-h">
        <SectionHeader
          id="recs-h"
          index={4}
          kicker={`downstream of the decision · ${strategy.abstained ? "standard (fallback)" : strategyLabel(strategy.answer)}`}
          title="What that decision recommends"
          action={<Link href="/recommendations" className="text-sm text-primary hover:underline">Full ranking →</Link>}
        />
        {recs.length ? (
          <RecommendationCards items={recs} strategyId={strategy.id} />
        ) : (
          <EmptyState title="No recommendations" body="The strategy decision produced no list for you yet." />
        )}
      </section>

      <section aria-labelledby="whatif-h">
        <SectionHeader id="whatif-h" index={5} kicker="scenarios · projected preferences" title="What if my taste keeps changing?" />
        <ScenarioPanel />
      </section>

      <section aria-labelledby="signals-h">
        <SectionHeader id="signals-h" index={6} kicker="your signals" title="What moved recently" />
        {!m.signals.length ? (
          <p className="text-sm text-muted-foreground">No signals about your activity in this run.</p>
        ) : (
          <ul className="divide-y hairline rounded-lg border bg-card">
            {m.signals.map((g) => (
              <li key={g.id} className="grid grid-cols-[auto_minmax(0,1fr)_96px] items-center gap-3 px-4 py-3">
                <DirectionIcon direction={g.direction} />
                <span className="min-w-0 text-sm">
                  <span className="block break-words">{g.title}</span>
                  <span className="block font-mono text-xs text-muted-foreground">{humanize(g.kind)} · {g.entity} · {fmtDay(g.observed_at)}</span>
                </span>
                <Meter value={g.strength} label={`Strength ${g.strength.toFixed(2)}`} />
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

export default function MyIntelligencePage() {
  const { user, loading } = useSession();
  const router = useRouter();
  const { data, error, mutate } = useSWR<MeIntelligence>(user ? "/me/intelligence" : null);

  useEffect(() => {
    if (!loading && !user) router.replace(`/login?next=${encodeURIComponent("/me/intelligence")}`);
  }, [loading, user, router]);

  return (
    <Container>
      <PageHeader eyebrow={data ? `My intelligence · as of ${fmtDay(data.as_of)}` : "My intelligence"} title="What JEV sees in your taste.">
        JEV tests whether your recent viewing differs from your history, decides how your recommendations should be produced, and shows the
        evidence for both. Confidence is always labelled with its kind; none of it is a guarantee.
      </PageHeader>

      {loading || (user && !data && !error) ? (
        <Loading />
      ) : !user ? null : error ? (
        isNotDeployed(error) ? (
          <NotDeployedState what="per-user intelligence (GET /me/intelligence)" since="v1.2" />
        ) : error instanceof ApiError && error.status === 404 ? (
          <EmptyState
            title="Nothing to analyse yet"
            body={`${error.message}. Rate or watch a few films and JEV can start comparing your history with your recent taste.`}
            action={<Button asChild variant="outline"><Link href="/discover">Find films to rate</Link></Button>}
          />
        ) : (
          <ErrorState error={error} retry={() => mutate()} />
        )
      ) : data && data.profile.n_events === 0 ? (
        <EmptyState
          title="No history yet"
          body="JEV needs your ratings and watches to test for drift and choose a strategy. Start with a few films you know."
          action={<Button asChild><Link href="/onboarding">Pick some films</Link></Button>}
        />
      ) : data ? (
        <>
          <dl className="mb-10 grid max-w-xl grid-cols-3 gap-4 border-t hairline pt-4">
            <div><dt className="eyebrow">Events</dt><dd className="num mt-1 text-xl">{data.profile.n_events.toLocaleString()}</dd></div>
            <div><dt className="eyebrow">First</dt><dd className="num mt-1 text-sm">{fmtDay(data.profile.first_event)}</dd></div>
            <div><dt className="eyebrow">Latest</dt><dd className="num mt-1 text-sm">{fmtDay(data.profile.last_event)}</dd></div>
          </dl>
          <Report m={data} />
        </>
      ) : null}
    </Container>
  );
}
