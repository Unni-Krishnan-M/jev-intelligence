"use client";

import { ArrowRight } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import useSWR from "swr";

import { MovieCard, ReasonBadge, RecCard } from "@/components/jev/movie-card";
import { Poster } from "@/components/jev/poster";
import { Shelf } from "@/components/jev/shelf";
import { WeightBar } from "@/components/jev/signal-bar";
import { Container, EmptyState, ErrorState, SectionHeader } from "@/components/jev/states";
import { WhyDialog } from "@/components/jev/why-dialog";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/hooks/use-session";
import type { MovieBrief, RecItem, RecResponse, SimpleResponse, TasteProfile } from "@/lib/types";

function greeting(): string {
  const h = new Date().getHours();
  if (h < 5) return "Late show";
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

const STAGE_COPY = {
  cold: "No history yet: this list is built from your genres and what's popular.",
  warming: "Still learning: collaborative and latent-factor signals ramp up as you rate.",
  warm: "Fully personalised: every signal has enough of your history to work with.",
};

function Feature({ item, modelVersion }: { item: RecItem; modelVersion: string }) {
  return (
    <article className="grid gap-6 sm:grid-cols-[minmax(0,240px)_1fr] sm:gap-8">
      <Link href={`/movies/${item.movie_id}`} className="block max-w-[240px]">
        <Poster movie={{ id: item.movie_id, title: item.title, year: item.year, genres: item.genres, directors: item.directors }} priority />
      </Link>
      <div className="flex flex-col justify-end">
        <p className="eyebrow">Tonight&apos;s first pick · score <span className="num">{item.score.toFixed(2)}</span></p>
        <Link href={`/movies/${item.movie_id}`} className="font-display mt-2 text-[44px] leading-[0.95] hover:text-primary sm:text-[56px]">
          {item.title}
        </Link>
        <p className="mt-2 font-mono text-xs text-muted-foreground">
          {[item.year, item.directors[0], item.genres.slice(0, 3).join(" / ")].filter(Boolean).join(" · ")}
        </p>
        <ReasonBadge item={item} className="mt-5 text-[15px]" />
        {item.secondary_reasons.length > 0 && (
          <ul className="mt-2 space-y-1 pl-3 text-sm text-muted-foreground">
            {item.secondary_reasons.map((r) => <li key={r}>— {r}</li>)}
          </ul>
        )}
        <div className="mt-5 flex flex-wrap gap-2">
          <Button asChild><Link href={`/movies/${item.movie_id}`}>Open film <ArrowRight className="size-4" /></Link></Button>
          <WhyDialog item={item} modelVersion={modelVersion} trigger={<Button variant="outline">Why this pick?</Button>} />
        </div>
      </div>
    </article>
  );
}

function SimpleShelf({ data, error, label, empty }: { data?: SimpleResponse; error?: unknown; label: string; empty: React.ReactNode }) {
  if (error) return <ErrorState error={error} />;
  if (data && data.items.length === 0) return <>{empty}</>;
  return (
    <Shelf label={label} loading={!data}>
      {data?.items.map((m) => <div role="listitem" key={m.movie_id}><MovieCard movie={m} /></div>)}
    </Shelf>
  );
}

export default function HomePage() {
  const router = useRouter();
  const { user, loading } = useSession();
  const [hidden, setHidden] = useState<number[]>([]);

  useEffect(() => {
    if (!loading && user && !user.onboarding_completed) router.replace("/onboarding");
    if (!loading && !user) router.replace("/login?next=/home");
  }, [loading, user, router]);

  const enabled = Boolean(user);
  const recs = useSWR<RecResponse>(enabled ? "/recommendations?limit=24&context=home" : null);
  const profile = useSWR<TasteProfile>(enabled ? "/users/me/profile" : null);
  const byw = useSWR<SimpleResponse>(enabled ? "/recommendations/because-you-watched?limit=14" : null);
  const favs = useSWR<SimpleResponse>(enabled ? "/recommendations/similar-to-favorites?limit=14" : null);
  const trending = useSWR<SimpleResponse>("/recommendations/trending?limit=16");
  const popular = useSWR<MovieBrief[]>("/movies/popular?limit=16");
  const discover = useSWR<RecResponse>(enabled ? "/recommendations?limit=14&max_ratings=15&context=discovery" : null);

  const items = (recs.data?.items ?? []).filter((i) => !hidden.includes(i.movie_id));
  const [first, ...rest] = items;
  const p = profile.data;

  return (
    <Container className="pb-10">
      <header className="grid gap-6 pb-10 pt-10 sm:pt-14 lg:grid-cols-[1.4fr_1fr] lg:items-end">
        <div>
          <p className="eyebrow">{new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" })}</p>
          <h1 className="font-display mt-2 text-[48px] leading-[0.95] tracking-tight sm:text-[68px]">
            {greeting()}, <em className="text-primary">{user?.display_name ?? "…"}</em>.
          </h1>
          {p?.stage && <p className="mt-3 max-w-xl text-[15px] text-ink-2">{STAGE_COPY[p.stage]}</p>}
        </div>
        <div className="rounded-lg border bg-card/60 p-4">
          <div className="flex items-baseline justify-between gap-3">
            <p className="eyebrow">Your blend right now</p>
            <Link href="/profile" className="text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline">Taste profile</Link>
          </div>
          {recs.data ? (
            <>
              <WeightBar weights={recs.data.effective_weights} className="mt-3" />
              <p className="mt-3 font-mono text-[11px] text-muted-foreground">
                {recs.data.profile.interactions} interactions · {recs.data.profile.genres} genres · {recs.data.profile.excluded} hidden
                {p?.behavioral_ramp ? ` · full personalisation at ${p.behavioral_ramp}` : ""}
              </p>
            </>
          ) : <Skeleton className="mt-3 h-12 w-full" />}
        </div>
      </header>

      <section aria-labelledby="s1" className="mb-16">
        <SectionHeader index={1} id="s1" kicker="Hybrid ranking" title="Recommended for you"
          action={<Button variant="link" className="px-0 text-primary" asChild><Link href="/recommendations">Full ranked list <ArrowRight className="size-4" /></Link></Button>} />
        {recs.error && <ErrorState error={recs.error} retry={() => recs.mutate()} />}
        {!recs.data && !recs.error && (
          <div className="grid gap-6 sm:grid-cols-[240px_1fr]"><Skeleton className="aspect-[2/3] w-full max-w-[240px]" /><Skeleton className="h-40 w-full self-end" /></div>
        )}
        {recs.data && items.length === 0 && (
          <EmptyState title="Nothing to programme yet" body="Pick genres or rate a few films and your list appears here." action={<Button asChild><Link href="/onboarding">Set up my taste</Link></Button>} />
        )}
        {first && recs.data && (
          <>
            <Feature item={first} modelVersion={recs.data.model_version} />
            <div className="mt-12">
              <Shelf label="More recommendations">
                {rest.map((it) => (
                  <div role="listitem" key={it.movie_id}>
                    <RecCard item={it} modelVersion={recs.data?.model_version} onHide={(id) => setHidden((h) => [...h, id])} />
                  </div>
                ))}
              </Shelf>
            </div>
          </>
        )}
      </section>

      <section aria-labelledby="s2" className="mb-16">
        <SectionHeader index={2} id="s2" kicker="Item-to-item"
          title={byw.data?.anchor ? <>Because you {byw.data.anchor_kind === "rated" ? "loved" : "watched"} <em>{byw.data.anchor.title}</em></> : "Because you watched"} />
        <SimpleShelf data={byw.data} error={byw.error} label="Because you watched"
          empty={<EmptyState title="Watch or love something first" body="Mark a film as watched, or rate it 4★ or higher, and its closest neighbours show up here." />} />
      </section>

      <section aria-labelledby="s3" className="mb-16">
        <SectionHeader index={3} id="s3" kicker="From your favourites" title="Similar to your favourites" />
        <SimpleShelf data={favs.data} error={favs.error} label="Similar to your favourites"
          empty={<EmptyState title="No favourites yet" body="Tap the heart on any film to add it to your favourites." action={<Button variant="outline" asChild><Link href="/discover">Discover films</Link></Button>} />} />
      </section>

      <section aria-labelledby="s4" className="mb-16">
        <SectionHeader index={4} id="s4" kicker="Time-decayed activity" title="Trending" />
        <SimpleShelf data={trending.data} error={trending.error} label="Trending" empty={<EmptyState title="Nothing trending" />} />
      </section>

      <section aria-labelledby="s5" className="mb-16">
        <SectionHeader index={5} id="s5" kicker="Bayesian-average rating" title="Popular & acclaimed" />
        {popular.error ? <ErrorState error={popular.error} /> : (
          <Shelf label="Popular" loading={!popular.data}>
            {popular.data?.map((m) => <div role="listitem" key={m.id}><MovieCard movie={m} reason={m.mean_rating ? `★ ${m.mean_rating.toFixed(2)} from ${m.n_ratings} ratings` : null} /></div>)}
          </Shelf>
        )}
      </section>

      <section aria-labelledby="s6">
        <SectionHeader index={6} id="s6" kicker="Fewer than 15 ratings" title="New discoveries" />
        {discover.error ? <ErrorState error={discover.error} /> : discover.data && discover.data.items.length === 0 ? (
          <EmptyState title="No discoveries yet" body="As JEV learns your taste, lesser-known films that fit it appear here." />
        ) : (
          <Shelf label="New discoveries" loading={!discover.data}>
            {discover.data?.items.map((it) => (
              <div role="listitem" key={it.movie_id}><RecCard item={it} modelVersion={discover.data?.model_version} /></div>
            ))}
          </Shelf>
        )}
      </section>
    </Container>
  );
}
