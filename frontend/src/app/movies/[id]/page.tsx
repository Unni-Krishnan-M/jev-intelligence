"use client";

import { Check, Eye, Heart } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import useSWR from "swr";

import { MovieCard } from "@/components/jev/movie-card";
import { Poster } from "@/components/jev/poster";
import { RatingStars } from "@/components/jev/rating-stars";
import { Shelf } from "@/components/jev/shelf";
import { Container, EmptyState, ErrorState, SectionHeader } from "@/components/jev/states";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useInteractions } from "@/hooks/use-interactions";
import { useSession } from "@/hooks/use-session";
import { ApiError } from "@/lib/api";
import type { MovieDetail, SimpleResponse } from "@/lib/types";

function Facts({ m }: { m: MovieDetail }) {
  const rows: [string, React.ReactNode][] = [
    ["Directed by", m.directors.length ? m.directors.join(", ") : "—"],
    ["Year", m.year ?? "—"],
    ["Runtime", m.runtime_min ? `${m.runtime_min} min` : "—"],
    ["Country", m.countries.length ? m.countries.join(", ") : "—"],
    ["Community", m.mean_rating ? <span className="num">★ {m.mean_rating.toFixed(2)} · {m.n_ratings} ratings</span> : "No ratings yet"],
  ];
  return (
    <dl className="divide-y hairline border-y hairline text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[120px_1fr] gap-4 py-2.5">
          <dt className="eyebrow pt-0.5">{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export default function MoviePage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const movieId = Number(id);
  const { user } = useSession();
  const { rate, unrate, favorite, watch } = useInteractions();
  const { data: m, error, mutate } = useSWR<MovieDetail>(Number.isFinite(movieId) ? `/movies/${movieId}` : null);
  const similar = useSWR<SimpleResponse>(Number.isFinite(movieId) ? `/recommendations/similar/${movieId}?limit=14` : null);

  if (error instanceof ApiError && error.status === 404) {
    return <Container className="pt-16"><EmptyState title="Film not found" body="It may have been removed from the catalogue." action={<Button asChild><Link href="/discover">Back to Discover</Link></Button>} /></Container>;
  }
  if (error) return <Container className="pt-16"><ErrorState error={error} retry={() => mutate()} /></Container>;

  const needLogin = () => router.push(`/login?next=/movies/${movieId}`);

  return (
    <Container className="pb-10 pt-10 sm:pt-14">
      <div className="grid gap-10 md:grid-cols-[minmax(0,300px)_1fr] lg:gap-14">
        <div className="mx-auto w-full max-w-[300px] md:mx-0">
          {m ? <Poster movie={m} priority /> : <Skeleton className="aspect-[2/3] w-full" />}
        </div>
        <div className="min-w-0">
          {m ? (
            <>
              <p className="eyebrow">{m.genres.join(" / ") || "Film"}</p>
              <h1 className="font-display mt-2 text-[46px] leading-[0.95] tracking-tight text-balance sm:text-[68px]">{m.title}</h1>
              {m.description && <p className="mt-4 max-w-2xl text-[16px] leading-relaxed text-ink-2 first-letter:uppercase">{m.description}.</p>}

              <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-4 rounded-lg border bg-card/60 p-4">
                <div>
                  <p className="eyebrow mb-1.5">Your rating</p>
                  <RatingStars value={m.user_rating} onChange={(v) => (user ? void rate(m.id, v) : needLogin())} onClear={() => user && void unrate(m.id)} />
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button variant={m.is_favorite ? "default" : "outline"} onClick={() => (user ? void favorite(m.id, !m.is_favorite) : needLogin())} aria-pressed={m.is_favorite}>
                    <Heart className="size-4" fill={m.is_favorite ? "currentColor" : "none"} /> {m.is_favorite ? "Favourite" : "Add to favourites"}
                  </Button>
                  <Button variant="outline" onClick={() => (user ? void watch(m.id) : needLogin())}>
                    {m.watched ? <Check className="size-4" /> : <Eye className="size-4" />} {m.watched ? "Watched · log again" : "Mark as watched"}
                  </Button>
                  {m.user_rating !== null && <Button variant="ghost" onClick={() => void unrate(m.id)}>Clear rating</Button>}
                </div>
                {!m.in_model && (
                  <p className="basis-full text-xs text-muted-foreground">
                    This film was added after the current model was trained. Its similar titles below come from metadata only.
                  </p>
                )}
              </div>

              <div className="mt-8 grid gap-8 lg:grid-cols-2">
                <Facts m={m} />
                <div className="space-y-5">
                  {m.cast.length > 0 && (
                    <div>
                      <p className="eyebrow mb-2">Cast (from Wikidata)</p>
                      <p className="text-sm leading-relaxed text-ink-2">{m.cast.slice(0, 14).join(" · ")}{m.cast.length > 14 ? " …" : ""}</p>
                    </div>
                  )}
                  {(m.keywords.length > 0 || m.tags.length > 0) && (
                    <div>
                      <p className="eyebrow mb-2">Themes & tags</p>
                      <div className="flex flex-wrap gap-1.5">
                        {m.keywords.slice(0, 8).map((k) => <Badge key={`k-${k}`} variant="secondary" className="font-normal">{k}</Badge>)}
                        {m.tags.slice(0, 10).map((t) => <Badge key={`t-${t}`} variant="outline" className="font-normal text-muted-foreground">{t}</Badge>)}
                      </div>
                    </div>
                  )}
                  {m.imdb_id && (
                    <a className="inline-block font-mono text-xs text-muted-foreground underline-offset-4 hover:text-foreground hover:underline"
                      href={`https://www.imdb.com/title/${m.imdb_id}/`} target="_blank" rel="noopener noreferrer">IMDb {m.imdb_id} ↗</a>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="space-y-4"><Skeleton className="h-4 w-32" /><Skeleton className="h-16 w-3/4" /><Skeleton className="h-24 w-full" /></div>
          )}
        </div>
      </div>

      <section className="mt-20" aria-labelledby="more">
        <SectionHeader id="more" kicker="Content + co-watch similarity" title="More like this" />
        {similar.error ? <ErrorState error={similar.error} /> : similar.data && similar.data.items.length === 0 ? (
          <EmptyState title="No close matches" body="JEV couldn't find films that share enough with this one." />
        ) : (
          <Shelf label="More like this" loading={!similar.data}>
            {similar.data?.items.map((s) => <div role="listitem" key={s.movie_id}><MovieCard movie={s} /></div>)}
          </Shelf>
        )}
      </section>
    </Container>
  );
}
