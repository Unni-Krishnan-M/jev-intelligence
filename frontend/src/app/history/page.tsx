"use client";

import Link from "next/link";
import useSWR from "swr";

import { Poster } from "@/components/jev/poster";
import { RatingStars } from "@/components/jev/rating-stars";
import { Container, EmptyState, ErrorState, PageHeader } from "@/components/jev/states";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { MovieBrief } from "@/lib/types";

interface Paged<T> { items: T[]; total: number; page: number }
type Watch = { id: number; movie: MovieBrief; watched_at: string; rating: number | null };
type RatingRow = { movie: MovieBrief; rating: number; rated_at: string };
type RecRow = { id: number; movie_id: number; title: string; model_version: string; context: string; rank: number; score: number; reason: string; created_at: string; feedback: string | null };

const when = (iso: string) => new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

function MovieLine({ m, right, sub }: { m: MovieBrief; right?: React.ReactNode; sub?: React.ReactNode }) {
  return (
    <li className="flex items-center gap-4 border-b hairline py-3">
      <Link href={`/movies/${m.id}`} className="w-11 shrink-0"><Poster movie={m} /></Link>
      <div className="min-w-0 flex-1">
        <Link href={`/movies/${m.id}`} className="truncate text-[15px] hover:text-primary">{m.title}</Link>
        <p className="font-mono text-[11px] text-muted-foreground">{[m.year, m.directors[0]].filter(Boolean).join(" · ")}{sub ? <> · {sub}</> : null}</p>
      </div>
      {right}
    </li>
  );
}

function Loading() { return <div className="space-y-3">{Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-14 w-full" />)}</div>; }

export default function HistoryPage() {
  const watches = useSWR<Paged<Watch>>("/users/me/history?page_size=100");
  const ratings = useSWR<Paged<RatingRow>>("/users/me/ratings?page_size=200");
  const recs = useSWR<RecRow[]>("/recommendations/history?limit=100");

  return (
    <Container>
      <PageHeader eyebrow="History" title="Your ledger.">Everything you&apos;ve watched and rated, and every recommendation JEV served you, with the reason it gave at the time.</PageHeader>
      <Tabs defaultValue="watched">
        <TabsList>
          <TabsTrigger value="watched">Watched{watches.data ? ` · ${watches.data.total}` : ""}</TabsTrigger>
          <TabsTrigger value="rated">Rated{ratings.data ? ` · ${ratings.data.total}` : ""}</TabsTrigger>
          <TabsTrigger value="served">Recommended{recs.data ? ` · ${recs.data.length}` : ""}</TabsTrigger>
        </TabsList>
        <TabsContent value="watched" className="mt-6">
          {watches.error ? <ErrorState error={watches.error} /> : !watches.data ? <Loading /> : watches.data.items.length === 0 ? (
            <EmptyState title="Nothing watched yet" body="Use “Mark as watched” on any film page." />
          ) : (
            <ul>{watches.data.items.map((w) => (
              <MovieLine key={w.id} m={w.movie} sub={when(w.watched_at)} right={w.rating ? <RatingStars value={w.rating} readOnly size={14} /> : null} />
            ))}</ul>
          )}
        </TabsContent>
        <TabsContent value="rated" className="mt-6">
          {ratings.error ? <ErrorState error={ratings.error} /> : !ratings.data ? <Loading /> : ratings.data.items.length === 0 ? (
            <EmptyState title="No ratings yet" body="Ratings are the strongest signal JEV has." />
          ) : (
            <ul>{ratings.data.items.map((r) => (
              <MovieLine key={r.movie.id} m={r.movie} sub={when(r.rated_at)} right={<RatingStars value={r.rating} readOnly size={14} />} />
            ))}</ul>
          )}
        </TabsContent>
        <TabsContent value="served" className="mt-6">
          {recs.error ? <ErrorState error={recs.error} /> : !recs.data ? <Loading /> : recs.data.length === 0 ? (
            <EmptyState title="No recommendations served yet" body="Open Tonight or For you to get your first list." />
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[720px] text-sm">
                <thead>
                  <tr className="border-b hairline text-left text-xs text-muted-foreground">
                    <th className="py-2 font-normal">When</th><th className="py-2 font-normal">#</th><th className="py-2 font-normal">Film</th>
                    <th className="py-2 font-normal">Reason given</th><th className="py-2 text-right font-normal">Score</th><th className="py-2 pl-4 font-normal">Feedback</th>
                  </tr>
                </thead>
                <tbody>
                  {recs.data.map((r) => (
                    <tr key={r.id} className="border-b hairline align-top">
                      <td className="whitespace-nowrap py-2 pr-4 font-mono text-[11px] text-muted-foreground">{when(r.created_at)}<br />{r.context}</td>
                      <td className="num py-2 pr-3">{r.rank}</td>
                      <td className="py-2 pr-4"><Link href={`/movies/${r.movie_id}`} className="hover:text-primary">{r.title}</Link></td>
                      <td className="py-2 pr-4 text-ink-2">{r.reason}</td>
                      <td className="num py-2 text-right">{r.score.toFixed(3)}</td>
                      <td className="py-2 pl-4 text-xs text-muted-foreground">{r.feedback?.replace("_", " ") ?? "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </Container>
  );
}
