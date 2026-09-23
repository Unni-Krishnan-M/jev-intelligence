"use client";

import Link from "next/link";
import useSWR from "swr";

import { MovieCard } from "@/components/jev/movie-card";
import { Container, EmptyState, ErrorState, PageHeader, PosterSkeleton } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { useInteractions } from "@/hooks/use-interactions";
import type { MovieBrief } from "@/lib/types";

type Fav = { movie: MovieBrief; source: "user" | "onboarding"; added_at: string };

export default function FavoritesPage() {
  const { data, error, mutate } = useSWR<{ items: Fav[]; total: number }>("/users/me/favorites?page_size=200");
  const { favorite } = useInteractions();
  return (
    <Container>
      <PageHeader eyebrow="Library" title="Favourites.">
        {data ? `${data.total} film${data.total === 1 ? "" : "s"}. Each one is a strong positive signal in every model.` : "Films you'd watch again."}
      </PageHeader>
      {error && <ErrorState error={error} retry={() => mutate()} />}
      {data && data.items.length === 0 && (
        <EmptyState title="No favourites yet" body="Tap the heart on a film page to keep it here." action={<Button asChild><Link href="/discover">Discover films</Link></Button>} />
      )}
      <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
        {!data && !error && Array.from({ length: 12 }, (_, i) => <PosterSkeleton key={i} />)}
        {data?.items.map((f) => (
          <div key={f.movie.id}>
            <MovieCard movie={f.movie} reason={f.source === "onboarding" ? "Picked during onboarding" : null} />
            <Button variant="ghost" size="sm" className="mt-1 h-7 px-0 text-xs text-muted-foreground hover:text-foreground"
              onClick={async () => { await favorite(f.movie.id, false); await mutate(); }}>Remove</Button>
          </div>
        ))}
      </div>
    </Container>
  );
}
