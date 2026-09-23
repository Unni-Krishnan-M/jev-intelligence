"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import useSWR from "swr";

import { MovieCard } from "@/components/jev/movie-card";
import { Container, EmptyState, ErrorState, PageHeader, PosterSkeleton } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { qs } from "@/lib/api";
import type { Genre, MovieBrief, MoviePage } from "@/lib/types";

const SORTS = [
  { v: "popular", l: "Most rated" },
  { v: "rating", l: "Highest rated" },
  { v: "year", l: "Newest" },
  { v: "title", l: "A–Z" },
];
const DECADES = ["all", "2010", "2000", "1990", "1980", "1970", "1960", "1950", "1900"];

function DiscoverInner() {
  const router = useRouter();
  const sp = useSearchParams();
  const q = sp.get("q") ?? "";
  const genre = sp.get("genre") ?? "all";
  const sort = sp.get("sort") ?? "popular";
  const decade = sp.get("decade") ?? "all";
  const page = Math.max(1, Number(sp.get("page") ?? 1));

  const set = (patch: Record<string, string | number | null>) => {
    const next = new URLSearchParams(sp.toString());
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === "" || v === "all") next.delete(k);
      else next.set(k, String(v));
    }
    if (!("page" in patch)) next.delete("page");
    router.replace(`/discover${next.toString() ? `?${next}` : ""}`, { scroll: false });
  };

  const { data: genres } = useSWR<Genre[]>("/genres");
  const yearMin = decade === "all" ? undefined : Number(decade);
  const yearMax = decade === "all" ? undefined : decade === "1900" ? 1949 : Number(decade) + 9;
  const listKey = q
    ? `/movies/search${qs({ q, limit: 48 })}`
    : `/movies${qs({ page, page_size: 36, genre: genre === "all" ? undefined : genre, sort, year_min: yearMin, year_max: yearMax })}`;
  const { data, error, isLoading, mutate } = useSWR<MoviePage | MovieBrief[]>(listKey);
  const items = Array.isArray(data) ? data : data?.items;
  const total = Array.isArray(data) ? data.length : data?.total;
  const pages = !Array.isArray(data) && data ? Math.ceil(data.total / data.page_size) : 1;

  return (
    <Container>
      <PageHeader eyebrow="Discover" title="The whole catalogue.">
        {total !== undefined ? <>{total.toLocaleString()} films{q ? <> matching “{q}”</> : genre !== "all" ? <> in {genre}</> : null}.</> : "Browse, filter, search."}
      </PageHeader>

      <form className="sticky top-14 z-30 -mx-5 mb-8 flex flex-wrap items-center gap-2 border-y hairline bg-background/90 px-5 py-3 backdrop-blur sm:-mx-8 sm:px-8"
        onSubmit={(e) => { e.preventDefault(); set({ q: String(new FormData(e.currentTarget).get("q") ?? "").trim() || null }); }} role="search">
        <Input key={q} name="q" defaultValue={q} placeholder="Search titles, directors, cast" className="h-9 w-full sm:w-72" aria-label="Search" />
        <Select value={genre} onValueChange={(v) => set({ genre: v, q: null })} disabled={Boolean(q)}>
          <SelectTrigger className="h-9 w-[150px]" aria-label="Genre"><SelectValue placeholder="Genre" /></SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All genres</SelectItem>
            {genres?.map((g) => <SelectItem key={g.name} value={g.name}>{g.name}</SelectItem>)}
          </SelectContent>
        </Select>
        <Select value={decade} onValueChange={(v) => set({ decade: v, q: null })} disabled={Boolean(q)}>
          <SelectTrigger className="h-9 w-[130px]" aria-label="Decade"><SelectValue /></SelectTrigger>
          <SelectContent>
            {DECADES.map((d) => <SelectItem key={d} value={d}>{d === "all" ? "Any year" : d === "1900" ? "Before 1950" : `${d}s`}</SelectItem>)}
          </SelectContent>
        </Select>
        <Select value={sort} onValueChange={(v) => set({ sort: v, q: null })} disabled={Boolean(q)}>
          <SelectTrigger className="h-9 w-[150px]" aria-label="Sort"><SelectValue /></SelectTrigger>
          <SelectContent>{SORTS.map((s) => <SelectItem key={s.v} value={s.v}>{s.l}</SelectItem>)}</SelectContent>
        </Select>
        {q && <Button type="button" variant="ghost" size="sm" onClick={() => set({ q: null })}>Clear search</Button>}
      </form>

      {error && <ErrorState error={error} retry={() => mutate()} />}
      {items && items.length === 0 && <EmptyState title="No films found" body="Try a different spelling, or loosen the filters." />}
      <div className="grid grid-cols-2 gap-x-4 gap-y-8 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6">
        {isLoading && !items && Array.from({ length: 18 }, (_, i) => <PosterSkeleton key={i} />)}
        {items?.map((m) => <MovieCard key={m.id} movie={m} reason={m.mean_rating ? `★ ${m.mean_rating.toFixed(1)} · ${m.n_ratings} ratings` : null} />)}
      </div>

      {!q && pages > 1 && (
        <nav className="mt-12 flex items-center justify-between border-t hairline pt-5" aria-label="Pagination">
          <Button variant="outline" disabled={page <= 1} onClick={() => set({ page: page - 1 })}>Previous</Button>
          <p className="num text-sm text-muted-foreground">page {page} / {pages}</p>
          <Button variant="outline" disabled={page >= pages} onClick={() => set({ page: page + 1 })}>Next</Button>
        </nav>
      )}
    </Container>
  );
}

export default function DiscoverPage() {
  return <Suspense><DiscoverInner /></Suspense>;
}
