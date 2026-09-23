"use client";

import { EyeOff, ThumbsUp } from "lucide-react";
import Link from "next/link";
import { useState } from "react";
import useSWR from "swr";
import useSWRInfinite from "swr/infinite";
import { toast } from "sonner";

import { Poster } from "@/components/jev/poster";
import { SignalBar } from "@/components/jev/signal-bar";
import { Container, EmptyState, ErrorState, PageHeader } from "@/components/jev/states";
import { WhyDialog } from "@/components/jev/why-dialog";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useInteractions } from "@/hooks/use-interactions";
import { api, errorMessage, qs } from "@/lib/api";
import { SIGNAL_META, SIGNALS } from "@/lib/signals";
import type { Genre, RecItem, RecResponse, TasteProfile } from "@/lib/types";
import { cn } from "@/lib/utils";

const PAGE = 20;

function Row({ item, modelVersion, onHide }: { item: RecItem; modelVersion: string; onHide: (id: number) => void }) {
  const { feedback } = useInteractions();
  return (
    <li className="grid grid-cols-[2.5rem_64px_1fr] items-start gap-4 border-b hairline py-5 sm:grid-cols-[3.5rem_76px_1fr_220px] sm:gap-6">
      <span className="font-display pt-1 text-right text-[34px] leading-none text-muted-foreground sm:text-[44px]">{item.rank}</span>
      <Link href={`/movies/${item.movie_id}`} onClick={() => void feedback(item.movie_id, "clicked", item.recommendation_id)}>
        <Poster movie={{ id: item.movie_id, title: item.title, year: item.year, genres: item.genres, directors: item.directors }} />
      </Link>
      <div className="min-w-0">
        <Link href={`/movies/${item.movie_id}`} className="font-display text-[26px] leading-tight hover:text-primary"
          onClick={() => void feedback(item.movie_id, "clicked", item.recommendation_id)}>{item.title}</Link>
        <p className="font-mono text-[11px] text-muted-foreground">{[item.year, item.directors[0], item.genres.slice(0, 3).join(" / ")].filter(Boolean).join(" · ")}</p>
        <p className="mt-2 text-[14px]">{item.reason}</p>
        {item.secondary_reasons.length > 0 && <p className="mt-0.5 text-[13px] text-muted-foreground">{item.secondary_reasons.join(" · ")}</p>}
        <div className="mt-2 flex flex-wrap items-center gap-0.5">
          <WhyDialog item={item} modelVersion={modelVersion} />
          <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs text-muted-foreground" onClick={() => void feedback(item.movie_id, "like", item.recommendation_id)}>
            <ThumbsUp className="size-3.5" /> Good pick
          </Button>
          <Button variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs text-muted-foreground"
            onClick={async () => { onHide(item.movie_id); await feedback(item.movie_id, "not_interested", item.recommendation_id); }}>
            <EyeOff className="size-3.5" /> Not for me
          </Button>
        </div>
      </div>
      <div className="col-span-3 sm:col-span-1 sm:pt-2">
        <div className="flex items-baseline justify-between">
          <span className="eyebrow">score</span>
          <span className="num text-sm">{item.score.toFixed(3)}</span>
        </div>
        <SignalBar signals={item.signals} className="mt-2" height={7} />
      </div>
    </li>
  );
}

export default function RecommendationsPage() {
  const [genres, setGenres] = useState<string[]>([]);
  const [discovery, setDiscovery] = useState(false);
  const [hidden, setHidden] = useState<number[]>([]);
  const { data: allGenres } = useSWR<Genre[]>("/genres");
  const profile = useSWR<TasteProfile>("/users/me/profile");
  const diversity = profile.data?.preferences?.diversity ?? "balanced";

  const getKey = (i: number, prev: RecResponse | null) => {
    if (prev && prev.items.length < PAGE) return null;
    return `/recommendations${qs({ limit: PAGE, offset: i * PAGE, genres, max_ratings: discovery ? 15 : undefined, context: "ranked_list" })}`;
  };
  const { data, error, size, setSize, isLoading, isValidating, mutate } = useSWRInfinite<RecResponse>(getKey, { revalidateFirstPage: false });
  const items = (data ?? []).flatMap((d) => d.items).filter((i) => !hidden.includes(i.movie_id));
  const last = data?.[data.length - 1];
  const more = Boolean(last && last.items.length === PAGE && last.offset + PAGE < 500);

  async function setDiversity(v: string) {
    if (!v) return;
    try {
      await api("/users/me/preferences", { method: "PATCH", json: { diversity: v } });
      await profile.mutate();
      await mutate();
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  return (
    <Container>
      <PageHeader eyebrow="For you · full ranking" title="Your programme, ranked.">
        Every film is scored by six signals and the blend is re-ranked for variety. Open <em>Why this?</em> on any row to see the exact
        numbers.
      </PageHeader>

      <div className="mb-6 grid gap-5 rounded-lg border bg-card/60 p-4 lg:grid-cols-[1fr_auto]">
        <div>
          <p className="eyebrow mb-2">Only these genres</p>
          <div className="flex flex-wrap gap-1.5">
            {(allGenres ?? []).filter((g) => g.movie_count >= 30).map((g) => {
              const on = genres.includes(g.name);
              return (
                <button key={g.name} type="button" aria-pressed={on}
                  onClick={() => { setHidden([]); setGenres((cur) => (on ? cur.filter((x) => x !== g.name) : [...cur, g.name])); }}
                  className={cn("rounded-full border px-3 py-1 text-[13px] transition-colors", on ? "border-primary bg-primary text-primary-foreground" : "text-ink-2 hover:border-foreground/40")}>
                  {g.name}
                </button>
              );
            })}
          </div>
        </div>
        <div className="flex flex-wrap items-end gap-6">
          <div>
            <p className="eyebrow mb-2">Variety</p>
            <ToggleGroup type="single" value={diversity} onValueChange={setDiversity} variant="outline" size="sm" aria-label="Recommendation variety">
              <ToggleGroupItem value="focused">Focused</ToggleGroupItem>
              <ToggleGroupItem value="balanced">Balanced</ToggleGroupItem>
              <ToggleGroupItem value="adventurous">Adventurous</ToggleGroupItem>
            </ToggleGroup>
          </div>
          <label className="flex items-center gap-2 pb-1 text-sm">
            <Switch checked={discovery} onCheckedChange={(v) => { setHidden([]); setDiscovery(v); }} /> Hidden gems only
          </label>
        </div>
      </div>

      <ul className="mb-2 flex flex-wrap gap-x-4 gap-y-1" aria-label="Signal legend">
        {SIGNALS.map((s) => (
          <li key={s} className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <span className="size-2 rounded-full" style={{ background: SIGNAL_META[s].color }} aria-hidden /> {SIGNAL_META[s].label}
          </li>
        ))}
      </ul>

      {error && <ErrorState error={error} retry={() => mutate()} />}
      {isLoading && !data && <div className="space-y-4 pt-4">{Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-28 w-full" />)}</div>}
      {data && items.length === 0 && <EmptyState className="mt-6" title="No films match" body="Loosen the genre filter or turn off hidden gems." />}
      <ol className="border-t hairline">
        {items.map((it) => <Row key={`${it.movie_id}-${it.rank}`} item={it} modelVersion={it ? (data?.[0]?.model_version ?? "") : ""} onHide={(id) => setHidden((h) => [...h, id])} />)}
      </ol>
      {more && (
        <div className="mt-8 flex justify-center">
          <Button variant="outline" size="lg" disabled={isValidating} onClick={() => setSize(size + 1)}>{isValidating ? "Loading…" : "Show 20 more"}</Button>
        </div>
      )}
      {data?.[0] && (
        <p className="mt-8 font-mono text-[11px] text-muted-foreground">
          model {data[0].model_version} · generated {new Date(data[0].generated_at).toLocaleTimeString()} {data[0].cached ? "· served from cache" : ""}
        </p>
      )}
    </Container>
  );
}
