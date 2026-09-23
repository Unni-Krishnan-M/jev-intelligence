"use client";

import { EyeOff, ThumbsUp } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useInteractions } from "@/hooks/use-interactions";
import { dominantSignal, SIGNAL_META } from "@/lib/signals";
import type { MovieBrief, RecItem, SimpleItem } from "@/lib/types";
import { cn } from "@/lib/utils";

import { Poster } from "./poster";
import { WhyDialog } from "./why-dialog";

type CardMovie = { id: number; title: string; year: number | null; genres: string[]; directors: string[] };

function meta(m: CardMovie) {
  return [m.year, m.directors[0]].filter(Boolean).join(" · ");
}

/** Explanation badge: the dominant signal's colour dot + the model-derived reason. */
export function ReasonBadge({ item, className }: { item: RecItem; className?: string }) {
  const sig = dominantSignal(item.signals);
  return (
    <p className={cn("flex items-start gap-1.5 text-[12.5px] leading-snug text-ink-2", className)}>
      <span className="mt-[5px] size-1.5 shrink-0 rounded-full" style={{ background: SIGNAL_META[sig].color }} aria-hidden />
      <span className="line-clamp-2">{item.reason}</span>
    </p>
  );
}

export function RecCard({ item, modelVersion, onHide, className }: { item: RecItem; modelVersion?: string; onHide?: (id: number) => void; className?: string }) {
  const { feedback } = useInteractions();
  const movie: CardMovie = { id: item.movie_id, title: item.title, year: item.year, genres: item.genres, directors: item.directors };
  return (
    <article className={cn("group flex flex-col", className)}>
      <Link
        href={`/movies/${item.movie_id}`}
        className="block rounded-[6px] transition-transform duration-200 ease-out group-hover:-translate-y-0.5"
        onClick={() => void feedback(item.movie_id, "clicked", item.recommendation_id)}
      >
        <Poster movie={movie} />
      </Link>
      <div className="mt-2.5 flex items-baseline justify-between gap-2">
        <Link href={`/movies/${item.movie_id}`} className="min-w-0 truncate text-[14px] font-medium hover:text-primary">{item.title}</Link>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="num shrink-0 text-[11px] text-muted-foreground" tabIndex={0}>{item.score.toFixed(2)}</span>
          </TooltipTrigger>
          <TooltipContent>Hybrid score (0–1) for this request</TooltipContent>
        </Tooltip>
      </div>
      <p className="truncate font-mono text-[11px] text-muted-foreground">{meta(movie)}</p>
      <ReasonBadge item={item} className="mt-2" />
      <div className="mt-1.5 flex items-center gap-0.5 opacity-100 transition-opacity md:opacity-60 md:group-hover:opacity-100 md:focus-within:opacity-100">
        <WhyDialog item={item} modelVersion={modelVersion} />
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" className="size-7 text-muted-foreground" aria-label={`More like ${item.title}`}
              onClick={() => void feedback(item.movie_id, "like", item.recommendation_id)}>
              <ThumbsUp className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Good pick</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button variant="ghost" size="icon" className="size-7 text-muted-foreground" aria-label={`Not interested in ${item.title}`}
              onClick={async () => { onHide?.(item.movie_id); await feedback(item.movie_id, "not_interested", item.recommendation_id); }}>
              <EyeOff className="size-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>Not for me: never suggest it again</TooltipContent>
        </Tooltip>
      </div>
    </article>
  );
}

export function MovieCard({ movie, reason, className }: { movie: MovieBrief | SimpleItem; reason?: string | null; className?: string }) {
  const m: CardMovie = "id" in movie
    ? movie
    : { id: movie.movie_id, title: movie.title, year: movie.year, genres: movie.genres, directors: movie.directors };
  const why = reason ?? ("reason" in movie ? movie.reason : null);
  return (
    <article className={cn("group flex flex-col", className)}>
      <Link href={`/movies/${m.id}`} className="block rounded-[6px] transition-transform duration-200 ease-out group-hover:-translate-y-0.5">
        <Poster movie={m} />
      </Link>
      <Link href={`/movies/${m.id}`} className="mt-2.5 truncate text-[14px] font-medium hover:text-primary">{m.title}</Link>
      <p className="truncate font-mono text-[11px] text-muted-foreground">{meta(m)}</p>
      {why && <p className="mt-1.5 line-clamp-2 text-[12.5px] leading-snug text-ink-2">{why}</p>}
    </article>
  );
}
