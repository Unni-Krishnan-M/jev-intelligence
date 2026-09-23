"use client";

import { Check } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import useSWR from "swr";
import { toast } from "sonner";

import { Poster } from "@/components/jev/poster";
import { RatingStars } from "@/components/jev/rating-stars";
import { Container, ErrorState, PosterSkeleton } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { useSession } from "@/hooks/use-session";
import { api, errorMessage, qs } from "@/lib/api";
import type { Genre, MovieBrief, MoviePage, User } from "@/lib/types";
import { cn } from "@/lib/utils";

const STEPS = ["Genres", "Favourites", "Quick ratings"] as const;

export default function Onboarding() {
  const router = useRouter();
  const { user, refresh } = useSession();
  const [step, setStep] = useState(0);
  // null until the user touches the picker: then the saved genres (if any) are the starting point
  const [genreSel, setGenreSel] = useState<string[] | null>(null);
  const saved = user?.favorite_genres;
  const genres = useMemo(() => genreSel ?? saved ?? [], [genreSel, saved]);
  const setGenres = (fn: (cur: string[]) => string[]) => setGenreSel(fn(genres));
  const [picked, setPicked] = useState<number[]>([]);
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const [saving, setSaving] = useState(false);

  const { data: allGenres, error: genreError, mutate: retryGenres } = useSWR<Genre[]>("/genres");
  const { data: pool, error: poolError } = useSWR<MoviePage>(`/movies${qs({ sort: "popular", page_size: 100 })}`);

  const choices = useMemo(() => {
    const items = pool?.items ?? [];
    if (!genres.length) return items.slice(0, 30);
    const scored = items.map((m) => ({ m, s: m.genres.filter((g) => genres.includes(g)).length }));
    return scored.filter((x) => x.s > 0).sort((a, b) => b.s - a.s || b.m.n_ratings - a.m.n_ratings).slice(0, 30).map((x) => x.m);
  }, [pool, genres]);

  const toRate = useMemo(
    () => (pool?.items ?? []).filter((m) => !picked.includes(m.id) && !choices.slice(0, 12).some((c) => c.id === m.id)).slice(0, 12),
    [pool, picked, choices],
  );

  async function finish() {
    setSaving(true);
    try {
      const u = await api<User>("/users/me/onboarding", { json: { genres, movie_ids: picked } });
      await refresh(u, { revalidate: false });
      router.replace("/home");
    } catch (e) {
      toast.error(errorMessage(e));
      setSaving(false);
    }
  }

  async function rate(m: MovieBrief, v: number) {
    setRatings((r) => ({ ...r, [m.id]: v }));
    try {
      await api(`/movies/${m.id}/rate`, { json: { rating: v } });
    } catch (e) {
      toast.error(errorMessage(e));
    }
  }

  const visibleGenres = (allGenres ?? []).filter((g) => g.name !== "IMAX" && g.movie_count >= 30);

  return (
    <Container className="max-w-[1100px] pb-24 pt-10 sm:pt-14">
      <ol className="flex items-center gap-3 font-mono text-[11px] uppercase tracking-[0.14em]" aria-label="Onboarding steps">
        {STEPS.map((s, i) => (
          <li key={s} className={cn("flex items-center gap-2", i === step ? "text-foreground" : "text-muted-foreground")}>
            <span className={cn("grid size-5 place-items-center rounded-full border text-[10px]", i < step && "border-primary bg-primary text-primary-foreground", i === step && "border-primary")}>
              {i < step ? <Check className="size-3" /> : i + 1}
            </span>
            {s}
            {i < STEPS.length - 1 && <span className="mx-1 h-px w-6 bg-border" aria-hidden />}
          </li>
        ))}
      </ol>

      {step === 0 && (
        <section className="mt-10">
          <h1 className="font-display text-[44px] leading-[0.95] sm:text-[60px]">What do you like to watch?</h1>
          <p className="mt-3 max-w-xl text-ink-2">Pick a few genres. Until you&apos;ve rated some films, JEV uses these together with popularity.</p>
          {genreError && <ErrorState className="mt-6" error={genreError} retry={() => retryGenres()} />}
          <div className="mt-8 flex flex-wrap gap-2.5">
            {visibleGenres.map((g) => {
              const on = genres.includes(g.name);
              return (
                <button key={g.name} type="button" aria-pressed={on}
                  onClick={() => setGenres((cur) => (on ? cur.filter((x) => x !== g.name) : [...cur, g.name]))}
                  className={cn("rounded-full border px-4 py-2 text-sm transition-colors", on ? "border-primary bg-primary text-primary-foreground" : "hover:border-foreground/40")}>
                  {g.name}
                  <span className={cn("num ml-2 text-[11px]", on ? "text-primary-foreground/70" : "text-muted-foreground")}>{g.movie_count}</span>
                </button>
              );
            })}
          </div>
          <div className="mt-10 flex items-center gap-4">
            <Button size="lg" disabled={genres.length === 0} onClick={() => setStep(1)}>Continue</Button>
            <p className="text-sm text-muted-foreground">{genres.length ? `${genres.length} selected` : "Choose at least one"}</p>
          </div>
        </section>
      )}

      {step === 1 && (
        <section className="mt-10">
          <h1 className="font-display text-[44px] leading-[0.95] sm:text-[60px]">Films you love</h1>
          <p className="mt-3 max-w-xl text-ink-2">Tap any you&apos;d happily watch again. Each counts as a strong positive signal. Pick three or more for the best start.</p>
          {poolError && <ErrorState className="mt-6" error={poolError} />}
          <div className="mt-8 grid grid-cols-3 gap-4 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6">
            {!pool && Array.from({ length: 12 }, (_, i) => <PosterSkeleton key={i} />)}
            {choices.map((m) => {
              const on = picked.includes(m.id);
              return (
                <button key={m.id} type="button" aria-pressed={on} aria-label={`${on ? "Unselect" : "Select"} ${m.title}`}
                  onClick={() => setPicked((cur) => (on ? cur.filter((x) => x !== m.id) : [...cur, m.id]))}
                  className="group relative text-left">
                  <Poster movie={m} className={cn("transition-all", on ? "ring-2 ring-primary ring-offset-2 ring-offset-background" : "group-hover:opacity-90")} />
                  {on && <span className="absolute right-2 top-2 z-[3] grid size-6 place-items-center rounded-full bg-primary text-primary-foreground"><Check className="size-3.5" /></span>}
                  <p className="mt-2 truncate text-[13px]">{m.title}</p>
                </button>
              );
            })}
          </div>
          <div className="mt-10 flex flex-wrap items-center gap-4">
            <Button variant="outline" size="lg" onClick={() => setStep(0)}>Back</Button>
            <Button size="lg" onClick={() => setStep(2)}>{picked.length ? "Continue" : "Skip"}</Button>
            <p className="text-sm text-muted-foreground">{picked.length} picked</p>
          </div>
        </section>
      )}

      {step === 2 && (
        <section className="mt-10">
          <h1 className="font-display text-[44px] leading-[0.95] sm:text-[60px]">Seen any of these?</h1>
          <p className="mt-3 max-w-xl text-ink-2">
            Rate a few, low scores included. Ratings feed the collaborative and latent-factor models, and a 2★ is as useful as a 5★.
          </p>
          <div className="mt-8 grid gap-x-8 gap-y-4 sm:grid-cols-2">
            {toRate.map((m) => (
              <div key={m.id} className="flex items-center gap-4 border-b hairline pb-4">
                <Poster movie={m} className="w-14 shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{m.title}</p>
                  <p className="font-mono text-[11px] text-muted-foreground">{[m.year, m.directors[0]].filter(Boolean).join(" · ")}</p>
                  <div className="mt-1.5"><RatingStars value={ratings[m.id] ?? null} onChange={(v) => void rate(m, v)} size={20} /></div>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-10 flex flex-wrap items-center gap-4">
            <Button variant="outline" size="lg" onClick={() => setStep(1)}>Back</Button>
            <Button size="lg" onClick={finish} disabled={saving}>{saving ? "Building your programme…" : "See my programme"}</Button>
            <p className="text-sm text-muted-foreground">{Object.keys(ratings).length} rated · {picked.length} favourites · {genres.length} genres</p>
          </div>
        </section>
      )}
    </Container>
  );
}
