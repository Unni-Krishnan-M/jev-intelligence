"use client";

import { ArrowRight } from "lucide-react";
import Link from "next/link";
import useSWR from "swr";

import { Poster } from "@/components/jev/poster";
import { Container } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/hooks/use-session";
import { SIGNAL_META, SIGNALS } from "@/lib/signals";
import type { ActiveSummary, SimpleResponse } from "@/lib/types";

const MODEL_ROWS: { key: string; label: string; what: string }[] = [
  { key: "hybrid", label: "Hybrid", what: "all six signals, adaptive weights, MMR diversity" },
  { key: "itemknn", label: "Item-kNN", what: "collaborative filtering on co-watches" },
  { key: "als", label: "Implicit ALS", what: "matrix factorization on implicit feedback" },
  { key: "popularity", label: "Popularity", what: "non-personalised baseline" },
  { key: "content", label: "Content TF-IDF", what: "metadata only: genres, people, themes" },
  { key: "random", label: "Random", what: "sanity floor" },
];

export default function Landing() {
  const { user } = useSession();
  const { data: summary, error: summaryError } = useSWR<ActiveSummary>("/models/active/summary");
  const { data: trending } = useSWR<SimpleResponse>("/recommendations/trending?limit=7");
  const best = summary ? Math.max(...Object.values(summary.comparison).map((m) => m["ndcg@10"] ?? 0)) : 1;

  return (
    <>
      <Container className="grid gap-12 pb-16 pt-12 sm:pt-20 lg:grid-cols-[1.1fr_1fr] lg:gap-16">
        <div className="flex flex-col justify-center">
          <p className="eyebrow">No. 01 — a film programme that learns you</p>
          <h1 className="font-display mt-4 text-[52px] leading-[0.92] tracking-tight text-balance sm:text-[76px] xl:text-[88px]">
            Tell us what you love. <em className="text-primary">We&apos;ll programme the rest.</em>
          </h1>
          <p className="mt-6 max-w-xl text-[16px] leading-relaxed text-ink-2">
            JEV blends five recommenders into one ranked list: content similarity, collaborative filtering, matrix
            factorization, popularity and your own genre choices. Every pick shows its working, so you can see which signal put a
            film on your list and how much it counted.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            {user ? (
              <Button size="lg" asChild><Link href="/home">Open tonight&apos;s programme <ArrowRight className="size-4" /></Link></Button>
            ) : (
              <Button size="lg" asChild><Link href="/register">Start with three films <ArrowRight className="size-4" /></Link></Button>
            )}
            <Button size="lg" variant="outline" asChild><Link href="/discover">Browse the catalogue</Link></Button>
          </div>
          {summary && (
            <dl className="mt-10 grid max-w-lg grid-cols-3 gap-4 border-t hairline pt-5">
              <div><dt className="eyebrow">Films</dt><dd className="num mt-1 text-xl">{summary.n_items.toLocaleString()}</dd></div>
              <div><dt className="eyebrow">Ratings learned</dt><dd className="num mt-1 text-xl">{summary.trained_on_rows?.toLocaleString()}</dd></div>
              <div><dt className="eyebrow">Held-out users</dt><dd className="num mt-1 text-xl">{summary.eval_users ?? "—"}</dd></div>
            </dl>
          )}
        </div>

        {/* a programme plate of what's trending: real data, typeset covers */}
        <div className="relative">
          <div className="grid grid-cols-3 gap-3 sm:gap-4">
            {(trending?.items ?? Array.from({ length: 6 }, () => null)).slice(0, 6).map((m, i) => (
              <div key={m ? m.movie_id : i} className={i % 3 === 1 ? "translate-y-8" : ""}>
                {m ? (
                  <Link href={`/movies/${m.movie_id}`} className="block transition-transform hover:-translate-y-1">
                    <Poster movie={{ id: m.movie_id, title: m.title, year: m.year, genres: m.genres, directors: m.directors }} />
                  </Link>
                ) : (
                  <Skeleton className="aspect-[2/3] w-full rounded-[6px]" />
                )}
              </div>
            ))}
          </div>
          <p className="eyebrow mt-12 text-right">Trending now · covers are typeset from each film&apos;s metadata</p>
        </div>
      </Container>

      <Container className="py-12">
        <div className="grid gap-12 border-t hairline pt-10 lg:grid-cols-[1fr_1.3fr]">
          <div>
            <p className="eyebrow">The working, shown</p>
            <h2 className="font-display mt-2 text-4xl leading-tight sm:text-5xl">Measured, not claimed.</h2>
            <p className="mt-4 max-w-md text-[15px] leading-relaxed text-ink-2">
              Every model in the serving version was evaluated the same way. For each user we held out their latest
              ratings, ranked the whole catalogue, and counted how many of the films they went on to rate 4★ or higher
              landed in the top 10.
            </p>
            {summary && (
              <p className="mt-4 font-mono text-xs text-muted-foreground">
                model {summary.model_version}<br />split {summary.split} · dataset {summary.dataset_version}
              </p>
            )}
          </div>
          <div>
            {summaryError && <p className="text-sm text-muted-foreground">Metrics appear once a model is trained and the API is running.</p>}
            {!summary && !summaryError && <Skeleton className="h-64 w-full" />}
            {summary && (
              <table className="w-full text-sm">
                <caption className="sr-only">NDCG@10 and Recall@10 per model on the held-out test split</caption>
                <thead>
                  <tr className="border-b hairline text-left text-xs text-muted-foreground">
                    <th className="py-2 font-normal">Model</th>
                    <th className="py-2 font-normal">NDCG@10</th>
                    <th className="py-2 text-right font-normal">Recall@10</th>
                  </tr>
                </thead>
                <tbody>
                  {MODEL_ROWS.filter((r) => summary.comparison[r.key]).map((r) => {
                    const m = summary.comparison[r.key];
                    const v = m["ndcg@10"] ?? 0;
                    return (
                      <tr key={r.key} className="border-b hairline">
                        <td className="py-3 pr-4">
                          <p className={r.key === "hybrid" ? "font-medium" : ""}>{r.label}</p>
                          <p className="text-xs text-muted-foreground">{r.what}</p>
                        </td>
                        <td className="w-[45%] py-3 pr-4">
                          <div className="flex items-center gap-3">
                            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                              <div className="h-full rounded-full" style={{ width: `${(v / best) * 100}%`, background: r.key === "hybrid" ? "var(--primary)" : "var(--muted-foreground)", opacity: r.key === "hybrid" ? 1 : 0.55 }} />
                            </div>
                            <span className="num w-12 text-right">{v.toFixed(3)}</span>
                          </div>
                        </td>
                        <td className="num py-3 text-right">{(m["recall@10"] ?? 0).toFixed(3)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            )}
          </div>
        </div>
      </Container>

      <Container className="py-12">
        <div className="border-t hairline pt-10">
          <p className="eyebrow">How a pick is made</p>
          <h2 className="font-display mt-2 max-w-3xl text-4xl leading-tight sm:text-5xl">Six signals, one ranked list.</h2>
          <ol className="mt-10 grid gap-x-10 gap-y-8 sm:grid-cols-2 lg:grid-cols-3">
            {SIGNALS.map((s, i) => (
              <li key={s} className="border-t hairline pt-4">
                <p className="flex items-center gap-2 font-mono text-xs text-muted-foreground">
                  <span className="size-2 rounded-full" style={{ background: SIGNAL_META[s].color }} aria-hidden />
                  {String(i + 1).padStart(2, "0")} · {SIGNAL_META[s].model}
                </p>
                <p className="font-display mt-2 text-2xl">{SIGNAL_META[s].label}</p>
                <p className="mt-1 text-sm leading-relaxed text-ink-2">{SIGNAL_META[s].blurb}</p>
              </li>
            ))}
          </ol>
          <p className="mt-10 max-w-3xl text-sm leading-relaxed text-muted-foreground">
            With no history, JEV leans on popularity and the genres you pick. As you rate and watch, collaborative and
            latent-factor weights ramp up. A diversity pass (maximal marginal relevance) keeps one director from taking over the
            page, and anything you mark &ldquo;not for me&rdquo; is never shown again.
          </p>
        </div>
      </Container>
    </>
  );
}
