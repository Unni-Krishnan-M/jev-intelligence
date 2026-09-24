"use client";

import { ArrowRight, ArrowUp, CheckCircle2, CircleSlash } from "lucide-react";
import Link from "next/link";
import useSWR from "swr";

import { fmtDate } from "@/components/jev/admin/ui";
import { Poster } from "@/components/jev/poster";
import { Container } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/hooks/use-session";
import { intelHref } from "@/lib/domain";
import { humanize, isNotDeployed } from "@/lib/intel";
import type { DomainInfo, DomainList } from "@/lib/intel-types";
import type { ActiveSummary, SimpleResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

/** The engine's running order (docs/platform.md §1), grouped into four acts. */
const ACTS: { label: string; span: string; stages: { title: string; body: string; key?: boolean }[] }[] = [
  {
    label: "Observe",
    span: "lg:col-span-3",
    stages: [
      { title: "Data", body: "A domain adapter loads time-stamped observations; freshness and quality are checked first." },
      { title: "Signals", body: "Each change worth a look, with its strength, source and age." },
      { title: "Trends & anomalies", body: "Theil–Sen slopes with q-values, change points and robust-z breaks." },
    ],
  },
  {
    label: "Anticipate",
    span: "lg:col-span-2",
    stages: [
      { title: "Forecasts", body: "Damped-trend models with 80 % intervals, backtested on rolling origins." },
      { title: "Risk", body: "Likelihood × impact, shrunk by confidence and data quality." },
    ],
  },
  {
    label: "Decide",
    span: "lg:col-span-2",
    stages: [
      { title: "JEV decisions", body: "Fixed questions, versioned policies and a declared kind of confidence. It abstains when the evidence is thin.", key: true },
      { title: "Warnings & recommendations", body: "Raised only downstream of a decision, and each carries the decision id." },
    ],
  },
  {
    label: "Act & learn",
    span: "lg:col-span-2",
    stages: [
      { title: "Actions", body: "Ranked next steps, each with its expected impact, effort and first move." },
      { title: "Feedback", body: "Verdicts and outcomes are fed into evaluation, and evaluation checks every stage." },
    ],
  },
];

/** Stage numbers run across the acts: 01–09. */
const ACT_START = ACTS.map((_, i) => ACTS.slice(0, i).reduce((sum, a) => sum + a.stages.length, 0));

function Pipeline() {
  return (
    <figure aria-labelledby="pipeline-caption">
      <ol className="grid grid-cols-1 gap-x-4 lg:grid-cols-9">
        {ACTS.map((act, ai) => (
          <li key={act.label} className={cn("grid grid-cols-1 gap-x-4", act.span, act.stages.length === 3 ? "lg:grid-cols-3" : "lg:grid-cols-2")}>
            <p className="eyebrow border-t hairline pb-3 pt-2 lg:col-span-full">{act.label}</p>
            <ol className="contents">
              {act.stages.map((s, si) => {
                const n = ACT_START[ai] + si + 1;
                const last = n === 9;
                return (
                  <li key={s.title} className="relative flex gap-4 pb-6 lg:block lg:pb-0">
                    <div className="flex flex-col items-center lg:flex-row">
                      <span
                        className={cn(
                          "num grid size-8 shrink-0 place-items-center rounded-full border text-xs",
                          s.key ? "border-primary bg-primary text-primary-foreground" : "hairline bg-background",
                        )}
                        aria-hidden
                      >
                        {String(n).padStart(2, "0")}
                      </span>
                      {!last && <span className="w-px flex-1 bg-rule lg:h-px lg:w-auto" aria-hidden />}
                    </div>
                    <div className="min-w-0 lg:mt-3 lg:pr-2">
                      <p className={cn("font-display text-[22px] leading-tight hyphens-auto", s.key && "text-primary")}>{s.title}</p>
                      <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{s.body}</p>
                    </div>
                  </li>
                );
              })}
            </ol>
          </li>
        ))}
      </ol>
      {/* the loop: feedback returns to the start (desktop draws it; phones get the sentence) */}
      <div className="relative mx-[calc(100%/18)] mt-6 hidden h-5 rounded-b border-x border-b border-dashed border-muted-foreground/50 lg:block" aria-hidden>
        <ArrowUp className="absolute -left-[8.5px] -top-3 size-4 bg-background text-muted-foreground" />
        <span className="absolute left-1/2 top-full -translate-x-1/2 -translate-y-1/2 bg-background px-3 font-mono text-[11px] text-muted-foreground">
          feedback → evaluation → the next run
        </span>
      </div>
      <p className="-mt-2 flex items-center gap-2 font-mono text-[11px] text-muted-foreground lg:hidden">
        <ArrowUp className="size-3.5" aria-hidden /> 09 feeds back into 01: feedback → evaluation → the next run
      </p>
      <figcaption id="pipeline-caption" className="mt-8 max-w-3xl text-sm leading-relaxed text-muted-foreground">
        Every stage writes its evidence, so a warning can be traced back through the decision, the risk and the forecast to the rows it came
        from. Nothing past the as-of date is visible to a run, so any month can be replayed without leakage.
      </figcaption>
    </figure>
  );
}

// ---- domains ---------------------------------------------------------------------------------

const STATIC_DOMAINS: { key: string; title: string; kicker: string; body: string }[] = [
  {
    key: "movie",
    title: "Movies",
    kicker: "first domain adapter",
    body: "MovieLens ratings plus the app's own activity. It adds genre demand, audience lapse, rater behaviour and model governance, and gives each member the drift test and strategy decision behind their recommendations.",
  },
  {
    key: "generic:us-unemployment",
    title: "US unemployment",
    kicker: "generic dataset adapter",
    body: "Monthly unemployment rates for the US and a set of states (BLS via FRED, public domain), loaded from a CSV file and a config file. The engine is the same and none of the movie code runs.",
  },
];

function kickerFor(d: Pick<DomainInfo, "key">): string {
  if (d.key === "movie") return "first domain adapter";
  if (d.key.startsWith("generic:")) return "generic dataset adapter";
  return "domain adapter";
}

function MovieAppLink({ signedIn }: { signedIn: boolean }) {
  return (
    <Link href={signedIn ? "/home" : "/discover"} className="inline-flex items-center gap-1 text-sm text-primary hover:underline">
      Open the movie app <ArrowRight className="size-3.5" aria-hidden />
    </Link>
  );
}

function LiveDomainCard({ d, signedIn }: { d: DomainInfo; signedIn: boolean }) {
  const run = d.latest_run;
  return (
    <article className="flex flex-col rounded-lg border bg-card p-5">
      <p className="eyebrow">{kickerFor(d)} · <span className="font-mono normal-case tracking-normal">{d.key}</span></p>
      <h3 className="font-display mt-2 text-[32px] leading-none">{d.name}</h3>
      <p className="mt-3 text-sm leading-relaxed text-ink-2">{d.description}</p>
      <p className="mt-3 inline-flex items-center gap-1.5 text-sm">
        {d.available ? (
          <><CheckCircle2 className="size-4" style={{ color: "var(--status-good)" }} aria-hidden /> Available</>
        ) : (
          <><CircleSlash className="size-4 text-muted-foreground" aria-hidden /> Unavailable{d.reason ? <span className="text-muted-foreground"> — {d.reason}</span> : null}</>
        )}
      </p>
      <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t hairline pt-4 text-sm">
        <div><dt className="eyebrow">Frequency</dt><dd className="mt-0.5">{d.frequency}</dd></div>
        <div><dt className="eyebrow">Open warnings</dt><dd className="num mt-0.5">{d.warnings_open.toLocaleString()}</dd></div>
        <div><dt className="eyebrow">Latest run</dt><dd className="num mt-0.5 text-xs">{run ? `${run.status} · as of ${fmtDate(run.as_of).slice(0, 10)}` : "never run"}</dd></div>
        <div><dt className="eyebrow">Entities</dt><dd className="mt-0.5 text-xs">{d.entity_types.map(humanize).join(", ") || "—"}</dd></div>
      </dl>
      {d.sources.length > 0 && (
        <p className="mt-3 font-mono text-[11px] leading-relaxed text-muted-foreground">
          {d.sources.map((s) => `${s.source}${s.license ? ` (${s.license})` : ""}`).join(" · ")}
        </p>
      )}
      <div className="mt-auto flex flex-wrap gap-x-5 gap-y-2 pt-5">
        <Link href={intelHref("/intel", d.key)} className="inline-flex items-center gap-1 text-sm text-primary hover:underline">
          Console for {d.name} <ArrowRight className="size-3.5" aria-hidden />
        </Link>
        {d.key === "movie" && <MovieAppLink signedIn={signedIn} />}
      </div>
    </article>
  );
}

function StaticDomainCard({ d, signedIn }: { d: (typeof STATIC_DOMAINS)[number]; signedIn: boolean }) {
  return (
    <article className="flex flex-col rounded-lg border bg-card p-5">
      <p className="eyebrow">{d.kicker}</p>
      <h3 className="font-display mt-2 text-[32px] leading-none">{d.title}</h3>
      <p className="mt-3 text-sm leading-relaxed text-ink-2">{d.body}</p>
      <div className="mt-auto pt-5">
        {d.key === "movie" ? <MovieAppLink signedIn={signedIn} /> : <p className="text-xs text-muted-foreground">Visible in the Intelligence console to administrators.</p>}
      </div>
    </article>
  );
}

function Domains() {
  const { user } = useSession();
  const admin = Boolean(user?.is_admin);
  const { data, error } = useSWR<DomainList>(admin ? "/intel/domains" : null);
  const live = admin && data?.items.length ? data.items : null;

  return (
    <div>
      {admin && !data && !error ? (
        <div className="grid gap-4 md:grid-cols-2"><Skeleton className="h-72" /><Skeleton className="h-72" /></div>
      ) : live ? (
        <div className="grid gap-4 md:grid-cols-2">{live.map((d) => <LiveDomainCard key={d.key} d={d} signedIn={Boolean(user)} />)}</div>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2">{STATIC_DOMAINS.map((d) => <StaticDomainCard key={d.key} d={d} signedIn={Boolean(user)} />)}</div>
          {admin && error && (
            <p className="mt-3 text-xs text-muted-foreground">
              {isNotDeployed(error) ? "Live domain status (GET /intel/domains) is not available on this API yet." : "Live domain status could not be loaded."}
            </p>
          )}
        </>
      )}
    </div>
  );
}

// ---- the movie domain, measured ---------------------------------------------------------------

const MODEL_ROWS: { key: string; label: string; what: string }[] = [
  { key: "hybrid", label: "Hybrid", what: "all six signals, adaptive weights, MMR diversity" },
  { key: "itemknn", label: "Item-kNN", what: "collaborative filtering on co-watches" },
  { key: "als", label: "Implicit ALS", what: "matrix factorization on implicit feedback" },
  { key: "popularity", label: "Popularity", what: "non-personalised baseline" },
  { key: "content", label: "Content TF-IDF", what: "metadata only: genres, people, themes" },
  { key: "random", label: "Random", what: "sanity floor" },
];

function MoviesMeasured() {
  const { data: summary, error: summaryError } = useSWR<ActiveSummary>("/models/active/summary");
  const { data: trending } = useSWR<SimpleResponse>("/recommendations/trending?limit=7");
  const best = summary ? Math.max(...Object.values(summary.comparison).map((m) => m["ndcg@10"] ?? 0)) : 1;

  return (
    <div className="grid gap-12 lg:grid-cols-[1fr_1.3fr]">
      <div>
        <p className="max-w-md text-[15px] leading-relaxed text-ink-2">
          The movie adapter serves a hybrid recommender. Every model in the serving version was evaluated the same way: for each user the latest
          ratings were held out, the whole catalogue was ranked, and we counted how many of the films they went on to rate 4★ or higher landed in
          the top 10.
        </p>
        {summary && (
          <dl className="mt-6 grid max-w-md grid-cols-3 gap-4 border-t hairline pt-4">
            <div><dt className="eyebrow">Films</dt><dd className="num mt-1 text-xl">{summary.n_items.toLocaleString()}</dd></div>
            <div><dt className="eyebrow">Ratings learned</dt><dd className="num mt-1 text-xl">{summary.trained_on_rows?.toLocaleString()}</dd></div>
            <div><dt className="eyebrow">Held-out users</dt><dd className="num mt-1 text-xl">{summary.eval_users ?? "—"}</dd></div>
          </dl>
        )}
        {summary && (
          <p className="mt-4 font-mono text-xs text-muted-foreground">
            model {summary.model_version}<br />split {summary.split} · dataset {summary.dataset_version}
          </p>
        )}
        <div className="mt-8 grid max-w-md grid-cols-3 gap-3">
          {(trending?.items ?? Array.from({ length: 3 }, () => null)).slice(0, 3).map((m, i) =>
            m ? (
              <Link key={m.movie_id} href={`/movies/${m.movie_id}`} className="block transition-transform hover:-translate-y-1">
                <Poster movie={{ id: m.movie_id, title: m.title, year: m.year, genres: m.genres, directors: m.directors }} />
              </Link>
            ) : (
              <Skeleton key={i} className="aspect-[2/3] w-full rounded-[6px]" />
            ),
          )}
        </div>
        <p className="eyebrow mt-3">Trending now · covers are typeset from each film&apos;s metadata</p>
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
  );
}

export default function Landing() {
  const { user } = useSession();

  return (
    <>
      <Container className="pb-14 pt-12 sm:pt-20">
        <p className="eyebrow">JEV · intelligent decision &amp; early-warning engine</p>
        <h1 className="font-display mt-4 max-w-5xl text-[48px] leading-[0.92] tracking-tight text-balance sm:text-[72px] xl:text-[84px]">
          Changing data in. <em className="text-primary">Decisions and early warnings out.</em>
        </h1>
        <p className="mt-6 max-w-2xl text-[16px] leading-relaxed text-ink-2">
          JEV reads a domain&apos;s time-stamped data and finds what is moving. It forecasts where that may lead and scores the risk, then asks
          fixed questions answered by versioned policies. Warnings, recommendations and actions all come downstream of those decisions, and each one
          says how confident it is, what kind of confidence that is, and which evidence it rests on.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          {user?.is_admin ? (
            <Button size="lg" asChild><Link href="/intel">Open the Intelligence console <ArrowRight className="size-4" /></Link></Button>
          ) : user ? (
            <Button size="lg" asChild><Link href="/me/intelligence">See my intelligence <ArrowRight className="size-4" /></Link></Button>
          ) : (
            <Button size="lg" asChild><Link href="/register">Create an account <ArrowRight className="size-4" /></Link></Button>
          )}
          <Button size="lg" variant="outline" asChild><Link href={user ? "/home" : "/discover"}>{user ? "Tonight's programme" : "Browse the films"}</Link></Button>
        </div>
      </Container>

      <Container className="py-12">
        <div className="border-t hairline pt-10">
          <p className="eyebrow">The running order</p>
          <h2 className="font-display mt-2 max-w-3xl text-4xl leading-tight sm:text-5xl">From raw rows to a decision someone can act on.</h2>
          <div className="mt-10"><Pipeline /></div>
        </div>
      </Container>

      <Container className="py-12">
        <div className="border-t hairline pt-10">
          <p className="eyebrow">Domains</p>
          <h2 className="font-display mt-2 max-w-3xl text-4xl leading-tight sm:text-5xl">One engine, one adapter per domain.</h2>
          <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-ink-2">
            Everything domain-specific sits behind an adapter. It loads the data, declares which series to build and can add stages of its own. The
            core engine never imports movie code.
          </p>
          <div className="mt-8"><Domains /></div>
        </div>
      </Container>

      <Container className="py-12">
        <div className="border-t hairline pt-10">
          <p className="eyebrow">Movies domain · measured, not claimed</p>
          <h2 className="font-display mt-2 mb-8 max-w-3xl text-4xl leading-tight sm:text-5xl">The recommender behind the movie app.</h2>
          <MoviesMeasured />
        </div>
      </Container>
    </>
  );
}
