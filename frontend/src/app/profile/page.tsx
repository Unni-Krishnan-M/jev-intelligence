"use client";

import Link from "next/link";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import useSWR from "swr";
import { toast } from "sonner";

import { ChartTooltip } from "@/components/jev/admin/charts";
import { WeightBar } from "@/components/jev/signal-bar";
import { Container, EmptyState, ErrorState, PageHeader, SectionHeader } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { api, errorMessage } from "@/lib/api";
import type { Genre, TasteProfile } from "@/lib/types";
import { cn } from "@/lib/utils";

const TICK = { fill: "var(--muted-foreground)", fontSize: 11, fontFamily: "var(--font-plex-mono)" };

function GenreEditor({ profile, onSaved }: { profile: TasteProfile; onSaved: () => void }) {
  const { data: genres } = useSWR<Genre[]>("/genres");
  const [sel, setSel] = useState<string[]>(profile.explicit_genres);
  const [saving, setSaving] = useState(false);
  const dirty = sel.slice().sort().join() !== profile.explicit_genres.slice().sort().join();
  return (
    <div>
      <div className="flex flex-wrap gap-1.5">
        {(genres ?? []).filter((g) => g.movie_count >= 30).map((g) => {
          const on = sel.includes(g.name);
          return (
            <button key={g.name} type="button" aria-pressed={on} onClick={() => setSel((c) => (on ? c.filter((x) => x !== g.name) : [...c, g.name]))}
              className={cn("rounded-full border px-3 py-1 text-[13px]", on ? "border-primary bg-primary text-primary-foreground" : "text-ink-2 hover:border-foreground/40")}>
              {g.name}
            </button>
          );
        })}
      </div>
      <Button className="mt-4" size="sm" disabled={!dirty || sel.length === 0 || saving} onClick={async () => {
        setSaving(true);
        try { await api("/users/me/preferences", { method: "PATCH", json: { genres: sel } }); toast.success("Genres updated"); onSaved(); }
        catch (e) { toast.error(errorMessage(e)); }
        finally { setSaving(false); }
      }}>Save genres</Button>
    </div>
  );
}

export default function ProfilePage() {
  const { data: p, error, mutate } = useSWR<TasteProfile>("/users/me/profile");

  if (error) return <Container className="pt-16"><ErrorState error={error} retry={() => mutate()} /></Container>;
  if (!p) return <Container className="pt-16 space-y-4"><Skeleton className="h-16 w-2/3" /><Skeleton className="h-64 w-full" /></Container>;

  const affinity = p.genre_affinity.slice(0, 10);
  const totalRated = p.rating_histogram.reduce((s, r) => s + r.count, 0);

  return (
    <Container>
      <PageHeader eyebrow={`Taste profile · ${p.stage ?? "—"}`} title={<>What JEV knows about <em className="text-primary">{p.user.display_name}</em>.</>}>
        Everything here is derived from your own ratings, favourites, watches and genre picks. Nothing is inferred from elsewhere.
      </PageHeader>

      <div className="grid gap-4 sm:grid-cols-4">
        {[
          ["Ratings", p.counts.ratings],
          ["Favourites", p.counts.favorites],
          ["Watches", p.counts.watches],
          ["Hidden picks", p.counts.negative_feedback],
        ].map(([k, v]) => (
          <div key={k} className="rounded-lg border bg-card/60 p-4">
            <p className="eyebrow">{k}</p>
            <p className="font-display mt-1 text-4xl">{v}</p>
          </div>
        ))}
      </div>

      {p.effective_weights && (
        <section className="mt-14">
          <SectionHeader kicker="Hybrid weights for you, right now" title="Your blend" />
          <WeightBar weights={p.effective_weights} />
          <p className="mt-3 max-w-2xl text-sm text-muted-foreground">
            With little history, popularity and your genres carry the ranking. Collaborative and latent-factor weights reach full
            strength after {p.behavioral_ramp ?? "—"} interactions; you have {p.counts.ratings + p.counts.favorites + p.counts.watches}.
          </p>
        </section>
      )}

      <div className="mt-14 grid gap-12 lg:grid-cols-2">
        <section>
          <SectionHeader kicker="Genres of films you liked (4★+ or favourite)" title="Genre affinity" />
          {affinity.length === 0 ? <EmptyState title="No likes yet" body="Rate films 4★ or higher, or add favourites." /> : (
            <>
              <ResponsiveContainer width="100%" height={Math.max(160, affinity.length * 30)}>
                <BarChart data={affinity} layout="vertical" margin={{ top: 0, right: 24, bottom: 0, left: 0 }} barCategoryGap={4}>
                  <CartesianGrid horizontal={false} stroke="var(--rule)" />
                  <XAxis type="number" tick={TICK} axisLine={false} tickLine={false} allowDecimals={false} />
                  <YAxis type="category" dataKey="genre" tick={{ ...TICK, fill: "var(--ink-2)" }} axisLine={false} tickLine={false} width={96} />
                  <Tooltip cursor={{ fill: "var(--accent)", opacity: 0.4 }} content={({ active, payload, label }) => (
                    <ChartTooltip active={active} label={label} digits={0}
                      payload={Array.isArray(payload) ? payload.map((x) => ({ name: "liked films", value: x.value as number, color: "var(--sig-preference)" })) : undefined} />
                  )} />
                  <Bar dataKey="count" fill="var(--sig-preference)" radius={[0, 4, 4, 0]} maxBarSize={18} isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
              <p className="sr-only">{affinity.map((a) => `${a.genre}: ${a.count}`).join(", ")}</p>
            </>
          )}
        </section>

        <section>
          <SectionHeader kicker={`${totalRated} ratings${p.mean_rating ? ` · mean ${p.mean_rating.toFixed(2)}★` : ""}`} title="How you rate" />
          {totalRated === 0 ? <EmptyState title="No ratings yet" body="Your rating distribution appears here." /> : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={p.rating_histogram} margin={{ top: 8, right: 8, bottom: 0, left: 0 }} barCategoryGap={2}>
                <CartesianGrid vertical={false} stroke="var(--rule)" />
                <XAxis dataKey="rating" tick={TICK} axisLine={{ stroke: "var(--rule)" }} tickLine={false} tickFormatter={(v: number) => `${v}★`} />
                <YAxis tick={TICK} axisLine={false} tickLine={false} width={32} allowDecimals={false} />
                <Tooltip cursor={{ fill: "var(--accent)", opacity: 0.4 }} content={({ active, payload, label }) => (
                  <ChartTooltip active={active} label={`${label}★`} digits={0}
                    payload={Array.isArray(payload) ? payload.map((x) => ({ name: "films", value: x.value as number, color: "var(--sig-content)" })) : undefined} />
                )} />
                <Bar dataKey="count" fill="var(--sig-content)" radius={[4, 4, 0, 0]} maxBarSize={32} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </section>
      </div>

      <div className="mt-14 grid gap-12 lg:grid-cols-3">
        <section>
          <SectionHeader kicker="Directors of films you liked" title="Directors" />
          {p.favorite_directors.length === 0 ? <p className="text-sm text-muted-foreground">None yet.</p> : (
            <ul className="divide-y hairline border-y hairline">
              {p.favorite_directors.map((d) => (
                <li key={d.name} className="flex items-baseline justify-between py-2 text-sm">
                  <Link className="hover:text-primary" href={`/discover?q=${encodeURIComponent(d.name)}`}>{d.name}</Link>
                  <span className="num text-muted-foreground">{d.count}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
        <section>
          <SectionHeader kicker="In 2+ films you liked" title="Actors" />
          {p.favorite_actors.length === 0 ? <p className="text-sm text-muted-foreground">No actor appears in two of your liked films yet.</p> : (
            <ul className="divide-y hairline border-y hairline">
              {p.favorite_actors.map((d) => (
                <li key={d.name} className="flex items-baseline justify-between py-2 text-sm">
                  <Link className="hover:text-primary" href={`/discover?q=${encodeURIComponent(d.name)}`}>{d.name}</Link>
                  <span className="num text-muted-foreground">{d.count}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
        <section>
          <SectionHeader kicker="Normalised: picks + ratings" title="Preference vector" />
          {!p.preference_vector?.length ? <p className="text-sm text-muted-foreground">Empty until you pick genres or rate films.</p> : (
            <ul className="space-y-2">
              {p.preference_vector.slice(0, 10).map((g) => {
                const max = Math.max(...p.preference_vector!.map((x) => Math.abs(x.weight)));
                return (
                  <li key={g.genre} className="grid grid-cols-[88px_1fr_48px] items-center gap-2 text-sm">
                    <span className="truncate text-ink-2">{g.genre}</span>
                    <span className="h-1.5 rounded-full bg-muted">
                      <span className="block h-full rounded-full" style={{ width: `${(Math.abs(g.weight) / max) * 100}%`, background: g.weight >= 0 ? "var(--sig-preference)" : "var(--muted-foreground)" }} />
                    </span>
                    <span className="num text-right text-xs text-muted-foreground">{g.weight.toFixed(2)}</span>
                  </li>
                );
              })}
            </ul>
          )}
        </section>
      </div>

      <section className="mt-14 grid gap-10 lg:grid-cols-2">
        <div>
          <SectionHeader kicker="Onboarding choices" title="Favourite genres" />
          <GenreEditor key={p.explicit_genres.join()} profile={p} onSaved={() => void mutate()} />
        </div>
        <div>
          <SectionHeader kicker="Diversity re-ranking (MMR λ)" title="Recommendation variety" />
          <ToggleGroup type="single" variant="outline" value={p.preferences.diversity ?? "balanced"} aria-label="Recommendation variety"
            onValueChange={async (v) => {
              if (!v) return;
              try { await api("/users/me/preferences", { method: "PATCH", json: { diversity: v } }); toast.success("Saved"); void mutate(); }
              catch (e) { toast.error(errorMessage(e)); }
            }}>
            <ToggleGroupItem value="focused">Focused</ToggleGroupItem>
            <ToggleGroupItem value="balanced">Balanced</ToggleGroupItem>
            <ToggleGroupItem value="adventurous">Adventurous</ToggleGroupItem>
          </ToggleGroup>
          <p className="mt-3 text-sm text-muted-foreground">Focused ranks purely by score. Adventurous trades a little relevance for more variety between neighbouring picks.</p>
        </div>
      </section>
      {p.model_version && <p className="mt-12 font-mono text-[11px] text-muted-foreground">profile computed with model {p.model_version}</p>}
    </Container>
  );
}
