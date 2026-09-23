"use client";

import { useSWRConfig } from "swr";
import { toast } from "sonner";

import { api, errorMessage } from "@/lib/api";

/** Movie interactions + feedback. After any taste-relevant event, personal shelves revalidate. */
export function useInteractions() {
  const { mutate } = useSWRConfig();

  const refreshPersonal = () =>
    mutate(
      (key) =>
        typeof key === "string" &&
        (key.startsWith("/recommendations") || key.startsWith("/users/me") || key.startsWith("/movies/")),
    );

  async function run<T>(fn: () => Promise<T>, ok?: string): Promise<T | undefined> {
    try {
      const out = await fn();
      if (ok) toast.success(ok);
      void refreshPersonal();
      return out;
    } catch (e) {
      toast.error(errorMessage(e));
      return undefined;
    }
  }

  return {
    rate: (movieId: number, rating: number) =>
      run(() => api(`/movies/${movieId}/rate`, { json: { rating } }), `Rated ${rating} ★`),
    unrate: (movieId: number) => run(() => api(`/movies/${movieId}/rate`, { method: "DELETE" }), "Rating removed"),
    favorite: (movieId: number, favorite: boolean) =>
      run(() => api(`/movies/${movieId}/favorite`, { json: { favorite } }), favorite ? "Added to favourites" : "Removed from favourites"),
    watch: (movieId: number) => run(() => api(`/movies/${movieId}/watch`, { method: "POST" }), "Logged as watched"),
    feedback: (movieId: number, feedback: "like" | "dislike" | "not_interested" | "clicked", recommendationId?: number | null) =>
      run(
        () => api(`/recommendations/feedback`, { json: { movie_id: movieId, feedback, recommendation_id: recommendationId ?? null } }),
        feedback === "not_interested" ? "Got it — we won't suggest that again" : feedback === "like" ? "Thanks — noted" : undefined,
      ),
    refreshPersonal,
  };
}
