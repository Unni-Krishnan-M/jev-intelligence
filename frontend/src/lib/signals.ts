import type { SignalName } from "./types";

/** Fixed order = fixed colour slot. Never reorder: colour follows the signal, not its rank. */
export const SIGNALS: SignalName[] = ["content", "collaborative", "latent", "popularity", "preference", "recency"];

export const SIGNAL_META: Record<SignalName, { label: string; short: string; model: string; color: string; blurb: string }> = {
  content: {
    label: "Content similarity", short: "Content", model: "TF-IDF + cosine", color: "var(--sig-content)",
    blurb: "Shared genres, directors, cast, themes and tags with films you liked.",
  },
  collaborative: {
    label: "Collaborative filtering", short: "Co-watch", model: "Item-kNN", color: "var(--sig-collaborative)",
    blurb: "Viewers who liked what you liked also liked this.",
  },
  latent: {
    label: "Latent factors", short: "Latent", model: "Implicit ALS", color: "var(--sig-latent)",
    blurb: "Matrix factorization: hidden taste dimensions learned from the full rating history.",
  },
  popularity: {
    label: "Popularity", short: "Popular", model: "Like counts", color: "var(--sig-popularity)",
    blurb: "How many viewers rated it 4 stars or more.",
  },
  preference: {
    label: "Genre preference", short: "Genres", model: "Profile vector", color: "var(--sig-preference)",
    blurb: "Overlap with genres you picked and genres of films you rated.",
  },
  recency: {
    label: "Recency", short: "Recent", model: "Release year decay", color: "var(--sig-recency)",
    blurb: "A gentle preference for newer releases.",
  },
};

export function dominantSignal(signals: Record<SignalName, { contribution: number }>): SignalName {
  return SIGNALS.reduce((best, s) => (signals[s].contribution > signals[best].contribution ? s : best), SIGNALS[0]);
}
