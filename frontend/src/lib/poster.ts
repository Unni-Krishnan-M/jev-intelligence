/** Deterministic typographic poster styling from real movie metadata (no images). */

export interface PosterTheme {
  bg: string;
  ink: string;
  accent: string;
}

const THEMES: Record<string, PosterTheme> = {
  "Sci-Fi": { bg: "#10222b", ink: "#d8ecef", accent: "#6fc3cf" },
  Horror: { bg: "#1a0d0d", ink: "#efe0dc", accent: "#c8372d" },
  Animation: { bg: "#1d2f6f", ink: "#f5efe2", accent: "#f2b134" },
  Documentary: { bg: "#2c2f27", ink: "#e7e4d8", accent: "#b5c27a" },
  Western: { bg: "#4a2a17", ink: "#f1dfc6", accent: "#e08a3c" },
  War: { bg: "#2e3322", ink: "#e6e2cf", accent: "#c9b458" },
  "Film-Noir": { bg: "#121212", ink: "#eeeeea", accent: "#a3a39c" },
  Musical: { bg: "#4b1d3f", ink: "#f7e6f0", accent: "#e79ac7" },
  Crime: { bg: "#1b1b1f", ink: "#ebe7df", accent: "#d9a441" },
  Thriller: { bg: "#16202a", ink: "#e5e9ee", accent: "#e05a47" },
  Fantasy: { bg: "#2a1f3d", ink: "#efe7fb", accent: "#c7a6f2" },
  Romance: { bg: "#5a1f2b", ink: "#fbe9e6", accent: "#f0a39a" },
  Comedy: { bg: "#d9aa35", ink: "#1d1a14", accent: "#8a2d1d" },
  Mystery: { bg: "#1e2530", ink: "#e3e7ec", accent: "#8fb3d9" },
  Adventure: { bg: "#1f3a2c", ink: "#eef3e6", accent: "#e8c268" },
  Action: { bg: "#2a1410", ink: "#f4e7df", accent: "#ff6a3d" },
  Drama: { bg: "#3b2a24", ink: "#f1e6dc", accent: "#d8a67d" },
  Children: { bg: "#efe0bf", ink: "#20304a", accent: "#d9573a" },
};
const DEFAULT: PosterTheme = { bg: "#24211d", ink: "#ece5d8", accent: "#e9a23b" };

// most visually defining genre first (MovieLens genre lists are alphabetical)
const PRIORITY = [
  "Animation", "Documentary", "Horror", "Film-Noir", "Western", "War", "Sci-Fi", "Musical", "Fantasy", "Crime",
  "Thriller", "Mystery", "Romance", "Comedy", "Children", "Adventure", "Action", "Drama",
];

export function primaryGenre(genres: string[]): string | null {
  for (const g of PRIORITY) if (genres.includes(g)) return g;
  return genres[0] ?? null;
}

export function posterTheme(genres: string[]): PosterTheme {
  const g = primaryGenre(genres);
  return (g && THEMES[g]) || DEFAULT;
}

export function hash(n: number): number {
  let x = n | 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  return (x ^ (x >>> 16)) >>> 0;
}

export type PosterLayout = "classic" | "numeral" | "rule";

export function posterLayout(id: number): PosterLayout {
  return (["classic", "numeral", "rule"] as const)[hash(id) % 3];
}

/** Title size tuned to length so long titles still set well on a narrow card. */
export function titleScale(title: string): number {
  const n = title.length;
  if (n <= 8) return 1.0;
  if (n <= 14) return 0.82;
  if (n <= 22) return 0.68;
  if (n <= 34) return 0.56;
  return 0.46;
}
