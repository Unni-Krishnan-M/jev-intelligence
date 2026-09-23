import { cn } from "@/lib/utils";
import { posterLayout, posterTheme, primaryGenre, titleScale } from "@/lib/poster";

interface PosterMovie {
  id: number;
  title: string;
  year: number | null;
  genres: string[];
  directors?: string[];
}

/**
 * A typeset cover built from the movie's real metadata. The container query unit (cqw) scales the
 * type with the card width, so the same component works from a 120px shelf tile to a 360px hero.
 */
export function Poster({ movie, className, priority }: { movie: PosterMovie; className?: string; priority?: boolean }) {
  const theme = posterTheme(movie.genres);
  const layout = posterLayout(movie.id);
  const genre = primaryGenre(movie.genres);
  const director = movie.directors?.[0];
  const scale = titleScale(movie.title);
  const yy = movie.year ? `’${String(movie.year).slice(-2)}` : "";

  return (
    <div
      className={cn("grain @container relative aspect-[2/3] w-full overflow-hidden rounded-[6px] select-none", className)}
      style={{ background: theme.bg, color: theme.ink }}
      aria-hidden={!priority}
      role="img"
      aria-label={`${movie.title}${movie.year ? ` (${movie.year})` : ""}`}
    >
      {/* inner frame, like a printed programme plate */}
      <div className="pointer-events-none absolute inset-[5%] rounded-[3px] border" style={{ borderColor: `${theme.ink}26` }} />

      {layout === "numeral" && yy && (
        <span
          className="font-display pointer-events-none absolute -bottom-[6%] -right-[4%] leading-none"
          style={{ fontSize: "62cqw", color: theme.accent, opacity: 0.28 }}
        >
          {yy}
        </span>
      )}

      <div className="relative z-[2] flex h-full flex-col p-[11%]">
        <div className="flex items-center justify-between font-mono uppercase" style={{ fontSize: "5.2cqw", letterSpacing: "0.14em" }}>
          <span style={{ color: theme.accent }}>{genre ?? "Film"}</span>
          <span style={{ opacity: 0.75 }}>{movie.year ?? ""}</span>
        </div>

        {layout === "rule" ? (
          <div className="my-auto text-center">
            <div className="mx-auto mb-[7%] h-px w-1/3" style={{ background: theme.accent }} />
            <h3 className="font-display leading-[0.95] text-balance" style={{ fontSize: `${18 * scale}cqw` }}>
              {movie.title}
            </h3>
            <div className="mx-auto mt-[7%] h-px w-1/3" style={{ background: theme.accent }} />
            {director && (
              <p className="mt-[8%] font-mono uppercase" style={{ fontSize: "4.6cqw", letterSpacing: "0.12em", opacity: 0.8 }}>
                a film by {director}
              </p>
            )}
          </div>
        ) : layout === "numeral" ? (
          <div className="mt-[10%]">
            <h3 className="font-display leading-[0.95] text-balance" style={{ fontSize: `${19 * scale}cqw` }}>
              {movie.title}
            </h3>
            {director && (
              <p className="mt-[6%] font-mono uppercase" style={{ fontSize: "4.6cqw", letterSpacing: "0.12em", opacity: 0.8 }}>
                dir. {director}
              </p>
            )}
          </div>
        ) : (
          <div className="mt-auto">
            <div className="mb-[6%] h-[2px] w-[18%]" style={{ background: theme.accent }} />
            <h3 className="font-display leading-[0.92] text-balance" style={{ fontSize: `${19 * scale}cqw` }}>
              {movie.title}
            </h3>
            {director && (
              <p className="mt-[6%] font-mono uppercase" style={{ fontSize: "4.6cqw", letterSpacing: "0.12em", opacity: 0.8 }}>
                dir. {director}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
