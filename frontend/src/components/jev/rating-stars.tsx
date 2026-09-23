"use client";

import { Star } from "lucide-react";
import { useState } from "react";

import { cn } from "@/lib/utils";

/** 0.5–5 star input. Left half of a star = x.5. Keyboard: arrows change by 0.5, Delete clears. */
export function RatingStars({ value, onChange, onClear, size = 22, readOnly }: { value: number | null; onChange?: (v: number) => void; onClear?: () => void; size?: number; readOnly?: boolean }) {
  const [hover, setHover] = useState<number | null>(null);
  const shown = hover ?? value ?? 0;
  return (
    <div
      className="inline-flex items-center"
      role={readOnly ? "img" : "slider"}
      aria-label={readOnly ? `Rated ${value ?? 0} of 5` : "Your rating"}
      aria-valuemin={0.5}
      aria-valuemax={5}
      aria-valuenow={value ?? undefined}
      aria-valuetext={value ? `${value} stars` : "not rated"}
      tabIndex={readOnly ? -1 : 0}
      onKeyDown={(e) => {
        if (readOnly || !onChange) return;
        if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); onChange(Math.min(5, (value ?? 0) + 0.5)); }
        if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); onChange(Math.max(0.5, (value ?? 1) - 0.5)); }
        if ((e.key === "Delete" || e.key === "Backspace") && onClear) onClear();
      }}
      onMouseLeave={() => setHover(null)}
    >
      {[1, 2, 3, 4, 5].map((i) => {
        const fill = shown >= i ? 1 : shown >= i - 0.5 ? 0.5 : 0;
        return (
          <span key={i} className="relative" style={{ width: size, height: size }}>
            <Star className="absolute inset-0 text-muted-foreground/40" style={{ width: size, height: size }} strokeWidth={1.4} />
            <span className="absolute inset-0 overflow-hidden" style={{ width: `${fill * 100}%` }}>
              <Star className="text-primary" style={{ width: size, height: size }} fill="currentColor" strokeWidth={1.4} />
            </span>
            {!readOnly && (
              <>
                <button type="button" tabIndex={-1} aria-label={`${i - 0.5} stars`} className="absolute inset-y-0 left-0 w-1/2 cursor-pointer"
                  onMouseEnter={() => setHover(i - 0.5)} onClick={() => onChange?.(i - 0.5)} />
                <button type="button" tabIndex={-1} aria-label={`${i} stars`} className="absolute inset-y-0 right-0 w-1/2 cursor-pointer"
                  onMouseEnter={() => setHover(i)} onClick={() => onChange?.(i)} />
              </>
            )}
          </span>
        );
      })}
      {!readOnly && value !== null && (
        <span className={cn("num ml-2 text-sm text-muted-foreground")}>{value.toFixed(1)}</span>
      )}
    </div>
  );
}
