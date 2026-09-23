"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/button";

import { PosterSkeleton } from "./states";

/** Horizontal, snap-scrolling row. Card width follows the viewport, so ~6 cards fit on a 1366px laptop. */
export function Shelf({ children, loading, label }: { children?: React.ReactNode; loading?: boolean; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ start: true, end: false });

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () => setEdges({ start: el.scrollLeft < 8, end: el.scrollLeft + el.clientWidth >= el.scrollWidth - 8 });
    update();
    el.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => { el.removeEventListener("scroll", update); window.removeEventListener("resize", update); };
  }, [children, loading]);

  const page = (dir: 1 | -1) => ref.current?.scrollBy({ left: dir * ref.current.clientWidth * 0.85, behavior: "smooth" });

  return (
    <div className="relative">
      <div ref={ref} role="list" aria-label={label}
        className="scrollbar-none -mx-5 flex snap-x snap-mandatory gap-4 overflow-x-auto scroll-px-5 px-5 pb-2 sm:-mx-8 sm:scroll-px-8 sm:px-8 [&>*]:w-[42vw] [&>*]:shrink-0 [&>*]:snap-start sm:[&>*]:w-[28vw] md:[&>*]:w-[21vw] lg:[&>*]:w-[15.5vw] xl:[&>*]:w-[188px]">
        {loading ? Array.from({ length: 7 }, (_, i) => <PosterSkeleton key={i} />) : children}
      </div>
      {!loading && (
        <div className="pointer-events-none absolute -top-12 right-0 hidden gap-1 md:flex">
          <Button variant="outline" size="icon" className="pointer-events-auto size-8" onClick={() => page(-1)} disabled={edges.start} aria-label={`Scroll ${label} left`}><ChevronLeft className="size-4" /></Button>
          <Button variant="outline" size="icon" className="pointer-events-auto size-8" onClick={() => page(1)} disabled={edges.end} aria-label={`Scroll ${label} right`}><ChevronRight className="size-4" /></Button>
        </div>
      )}
    </div>
  );
}
