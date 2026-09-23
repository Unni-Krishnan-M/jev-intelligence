"use client";

import { Search } from "lucide-react";
import { useRouter } from "next/navigation";
import { useEffect, useId, useRef, useState } from "react";
import useSWR from "swr";

import { qs } from "@/lib/api";
import type { MovieBrief } from "@/lib/types";
import { cn } from "@/lib/utils";

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setV(value), ms);
    return () => clearTimeout(t);
  }, [value, ms]);
  return v;
}

export function SearchBox({ className, onNavigate }: { className?: string; onNavigate?: () => void }) {
  const router = useRouter();
  const [q, setQ] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const listId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const term = useDebounced(q.trim(), 180);
  const { data, isLoading } = useSWR<MovieBrief[]>(term.length >= 2 ? `/movies/search${qs({ q: term, limit: 8 })}` : null);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && document.activeElement === document.body)) {
        e.preventDefault();
        inputRef.current?.focus();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  function go(id: number) {
    setOpen(false);
    setQ("");
    onNavigate?.();
    router.push(`/movies/${id}`);
  }

  const results = data ?? [];
  return (
    <div className={cn("relative", className)}>
      <label htmlFor={`${listId}-input`} className="sr-only">Search films</label>
      <div className="flex h-9 items-center gap-2 rounded-md border bg-card/60 px-3 focus-within:border-primary/60">
        <Search className="size-4 text-muted-foreground" aria-hidden />
        <input
          id={`${listId}-input`}
          ref={inputRef}
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            setOpen(true);
            setActive(0);
          }}
          onFocus={() => setOpen(true)}
          onBlur={() => setTimeout(() => setOpen(false), 120)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(a + 1, results.length - 1)); }
            if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(a - 1, 0)); }
            if (e.key === "Enter") {
              if (results[active]) go(results[active].id);
              else if (q.trim()) { onNavigate?.(); router.push(`/discover${qs({ q: q.trim() })}`); setOpen(false); }
            }
            if (e.key === "Escape") { setOpen(false); inputRef.current?.blur(); }
          }}
          placeholder="Titles, directors, cast"
          className="w-full bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          role="combobox"
          aria-expanded={open && results.length > 0}
          aria-controls={listId}
          aria-autocomplete="list"
          autoComplete="off"
        />
        <kbd className="hidden rounded border px-1.5 font-mono text-[10px] text-muted-foreground lg:inline">⌘K</kbd>
      </div>
      {open && term.length >= 2 && (
        <div id={listId} role="listbox" className="absolute left-0 right-0 top-11 z-50 overflow-hidden rounded-md border bg-popover shadow-2xl shadow-black/40">
          {isLoading && !data && <p className="px-3 py-3 text-sm text-muted-foreground">Searching…</p>}
          {data && results.length === 0 && <p className="px-3 py-3 text-sm text-muted-foreground">No films match “{term}”.</p>}
          {results.map((m, i) => (
            <button
              key={m.id}
              role="option"
              aria-selected={i === active}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => go(m.id)}
              onMouseEnter={() => setActive(i)}
              className={cn("flex w-full items-baseline justify-between gap-3 px-3 py-2 text-left", i === active && "bg-accent")}
            >
              <span className="truncate text-sm">{m.title}</span>
              <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
                {m.year ?? "—"}{m.directors[0] ? ` · ${m.directors[0]}` : ""}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
