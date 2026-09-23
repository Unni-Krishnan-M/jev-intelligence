import { CheckCircle2, XCircle } from "lucide-react";

import { cn } from "@/lib/utils";

export function Panel({ title, action, children, className }: { title?: React.ReactNode; action?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section className={cn("rounded-lg border bg-card p-4 sm:p-5", className)}>
      {(title || action) && (
        <div className="mb-3 flex items-center justify-between gap-3">
          {title && <h3 className="eyebrow">{title}</h3>}
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

/** Hairline "spec sheet" rows: label left, mono value right. */
export function SpecRows({ rows }: { rows: { label: string; value: React.ReactNode }[] }) {
  return (
    <dl className="text-sm">
      {rows.map((r) => (
        <div key={r.label} className="flex items-baseline justify-between gap-4 border-b hairline py-2 last:border-0">
          <dt className="text-muted-foreground">{r.label}</dt>
          <dd className="num min-w-0 truncate text-right">{r.value}</dd>
        </div>
      ))}
    </dl>
  );
}

/** Status is carried by icon + label, never by colour alone. */
export function Status({ ok, label }: { ok: boolean; label?: string }) {
  const Icon = ok ? CheckCircle2 : XCircle;
  return (
    <span className={cn("inline-flex items-center gap-1.5 font-sans", ok ? "text-[#0ca30c]" : "text-destructive")}>
      <Icon className="size-3.5" aria-hidden />
      <span className="text-foreground">{label ?? (ok ? "ok" : "error")}</span>
    </span>
  );
}

export function StatTile({ label, value, hint }: { label: string; value: React.ReactNode; hint?: string }) {
  return (
    <div className="rounded-lg border bg-card px-4 py-4">
      <p className="eyebrow">{label}</p>
      <p className="font-display mt-1 text-4xl leading-none">{value}</p>
      {hint && <p className="mt-2 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

export function fmtNum(v: number | null | undefined, digits = 4): string {
  return v === null || v === undefined || Number.isNaN(v) ? "—" : v.toFixed(digits);
}

export function fmtDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toISOString().slice(0, 16).replace("T", " ");
}
