import { AlertTriangle, ChevronLeft, ChevronRight } from "lucide-react";
import Link from "next/link";

import { EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, errorMessage } from "@/lib/api";
import { isNoRun, isNotDeployed } from "@/lib/intel";
import { cn } from "@/lib/utils";

export function NoRunState({ detail, className }: { detail?: string; className?: string }) {
  return (
    <EmptyState
      className={className}
      title="No intelligence run yet"
      body={`Run the pipeline from the overview to populate this view.${detail ? ` Server said: “${detail}”.` : ""}`}
      action={<Button asChild variant="outline"><Link href="/intel">Go to the overview</Link></Button>}
    />
  );
}

/** The API answering does not have this endpoint yet (an older build than the console). */
export function NotDeployedState({ what, className }: { what: string; className?: string }) {
  return (
    <EmptyState
      className={className}
      title="Not available on this API version"
      body={`The API that answered does not provide ${what} yet. Upgrade the backend to v1.1 or later; nothing is shown rather than guessed.`}
    />
  );
}

/**
 * Error panel for /intel/* calls. A 404/503 on a run-backed endpoint means "no run yet" and gets
 * the empty state instead (pass runBacked={false} for DB-backed lookups where 404 is a real miss).
 * Pass `what` on v1.1 endpoints: a route-miss 404 then reads "not available on this API version".
 */
export function IntelError({ error, retry, runBacked = true, what, className }: { error: unknown; retry?: () => void; runBacked?: boolean; what?: string; className?: string }) {
  if (what && isNotDeployed(error)) return <NotDeployedState what={what} className={className} />;
  if (runBacked && isNoRun(error)) return <NoRunState className={className} detail={error instanceof ApiError ? error.message : undefined} />;
  const status = error instanceof ApiError ? error.status : null;
  const requestId = error instanceof ApiError ? error.requestId : undefined;
  const message =
    status === 503 ? `The intelligence service is unavailable (${(error as ApiError).message}).`
      : status === 403 ? "This view is limited to administrator accounts."
        : errorMessage(error);
  return (
    <div role="alert" className={cn("flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-4", className)}>
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
      <div className="min-w-0 flex-1 text-sm">
        <p>{message}</p>
        {(status || requestId) && (
          <p className="mt-1 break-all font-mono text-xs text-muted-foreground">
            {status ? `HTTP ${status}` : ""}{status && requestId ? " · " : ""}{requestId ? `request ${requestId}` : ""}
          </p>
        )}
        {retry && (
          <Button variant="link" size="sm" className="h-auto px-0 text-primary" onClick={retry}>Try again</Button>
        )}
      </div>
    </div>
  );
}

export function RowsSkeleton({ rows = 6, className }: { rows?: number; className?: string }) {
  return (
    <div className={cn("space-y-px overflow-hidden rounded-lg border", className)} aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="flex items-center gap-4 bg-card px-4 py-3.5">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-3.5 flex-1" />
          <Skeleton className="hidden h-3 w-24 sm:block" />
        </div>
      ))}
    </div>
  );
}

export function PanelsSkeleton({ n = 3, className }: { n?: number; className?: string }) {
  return (
    <div className={cn("grid gap-4 lg:grid-cols-3", className)} aria-busy="true" aria-label="Loading">
      {Array.from({ length: n }, (_, i) => <Skeleton key={i} className="h-40 w-full" />)}
    </div>
  );
}

export function Pagination({ total, limit, offset, onChange, className }: { total: number; limit: number; offset: number; onChange: (offset: number) => void; className?: string }) {
  if (total <= limit) return null;
  const from = offset + 1;
  const to = Math.min(offset + limit, total);
  return (
    <nav aria-label="Pagination" className={cn("mt-4 flex items-center justify-between gap-3", className)}>
      <p className="num text-xs text-muted-foreground">{from}–{to} of {total.toLocaleString()}</p>
      <div className="flex gap-1.5">
        <Button variant="outline" size="sm" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>
          <ChevronLeft aria-hidden /> Previous
        </Button>
        <Button variant="outline" size="sm" disabled={to >= total} onClick={() => onChange(offset + limit)}>
          Next <ChevronRight aria-hidden />
        </Button>
      </div>
    </nav>
  );
}

/** Filter chip row: one row above the content it scopes; scrolls sideways on a phone. */
export function FilterRow({ label, options, value, onChange }: { label: string; options: { value: string; label: string }[]; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <span className="eyebrow shrink-0">{label}</span>
      <div className="scrollbar-none -mx-1 flex min-w-0 gap-1 overflow-x-auto px-1 py-0.5" role="radiogroup" aria-label={label}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            role="radio"
            aria-checked={value === o.value}
            onClick={() => onChange(o.value)}
            className={cn(
              "shrink-0 rounded border px-2 py-1 text-xs transition-colors",
              value === o.value ? "border-primary bg-primary/10 text-foreground" : "hairline text-muted-foreground hover:text-foreground",
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}
