import { AlertTriangle } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";

export function PosterSkeleton({ className }: { className?: string }) {
  return (
    <div className={cn("space-y-2", className)}>
      <Skeleton className="aspect-[2/3] w-full rounded-[6px]" />
      <Skeleton className="h-3.5 w-4/5" />
      <Skeleton className="h-3 w-1/2" />
    </div>
  );
}

export function EmptyState({ title, body, action, className }: { title: string; body?: string; action?: React.ReactNode; className?: string }) {
  return (
    <div className={cn("rounded-lg border border-dashed px-6 py-10 text-center", className)}>
      <p className="font-display text-2xl">{title}</p>
      {body && <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">{body}</p>}
      {action && <div className="mt-5">{action}</div>}
    </div>
  );
}

export function ErrorState({ error, retry, className }: { error: unknown; retry?: () => void; className?: string }) {
  return (
    <div role="alert" className={cn("flex items-start gap-3 rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-4", className)}>
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
      <div className="flex-1 text-sm">
        <p>{errorMessage(error)}</p>
        {retry && (
          <Button variant="link" size="sm" className="h-auto px-0 text-primary" onClick={retry}>Try again</Button>
        )}
      </div>
    </div>
  );
}

export function SectionHeader({ index, title, kicker, action, id }: { index?: number; title: React.ReactNode; kicker?: React.ReactNode; action?: React.ReactNode; id?: string }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-x-6 gap-y-2 border-t hairline pt-4">
      <div className="min-w-0">
        <p className="eyebrow">{index !== undefined ? `${String(index).padStart(2, "0")} — ` : ""}{kicker}</p>
        <h2 id={id} className="font-display mt-1 text-[28px] leading-tight sm:text-[32px]">{title}</h2>
      </div>
      {action}
    </div>
  );
}

export function PageHeader({ eyebrow, title, children }: { eyebrow?: string; title: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="mb-10 pt-10 sm:pt-14">
      {eyebrow && <p className="eyebrow">{eyebrow}</p>}
      <h1 className="font-display mt-2 text-[44px] leading-[0.95] tracking-tight text-balance sm:text-[64px]">{title}</h1>
      {children && <div className="mt-4 max-w-2xl text-[15px] leading-relaxed text-ink-2">{children}</div>}
    </div>
  );
}

export function Container({ children, className }: { children: React.ReactNode; className?: string }) {
  return <div className={cn("mx-auto w-full max-w-[1320px] px-5 sm:px-8", className)}>{children}</div>;
}
