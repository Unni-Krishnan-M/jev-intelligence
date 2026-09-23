import { fmtDate } from "@/components/jev/admin/ui";

/** Console page header: eyebrow, serif title, one-line description and the run it reads from. */
export function PageHeader({
  eyebrow,
  title,
  description,
  asOf,
  runId,
  action,
}: {
  eyebrow: string;
  title: React.ReactNode;
  description?: React.ReactNode;
  asOf?: string | null;
  runId?: string | null;
  action?: React.ReactNode;
}) {
  return (
    <div className="mb-8 flex flex-wrap items-end justify-between gap-x-8 gap-y-4 border-b hairline pb-6">
      <div className="min-w-0 max-w-2xl">
        <p className="eyebrow">Intelligence · {eyebrow}</p>
        <h1 className="font-display mt-2 text-[40px] leading-[0.95] tracking-tight text-balance sm:text-[52px]">{title}</h1>
        {description && <p className="mt-3 text-[15px] leading-relaxed text-ink-2">{description}</p>}
        {(asOf || runId) && (
          <p className="mt-3 font-mono text-xs text-muted-foreground">
            {asOf && <>as of {fmtDate(asOf)} UTC</>}
            {asOf && runId && " · "}
            {runId && <span className="break-all">run {runId.slice(0, 8)}</span>}
          </p>
        )}
      </div>
      {action}
    </div>
  );
}
