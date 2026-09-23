"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef } from "react";

import { Container, EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/hooks/use-session";
import { cn } from "@/lib/utils";

/** Sections in pipeline order: see what changed, what may happen, what to do, how well it works. */
const GROUPS: { label: string; items: { href: string; label: string }[] }[] = [
  { label: "Now", items: [{ href: "/intel", label: "Overview" }] },
  {
    label: "Detect",
    items: [
      { href: "/intel/signals", label: "Signals" },
      { href: "/intel/trends", label: "Trends" },
      { href: "/intel/anomalies", label: "Anomalies" },
    ],
  },
  {
    label: "Anticipate",
    items: [
      { href: "/intel/predictions", label: "Predictions" },
      { href: "/intel/risks", label: "Risks" },
      { href: "/intel/warnings", label: "Early warnings" },
    ],
  },
  {
    label: "Decide",
    items: [
      { href: "/intel/decisions", label: "Decisions" },
      { href: "/intel/actions", label: "Actions" },
      { href: "/intel/scenarios", label: "What-if" },
    ],
  },
  {
    label: "Review",
    items: [
      { href: "/intel/feedback", label: "Feedback" },
      { href: "/intel/evaluation", label: "Evaluation" },
      { href: "/intel/health", label: "System health" },
    ],
  },
];
const ITEMS = GROUPS.flatMap((g) => g.items);

function isActive(href: string, pathname: string) {
  if (href === "/intel") return pathname === "/intel";
  if (href === "/intel/trends") return pathname.startsWith("/intel/trends") || pathname.startsWith("/intel/series");
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function IntelLayout({ children }: { children: React.ReactNode }) {
  const { user, loading } = useSession();
  const router = useRouter();
  const pathname = usePathname();
  const strip = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!loading && !user) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [loading, user, router, pathname]);

  // keep the current section visible in the phone tab strip
  useEffect(() => {
    strip.current?.querySelector<HTMLElement>("[aria-current=page]")?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [pathname, user]);

  if (loading || !user) {
    return (
      <Container className="space-y-4 pt-14">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-14 w-80" />
        <Skeleton className="h-64 w-full" />
      </Container>
    );
  }
  if (!user.is_admin) {
    return (
      <Container className="pt-14">
        <EmptyState
          title="Admins only"
          body="The Intelligence console is limited to administrator accounts."
          action={<Button asChild variant="outline"><Link href="/home">Back to your programme</Link></Button>}
        />
      </Container>
    );
  }

  return (
    <Container>
      <div className="lg:grid lg:grid-cols-[176px_minmax(0,1fr)] lg:gap-12">
        {/* phone / tablet: one scrollable strip of tabs */}
        <nav aria-label="Intelligence sections" className="lg:hidden">
          <div ref={strip} className="scrollbar-none -mx-5 mt-6 flex gap-5 overflow-x-auto border-b hairline px-5 sm:-mx-8 sm:px-8">
            {ITEMS.map((t) => {
              const active = isActive(t.href, pathname);
              return (
                <Link
                  key={t.href}
                  href={t.href}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "-mb-px shrink-0 border-b py-2.5 text-sm transition-colors",
                    active ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
                  )}
                >
                  {t.label}
                </Link>
              );
            })}
          </div>
        </nav>

        {/* desktop: a left rail grouped by pipeline stage */}
        <nav aria-label="Intelligence sections" className="sticky top-20 hidden self-start pt-12 lg:block">
          <p className="font-display text-2xl leading-none">Intelligence</p>
          <p className="eyebrow mt-1.5">operator console</p>
          <div className="mt-6 space-y-5">
            {GROUPS.map((g) => (
              <div key={g.label}>
                <p className="eyebrow mb-1.5">{g.label}</p>
                <ul className="border-l hairline">
                  {g.items.map((t) => {
                    const active = isActive(t.href, pathname);
                    return (
                      <li key={t.href}>
                        <Link
                          href={t.href}
                          aria-current={active ? "page" : undefined}
                          className={cn(
                            "-ml-px block border-l py-1 pl-3 text-sm transition-colors",
                            active ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
                          )}
                        >
                          {t.label}
                        </Link>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </div>
          <Link href="/admin" className="mt-8 block text-xs text-muted-foreground hover:text-foreground">← Admin control room</Link>
        </nav>

        <div className="min-w-0 pt-8 lg:pt-12">{children}</div>
      </div>
    </Container>
  );
}
