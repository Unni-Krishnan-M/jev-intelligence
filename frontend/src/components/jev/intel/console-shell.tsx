"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Suspense, useEffect, useRef } from "react";

import { DomainPicker, DomainProvider, IntelLink } from "@/components/jev/intel/domain-context";
import { Container, EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/hooks/use-session";
import { cn } from "@/lib/utils";

type NavItem = { href: string; label: string; children?: NavItem[] };

/**
 * Sections in engine order (docs/platform.md §9): what changed, what may happen, what JEV decided
 * and recommends, where every claim came from, how well it works; then the system itself. Risks
 * and actions sit under the stage they feed.
 */
export const CONSOLE_NAV: NavItem[] = [
  { href: "/intel", label: "Overview" },
  { href: "/intel/signals", label: "Signals" },
  { href: "/intel/trends", label: "Trends" },
  { href: "/intel/anomalies", label: "Anomalies" },
  { href: "/intel/predictions", label: "Predictions", children: [{ href: "/intel/risks", label: "Risks" }] },
  { href: "/intel/warnings", label: "Early warnings" },
  { href: "/intel/decisions", label: "Decisions", children: [{ href: "/intel/actions", label: "Actions" }] },
  { href: "/intel/recommendations", label: "Recommendations" },
  { href: "/intel/evidence", label: "Evidence" },
  { href: "/intel/scenarios", label: "Scenarios" },
  { href: "/intel/feedback", label: "Feedback" },
  { href: "/intel/evaluation", label: "Model evaluation" },
];

export const SYSTEM_NAV: NavItem[] = [
  { href: "/intel/health", label: "Health" },
  { href: "/intel/audit", label: "Audit log" },
];

const FLAT = [...CONSOLE_NAV.flatMap((i) => [i, ...(i.children ?? [])]), ...SYSTEM_NAV];

function isActive(href: string, pathname: string) {
  if (href === "/intel") return pathname === "/intel";
  if (href === "/intel/trends") return pathname.startsWith("/intel/trends") || pathname.startsWith("/intel/series");
  return pathname === href || pathname.startsWith(`${href}/`);
}

function RailLink({ item, pathname, nested = false }: { item: NavItem; pathname: string; nested?: boolean }) {
  const active = isActive(item.href, pathname);
  return (
    <IntelLink
      href={item.href}
      aria-current={active ? "page" : undefined}
      className={cn(
        "-ml-px block border-l py-1 text-sm transition-colors",
        nested ? "pl-6 text-[13px]" : "pl-3",
        active ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
      )}
    >
      {item.label}
    </IntelLink>
  );
}

function Shell({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const strip = useRef<HTMLDivElement>(null);

  // keep the current section visible in the phone tab strip
  useEffect(() => {
    strip.current?.querySelector<HTMLElement>("[aria-current=page]")?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [pathname]);

  return (
    <div className="lg:grid lg:grid-cols-[184px_minmax(0,1fr)] lg:gap-12">
      {/* phone / tablet: the domain, then one scrollable strip of tabs */}
      <div className="lg:hidden">
        <DomainPicker className="mt-6 max-w-xs" />
        <nav aria-label="Intelligence sections">
          <div ref={strip} className="scrollbar-none -mx-5 mt-4 flex gap-5 overflow-x-auto border-b hairline px-5 sm:-mx-8 sm:px-8">
            {FLAT.map((t) => {
              const active = isActive(t.href, pathname);
              return (
                <IntelLink
                  key={t.href}
                  href={t.href}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "-mb-px shrink-0 border-b py-2.5 text-sm transition-colors",
                    active ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
                  )}
                >
                  {t.label}
                </IntelLink>
              );
            })}
          </div>
        </nav>
      </div>

      {/* desktop: a left rail in engine order, with the system group last */}
      <nav aria-label="Intelligence sections" className="sticky top-20 hidden self-start pt-12 lg:block">
        <p className="font-display text-2xl leading-none">Intelligence</p>
        <p className="eyebrow mt-1.5">operator console</p>
        <DomainPicker className="mt-5" />
        <ul className="mt-6 border-l hairline">
          {CONSOLE_NAV.map((t) => (
            <li key={t.href}>
              <RailLink item={t} pathname={pathname} />
              {t.children?.map((c) => <RailLink key={c.href} item={c} pathname={pathname} nested />)}
            </li>
          ))}
        </ul>
        <p className="eyebrow mb-1.5 mt-5">System</p>
        <ul className="border-l hairline">
          {SYSTEM_NAV.map((t) => <li key={t.href}><RailLink item={t} pathname={pathname} /></li>)}
        </ul>
        <Link href="/admin" className="mt-8 block text-xs text-muted-foreground hover:text-foreground">← Admin control room</Link>
      </nav>

      <div className="min-w-0 pt-8 lg:pt-12">{children}</div>
    </div>
  );
}

function ShellSkeleton() {
  return (
    <Container className="space-y-4 pt-14">
      <Skeleton className="h-4 w-32" />
      <Skeleton className="h-14 w-80 max-w-full" />
      <Skeleton className="h-64 w-full" />
    </Container>
  );
}

/** Admin gate + domain context + navigation for every /intel page. */
export function ConsoleShell({ children }: { children: React.ReactNode }) {
  const { user, loading } = useSession();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !user) router.replace(`/login?next=${encodeURIComponent(pathname)}`);
  }, [loading, user, router, pathname]);

  if (loading || !user) return <ShellSkeleton />;
  if (!user.is_admin) {
    return (
      <Container className="pt-14">
        <EmptyState
          title="Admins only"
          body="The Intelligence console is limited to administrator accounts. Your own intelligence (taste drift, the strategy behind your list) is under My intelligence."
          action={
            <div className="flex flex-wrap justify-center gap-2">
              <Button asChild><Link href="/me/intelligence">My intelligence</Link></Button>
              <Button asChild variant="outline"><Link href="/home">Back to your programme</Link></Button>
            </div>
          }
        />
      </Container>
    );
  }

  return (
    <Container>
      <Suspense fallback={<ShellSkeleton />}>
        <DomainProvider>
          <Shell>{children}</Shell>
        </DomainProvider>
      </Suspense>
    </Container>
  );
}
