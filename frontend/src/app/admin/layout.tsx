"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";

import { Container, EmptyState } from "@/components/jev/states";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { useSession } from "@/hooks/use-session";
import { cn } from "@/lib/utils";

const TABS = [
  { href: "/admin", label: "Overview" },
  { href: "/admin/models", label: "Models" },
  { href: "/admin/experiments", label: "Experiments" },
];

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const { user, loading } = useSession();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (!loading && !user) router.replace("/login?next=/admin");
  }, [loading, user, router]);

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
          body="The model and experiment dashboard is limited to administrator accounts."
          action={<Button asChild variant="outline"><Link href="/home">Back to your programme</Link></Button>}
        />
      </Container>
    );
  }

  return (
    <div>
      <Container>
        <nav aria-label="Admin sections" className="mt-8 flex gap-6 border-b hairline">
          {TABS.map((t) => {
            const active = t.href === "/admin" ? pathname === "/admin" : pathname.startsWith(t.href);
            return (
              <Link
                key={t.href}
                href={t.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "-mb-px border-b py-2.5 text-sm transition-colors",
                  active ? "border-primary text-foreground" : "border-transparent text-muted-foreground hover:text-foreground",
                )}
              >
                {t.label}
              </Link>
            );
          })}
        </nav>
      </Container>
      {children}
    </div>
  );
}
