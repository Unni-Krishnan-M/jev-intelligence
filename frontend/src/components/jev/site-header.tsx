"use client";

import { ChevronDown, Menu, Moon, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import { useState, useSyncExternalStore } from "react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { useSession } from "@/hooks/use-session";
import { cn } from "@/lib/utils";

import { SearchBox } from "./search-box";

/** The member app: one domain of the engine (docs/platform.md §9). */
export const MOVIES_NAV = [
  { href: "/home", label: "Tonight" },
  { href: "/recommendations", label: "For you" },
  { href: "/discover", label: "Discover" },
  { href: "/favorites", label: "Library" },
];

const MOVIE_PREFIXES = ["/home", "/recommendations", "/discover", "/favorites", "/movies", "/history", "/profile", "/onboarding"];

function isUnder(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(`${href}/`);
}

/** Where "Intelligence" goes: the console for admins, your own intelligence for everyone else. */
export function intelligenceHref(user: { is_admin: boolean } | null): string | null {
  if (!user) return null;
  return user.is_admin ? "/intel" : "/me/intelligence";
}

const TAB = "relative rounded px-2.5 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground";
const TAB_ACTIVE = "text-foreground after:absolute after:inset-x-2.5 after:-bottom-[13px] after:h-px after:bg-primary";

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  // false during SSR/hydration, true on the client: avoids a theme-icon hydration mismatch
  const mounted = useSyncExternalStore(() => () => {}, () => true, () => false);
  const dark = !mounted || resolvedTheme === "dark";
  return (
    <Button variant="ghost" size="icon" aria-label={dark ? "Switch to paper (light) theme" : "Switch to dark theme"} onClick={() => setTheme(dark ? "light" : "dark")}>
      {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </Button>
  );
}

export function SiteHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const { user, logout } = useSession();
  const [sheet, setSheet] = useState(false);
  const intel = intelligenceHref(user);
  const intelActive = isUnder(pathname, "/intel") || isUnder(pathname, "/me");
  const moviesActive = MOVIE_PREFIXES.some((p) => isUnder(pathname, p));
  const close = () => setSheet(false);

  return (
    <header className="glass-bar sticky top-0 z-40 border-b hairline">
      <div className="mx-auto flex h-14 max-w-[1320px] items-center gap-4 px-5 sm:px-8">
        <Link href="/" className="flex items-baseline gap-2" aria-label="JEV, the decision and early-warning engine">
          <span className="font-display text-[26px] leading-none tracking-tight">JEV</span>
          <span className="hidden font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground sm:inline">decision engine</span>
        </Link>

        <nav className="ml-4 hidden items-center gap-1 md:flex" aria-label="Main">
          {intel && (
            <Link href={intel} aria-current={intelActive ? "page" : undefined} className={cn(TAB, intelActive && TAB_ACTIVE)}>
              Intelligence
            </Link>
          )}
          {user ? (
            <DropdownMenu>
              <DropdownMenuTrigger className={cn(TAB, "inline-flex items-center gap-1 outline-none focus-visible:ring-2 focus-visible:ring-ring", moviesActive && TAB_ACTIVE)}>
                Movies <ChevronDown className="size-3.5" aria-hidden />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-56">
                <DropdownMenuLabel className="eyebrow font-normal">Movies · first domain</DropdownMenuLabel>
                {MOVIES_NAV.map((n) => (
                  <DropdownMenuItem key={n.href} asChild>
                    <Link href={n.href} aria-current={isUnder(pathname, n.href) ? "page" : undefined} className={cn(isUnder(pathname, n.href) && "font-medium")}>
                      {n.label}
                    </Link>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <Link href="/discover" aria-current={isUnder(pathname, "/discover") ? "page" : undefined} className={cn(TAB, isUnder(pathname, "/discover") && TAB_ACTIVE)}>
              Discover films
            </Link>
          )}
          {user?.is_admin && (
            <Link href="/admin" aria-current={isUnder(pathname, "/admin") ? "page" : undefined} className={cn(TAB, isUnder(pathname, "/admin") && TAB_ACTIVE)}>
              Admin
            </Link>
          )}
        </nav>

        <div className="ml-auto flex items-center gap-1.5">
          <SearchBox className="hidden w-64 lg:block xl:w-80" />
          <ThemeToggle />
          {user ? (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="hidden gap-2 px-2 md:inline-flex" aria-label="Account menu">
                  <span className="grid size-7 place-items-center rounded-full bg-secondary font-display text-sm">
                    {user.display_name.slice(0, 1).toUpperCase()}
                  </span>
                  <span className="max-w-28 truncate text-sm">{user.display_name}</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-52">
                <DropdownMenuLabel className="font-normal">
                  <p className="text-sm">{user.display_name}</p>
                  <p className="truncate text-xs text-muted-foreground">{user.email}</p>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={() => router.push("/me/intelligence")}>My intelligence</DropdownMenuItem>
                <DropdownMenuItem onSelect={() => router.push("/profile")}>Taste profile</DropdownMenuItem>
                <DropdownMenuItem onSelect={() => router.push("/history")}>History</DropdownMenuItem>
                <DropdownMenuItem onSelect={() => router.push("/favorites")}>Favourites</DropdownMenuItem>
                {user.is_admin && <DropdownMenuItem onSelect={() => router.push("/admin")}>Admin</DropdownMenuItem>}
                {user.is_admin && <DropdownMenuItem onSelect={() => router.push("/intel")}>Intelligence console</DropdownMenuItem>}
                <DropdownMenuSeparator />
                <DropdownMenuItem onSelect={async () => { await logout(); router.push("/"); }}>Log out</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          ) : (
            <div className="hidden items-center gap-1 md:flex">
              <Button variant="ghost" size="sm" asChild><Link href="/login">Log in</Link></Button>
              <Button size="sm" asChild><Link href="/register">Create account</Link></Button>
            </div>
          )}

          <Sheet open={sheet} onOpenChange={setSheet}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="md:hidden" aria-label="Open menu"><Menu className="size-5" /></Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-[86vw] max-w-sm overflow-y-auto">
              <SheetHeader><SheetTitle className="font-display text-2xl font-normal">JEV</SheetTitle></SheetHeader>
              <div className="flex flex-col gap-1 px-4 pb-6">
                <SearchBox className="mb-3" onNavigate={close} />
                {user && (
                  <SheetGroup label="Intelligence">
                    {user.is_admin && <SheetLink href="/intel" pathname={pathname} onClick={close}>Console</SheetLink>}
                    <SheetLink href="/me/intelligence" pathname={pathname} onClick={close}>My intelligence</SheetLink>
                  </SheetGroup>
                )}
                <SheetGroup label="Movies">
                  {(user ? MOVIES_NAV : MOVIES_NAV.filter((n) => n.href === "/discover")).map((n) => (
                    <SheetLink key={n.href} href={n.href} pathname={pathname} onClick={close}>{n.label}</SheetLink>
                  ))}
                </SheetGroup>
                {user && (
                  <SheetGroup label="Account">
                    <SheetLink href="/profile" pathname={pathname} onClick={close}>Taste profile</SheetLink>
                    <SheetLink href="/history" pathname={pathname} onClick={close}>History</SheetLink>
                    {user.is_admin && <SheetLink href="/admin" pathname={pathname} onClick={close}>Admin</SheetLink>}
                  </SheetGroup>
                )}
                <div className="mt-3 border-t hairline pt-3">
                  {user ? (
                    <Button variant="outline" className="w-full" onClick={async () => { close(); await logout(); router.push("/"); }}>Log out</Button>
                  ) : (
                    <div className="grid grid-cols-2 gap-2">
                      <Button variant="outline" asChild><Link href="/login" onClick={close}>Log in</Link></Button>
                      <Button asChild><Link href="/register" onClick={close}>Sign up</Link></Button>
                    </div>
                  )}
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}

function SheetGroup({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-2 border-t hairline pt-3 first:mt-0">
      <p className="eyebrow mb-1 px-2">{label}</p>
      <div className="flex flex-col">{children}</div>
    </div>
  );
}

function SheetLink({ href, pathname, onClick, children }: { href: string; pathname: string; onClick: () => void; children: React.ReactNode }) {
  const active = isUnder(pathname, href);
  return (
    <Link
      href={href}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={cn("rounded border-l-2 px-2 py-2.5 text-base hover:bg-accent", active ? "border-primary" : "border-transparent")}
    >
      {children}
    </Link>
  );
}
