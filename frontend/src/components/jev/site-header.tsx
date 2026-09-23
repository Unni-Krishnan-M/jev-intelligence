"use client";

import { Menu, Moon, Sun } from "lucide-react";
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

const NAV = [
  { href: "/home", label: "Tonight" },
  { href: "/recommendations", label: "For you" },
  { href: "/discover", label: "Discover" },
  { href: "/favorites", label: "Library" },
];

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
  const nav = user ? [...NAV, ...(user.is_admin ? [{ href: "/admin", label: "Admin" }, { href: "/intel", label: "Intelligence" }] : [])] : [{ href: "/discover", label: "Discover" }];

  return (
    <header className="glass-bar sticky top-0 z-40 border-b hairline">
      <div className="mx-auto flex h-14 max-w-[1320px] items-center gap-4 px-5 sm:px-8">
        <Link href={user ? "/home" : "/"} className="flex items-baseline gap-2" aria-label="JEV home">
          <span className="font-display text-[26px] leading-none tracking-tight">JEV</span>
          <span className="hidden font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground sm:inline">programme</span>
        </Link>

        <nav className="ml-4 hidden items-center gap-1 md:flex" aria-label="Main">
          {nav.map((n) => {
            const active = pathname === n.href || (n.href !== "/" && pathname.startsWith(`${n.href}/`));
            return (
              <Link
                key={n.href}
                href={n.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "relative rounded px-2.5 py-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground",
                  active && "text-foreground after:absolute after:inset-x-2.5 after:-bottom-[13px] after:h-px after:bg-primary",
                )}
              >
                {n.label}
              </Link>
            );
          })}
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
                <DropdownMenuItem onSelect={() => router.push("/profile")}>Taste profile</DropdownMenuItem>
                <DropdownMenuItem onSelect={() => router.push("/history")}>History</DropdownMenuItem>
                <DropdownMenuItem onSelect={() => router.push("/favorites")}>Favourites</DropdownMenuItem>
                {user.is_admin && <DropdownMenuItem onSelect={() => router.push("/admin")}>Admin</DropdownMenuItem>}
                {user.is_admin && <DropdownMenuItem onSelect={() => router.push("/intel")}>Intelligence</DropdownMenuItem>}
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
            <SheetContent side="right" className="w-[86vw] max-w-sm">
              <SheetHeader><SheetTitle className="font-display text-2xl font-normal">JEV</SheetTitle></SheetHeader>
              <div className="flex flex-col gap-1 px-4">
                <SearchBox className="mb-3" onNavigate={() => setSheet(false)} />
                {[...nav, ...(user ? [{ href: "/profile", label: "Taste profile" }, { href: "/history", label: "History" }] : [])].map((n) => (
                  <Link key={n.href} href={n.href} onClick={() => setSheet(false)} className="rounded px-2 py-2.5 text-base hover:bg-accent">
                    {n.label}
                  </Link>
                ))}
                <div className="mt-3 border-t hairline pt-3">
                  {user ? (
                    <Button variant="outline" className="w-full" onClick={async () => { setSheet(false); await logout(); router.push("/"); }}>Log out</Button>
                  ) : (
                    <div className="grid grid-cols-2 gap-2">
                      <Button variant="outline" asChild><Link href="/login" onClick={() => setSheet(false)}>Log in</Link></Button>
                      <Button asChild><Link href="/register" onClick={() => setSheet(false)}>Sign up</Link></Button>
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
