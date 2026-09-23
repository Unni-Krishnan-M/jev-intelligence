"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useSession } from "@/hooks/use-session";
import { api, errorMessage } from "@/lib/api";
import { safeNextPath } from "@/lib/safe-redirect";
import type { User } from "@/lib/types";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const router = useRouter();
  const params = useSearchParams();
  const { refresh } = useSession();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    const f = new FormData(e.currentTarget);
    const body = {
      email: String(f.get("email") ?? "").trim(),
      password: String(f.get("password") ?? ""),
      ...(mode === "register" ? { display_name: String(f.get("display_name") ?? "").trim() } : {}),
    };
    if (mode === "register" && body.password.length < 8) {
      setError("Use at least 8 characters for your password.");
      return;
    }
    setBusy(true);
    try {
      const res = await api<{ user: User }>(`/auth/${mode}`, { json: body });
      await refresh(res.user, { revalidate: false });
      const safeNext = safeNextPath(params.get("next"), window.location.origin);
      router.replace(!res.user.onboarding_completed ? "/onboarding" : (safeNext ?? "/home"));
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  const register = mode === "register";
  return (
    <form onSubmit={onSubmit} className="space-y-5" noValidate={false}>
      {register && (
        <div className="space-y-2">
          <Label htmlFor="display_name">Name</Label>
          <Input id="display_name" name="display_name" autoComplete="nickname" required maxLength={80} placeholder="What should we call you?" />
        </div>
      )}
      <div className="space-y-2">
        <Label htmlFor="email">Email</Label>
        <Input id="email" name="email" type="email" autoComplete="email" required placeholder="you@example.com" />
      </div>
      <div className="space-y-2">
        <Label htmlFor="password">Password</Label>
        <Input id="password" name="password" type="password" autoComplete={register ? "new-password" : "current-password"} required minLength={register ? 8 : 1} maxLength={128} />
        {register && <p className="text-xs text-muted-foreground">At least 8 characters. Stored as an Argon2id hash.</p>}
      </div>
      {error && <p role="alert" className="rounded-md border border-destructive/40 bg-destructive/5 px-3 py-2 text-sm">{error}</p>}
      <Button type="submit" size="lg" className="w-full" disabled={busy}>
        {busy ? "One moment…" : register ? "Create account" : "Log in"}
      </Button>
      <p className="text-center text-sm text-muted-foreground">
        {register ? <>Already have an account? <Link href="/login" className="text-foreground underline underline-offset-4">Log in</Link></>
          : <>New to JEV? <Link href="/register" className="text-foreground underline underline-offset-4">Create an account</Link></>}
      </p>
    </form>
  );
}

export function AuthShell({ title, kicker, children }: { title: string; kicker: string; children: React.ReactNode }) {
  return (
    <div className="mx-auto grid min-h-[calc(100vh-3.5rem)] max-w-[1320px] items-center gap-10 px-5 py-12 sm:px-8 lg:grid-cols-2">
      <div className="hidden lg:block">
        <p className="eyebrow">{kicker}</p>
        <p className="font-display mt-4 max-w-md text-[56px] leading-[0.95]">
          A good programme starts with <em className="text-primary">one honest rating.</em>
        </p>
        <p className="mt-6 max-w-sm text-sm leading-relaxed text-ink-2">
          Your ratings, favourites and watches stay in your JEV account and are used only to rank films for you. You can
          mark anything &ldquo;not for me&rdquo; at any time.
        </p>
      </div>
      <div className="mx-auto w-full max-w-sm">
        <h1 className="font-display text-5xl">{title}</h1>
        <div className="mt-8">{children}</div>
      </div>
    </div>
  );
}
