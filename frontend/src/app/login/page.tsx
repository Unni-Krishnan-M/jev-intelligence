import type { Metadata } from "next";
import { Suspense } from "react";

import { AuthForm, AuthShell } from "@/components/jev/auth/auth-form";

export const metadata: Metadata = { title: "Log in" };

export default function LoginPage() {
  return (
    <AuthShell title="Welcome back." kicker="Log in">
      <Suspense><AuthForm mode="login" /></Suspense>
    </AuthShell>
  );
}
