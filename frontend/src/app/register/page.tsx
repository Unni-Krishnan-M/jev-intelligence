import type { Metadata } from "next";
import { Suspense } from "react";

import { AuthForm, AuthShell } from "@/components/jev/auth/auth-form";

export const metadata: Metadata = { title: "Create account" };

export default function RegisterPage() {
  return (
    <AuthShell title="Take a seat." kicker="Create account">
      <Suspense><AuthForm mode="register" /></Suspense>
    </AuthShell>
  );
}
