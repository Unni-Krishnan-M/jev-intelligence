import { describe, expect, it } from "vitest";

import { buildCsp, newNonce } from "@/lib/csp";

describe("buildCsp", () => {
  it("allows scripts only by nonce in production", () => {
    const csp = buildCsp("abc123");
    expect(csp).toContain("script-src 'self' 'nonce-abc123' 'strict-dynamic'");
    expect(csp).not.toContain("unsafe-eval");
    expect(csp).not.toMatch(/script-src[^;]*unsafe-inline/);
    for (const d of ["default-src 'self'", "object-src 'none'", "base-uri 'self'", "frame-ancestors 'none'",
      "form-action 'self'", "connect-src 'self'", "font-src 'self'"]) {
      expect(csp).toContain(d);
    }
  });

  it("adds unsafe-eval only in development and upgrade-insecure-requests only over https", () => {
    expect(buildCsp("n", { dev: true })).toContain("'unsafe-eval'");
    expect(buildCsp("n")).not.toContain("upgrade-insecure-requests");
    expect(buildCsp("n", { https: true })).toContain("upgrade-insecure-requests");
  });

  it("uses a fresh nonce each time", () => {
    const a = newNonce();
    expect(a).not.toBe(newNonce());
    expect(a).toMatch(/^[A-Za-z0-9+/=]{40,}$/);
  });
});
