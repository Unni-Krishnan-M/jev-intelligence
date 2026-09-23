/**
 * Content-Security-Policy for every HTML response (set per request in src/proxy.ts), following the
 * Next.js 16 CSP guide (node_modules/next/dist/docs/01-app/02-guides/content-security-policy.md).
 *
 * Scripts need the per-request nonce: Next.js reads it from this header while rendering and puts it on
 * its own scripts, and 'strict-dynamic' lets those load the page chunks. No host allow-list and no
 * 'unsafe-inline' for scripts. Styles keep 'unsafe-inline': Radix, Recharts and sonner set inline
 * style attributes and elements that cannot carry a nonce, and inline CSS cannot run script.
 * Development adds 'unsafe-eval' (React's dev tooling needs it; production does not).
 */
export type CspOptions = { dev?: boolean; https?: boolean };

export function buildCsp(nonce: string, { dev = false, https = false }: CspOptions = {}): string {
  const directives = [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${dev ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' blob: data:",
    "font-src 'self'",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
    ...(https ? ["upgrade-insecure-requests"] : []),
  ];
  return directives.join("; ");
}

/** A fresh, unguessable nonce for one response (128 bits, base64). */
export function newNonce(): string {
  return btoa(crypto.randomUUID());
}

export const HSTS = "max-age=31536000; includeSubDomains";
