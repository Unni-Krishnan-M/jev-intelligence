import { NextResponse, type NextRequest } from "next/server";

import { buildCsp, HSTS, newNonce } from "@/lib/csp";

/**
 * Runs before every page request:
 * 1. Content-Security-Policy with a per-request nonce (src/lib/csp.ts). Next.js reads the nonce from
 *    the request's CSP header and adds it to its scripts, so pages render dynamically (the root
 *    layout reads the nonce too, for the theme script).
 * 2. Strict-Transport-Security, only when the site is served over HTTPS (JEV_HTTPS=true), so local
 *    http development is unaffected.
 * 3. Optimistic route guard: only checks that a session cookie exists. The API still verifies the
 *    JWT on every request, and client pages handle a 401 by sending the user to /login.
 */
const PROTECTED = ["/home", "/onboarding", "/recommendations", "/profile", "/history", "/favorites", "/admin", "/intel", "/me"];
const AUTH_PAGES = ["/login", "/register"];
const COOKIE = "jev_session";

function secured(response: NextResponse, csp: string, https: boolean): NextResponse {
  response.headers.set("Content-Security-Policy", csp);
  if (https) response.headers.set("Strict-Transport-Security", HSTS);
  return response;
}

export function proxy(request: NextRequest) {
  const https = process.env.JEV_HTTPS === "true";
  const nonce = newNonce();
  const csp = buildCsp(nonce, { dev: process.env.NODE_ENV === "development", https });

  const { pathname, search } = request.nextUrl;
  const hasSession = Boolean(request.cookies.get(COOKIE)?.value);
  if (!hasSession && PROTECTED.some((p) => pathname === p || pathname.startsWith(`${p}/`))) {
    const url = new URL("/login", request.url);
    url.searchParams.set("next", pathname + search);
    return secured(NextResponse.redirect(url), csp, https);
  }
  if (hasSession && AUTH_PAGES.includes(pathname)) {
    return secured(NextResponse.redirect(new URL("/home", request.url)), csp, https);
  }

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("Content-Security-Policy", csp);
  return secured(NextResponse.next({ request: { headers: requestHeaders } }), csp, https);
}

export const config = {
  // every page; not the API rewrite, build assets or the favicon
  matcher: ["/((?!api/|_next/static|_next/image|favicon.ico).*)"],
};
