import { NextResponse, type NextRequest } from "next/server";

import { CLIENT_HEADER, clientIp, signClient } from "@/lib/client-sign";
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
 * 4. /api/* (the rewrite to FastAPI): a client-sent x-jev-client is dropped and, when JEV_PROXY_SECRET
 *    is set, replaced by the signed client address (src/lib/client-sign.ts). No CSP on API calls.
 */
const PROTECTED = ["/home", "/onboarding", "/recommendations", "/profile", "/history", "/favorites", "/admin", "/intel", "/me"];
const AUTH_PAGES = ["/login", "/register"];
const COOKIE = "jev_session";

function secured(response: NextResponse, csp: string, https: boolean): NextResponse {
  response.headers.set("Content-Security-Policy", csp);
  if (https) response.headers.set("Strict-Transport-Security", HSTS);
  return response;
}

export async function proxy(request: NextRequest) {
  if (request.nextUrl.pathname.startsWith("/api/")) {
    const headers = new Headers(request.headers);
    headers.delete(CLIENT_HEADER); // never forward a client-supplied value
    const secret = process.env.JEV_PROXY_SECRET;
    const ip = secret ? clientIp(request.headers) : null;
    if (secret && ip) headers.set(CLIENT_HEADER, await signClient(ip, secret));
    return NextResponse.next({ request: { headers } });
  }

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
  // pages AND the API rewrite; not build assets or the favicon
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
