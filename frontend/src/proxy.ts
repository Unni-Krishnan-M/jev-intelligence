import { NextResponse, type NextRequest } from "next/server";

/**
 * Optimistic route guard: only checks that a session cookie exists. The API still verifies the
 * JWT on every request, and client pages handle a 401 by sending the user to /login.
 */
const PROTECTED = ["/home", "/onboarding", "/recommendations", "/profile", "/history", "/favorites", "/admin", "/intel"];
const AUTH_PAGES = ["/login", "/register"];
const COOKIE = "jev_session";

export function proxy(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  const hasSession = Boolean(request.cookies.get(COOKIE)?.value);
  if (!hasSession && PROTECTED.some((p) => pathname === p || pathname.startsWith(`${p}/`))) {
    const url = new URL("/login", request.url);
    url.searchParams.set("next", pathname + search);
    return NextResponse.redirect(url);
  }
  if (hasSession && AUTH_PAGES.includes(pathname)) {
    return NextResponse.redirect(new URL("/home", request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/home", "/onboarding", "/recommendations", "/profile", "/history", "/favorites", "/admin/:path*", "/intel", "/intel/:path*", "/login", "/register"],
};
