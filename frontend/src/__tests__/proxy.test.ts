// @vitest-environment node
import { createHmac } from "node:crypto";

import { NextRequest } from "next/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CLIENT_HEADER, clientIp, signClient } from "@/lib/client-sign";
import { config, proxy } from "@/proxy";

/** The request headers NextResponse.next({ request: { headers } }) forwards upstream. */
function forwarded(res: Response, name: string): string | null {
  return res.headers.get(`x-middleware-request-${name}`);
}

function apiRequest(headers: Record<string, string> = {}) {
  return new NextRequest("http://localhost:3000/api/intel/status", { headers });
}

afterEach(() => vi.unstubAllEnvs());

describe("signed client address (docs/SECURITY_AUDIT_PHASE2.md §5)", () => {
  it("signs v1|ip|ts with HMAC-SHA256 in the v1:<ip>:<ts>:<hex> format the API verifies", async () => {
    const v = await signClient("203.0.113.7", "s3cret", 1790000000);
    const expected = createHmac("sha256", "s3cret").update("v1|203.0.113.7|1790000000").digest("hex");
    expect(v).toBe(`v1:203.0.113.7:1790000000:${expected}`);
    // IPv6 keeps its colons: the verifier splits the ts and mac from the right
    const v6 = await signClient("2001:db8::1", "s3cret", 1790000000);
    const parts = v6.split(":");
    expect(parts.at(-1)).toHaveLength(64);
    expect(parts.at(-2)).toBe("1790000000");
    expect(v6.slice(3, v6.length - 1 - 64 - 1 - 10)).toBe("2001:db8::1");
  });

  it("takes the right-most X-Forwarded-For hop, or the configured edge header", () => {
    expect(clientIp(new Headers({ "x-forwarded-for": "1.1.1.1, 10.0.0.2" }))).toBe("10.0.0.2");
    vi.stubEnv("JEV_CLIENT_IP_HEADER", "x-real-ip");
    expect(clientIp(new Headers({ "x-real-ip": "198.51.100.4", "x-forwarded-for": "1.1.1.1" }))).toBe("198.51.100.4");
    expect(clientIp(new Headers())).toBeNull();
  });

  it("drops a client-supplied header and signs the address when the secret is set", async () => {
    vi.stubEnv("JEV_PROXY_SECRET", "s3cret");
    const res = await proxy(apiRequest({ [CLIENT_HEADER]: "v1:6.6.6.6:1:forged", "x-forwarded-for": "203.0.113.7" }));
    const v = forwarded(res, CLIENT_HEADER);
    expect(v).toMatch(/^v1:203\.0\.113\.7:\d+:[0-9a-f]{64}$/);
    const [, ip, ts, mac] = v!.split(":");
    expect(mac).toBe(createHmac("sha256", "s3cret").update(`v1|${ip}|${ts}`).digest("hex"));
  });

  it("forwards nothing extra without a secret, and still strips the client's value", async () => {
    vi.stubEnv("JEV_PROXY_SECRET", "");
    const res = await proxy(apiRequest({ [CLIENT_HEADER]: "v1:6.6.6.6:1:forged", "x-forwarded-for": "203.0.113.7" }));
    expect(forwarded(res, CLIENT_HEADER)).toBeNull();
    expect(res.headers.get("x-middleware-override-headers") ?? "").not.toContain(CLIENT_HEADER);
  });

  it("gives /api/* no CSP rewrite", async () => {
    const res = await proxy(apiRequest());
    expect(res.headers.get("Content-Security-Policy")).toBeNull();
    expect(forwarded(res, "x-nonce")).toBeNull();
  });
});

describe("page guard under the wider matcher", () => {
  it("matches the API rewrite and pages, not build assets", () => {
    const re = new RegExp(`^${config.matcher[0]}$`);
    expect(re.test("/api/intel/status")).toBe(true);
    expect(re.test("/intel/ops/models")).toBe(true);
    expect(re.test("/_next/static/chunks/a.js")).toBe(false);
    expect(re.test("/favicon.ico")).toBe(false);
  });

  it("still redirects a protected page without a session, and signed-in users away from /login", async () => {
    const res = await proxy(new NextRequest("http://localhost:3000/intel/ops/models?x=1"));
    expect(res.status).toBe(307);
    expect(new URL(res.headers.get("location")!).searchParams.get("next")).toBe("/intel/ops/models?x=1");
    expect(res.headers.get("Content-Security-Policy")).toContain("nonce-");
    const back = await proxy(new NextRequest("http://localhost:3000/login", { headers: { cookie: "jev_session=abc" } }));
    expect(new URL(back.headers.get("location")!).pathname).toBe("/home");
    const page = await proxy(new NextRequest("http://localhost:3000/intel", { headers: { cookie: "jev_session=abc" } }));
    expect(page.status).toBe(200);
    expect(forwarded(page, "x-nonce")).toBeTruthy();
  });
});
