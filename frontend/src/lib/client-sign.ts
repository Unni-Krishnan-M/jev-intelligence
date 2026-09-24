/**
 * Signed client address for /api/* requests (docs/SECURITY_AUDIT_PHASE2.md §5). The web proxy is the
 * only party that knows JEV_PROXY_SECRET, so the API can trust `x-jev-client` for per-IP limits;
 * backend/jev_api/security.verify_client_address checks the same `v1:<ip>:<ts>:<hex>` format.
 */

export const CLIENT_HEADER = "x-jev-client";

export function clientIp(headers: Headers): string | null {
  // an edge proxy that OVERWRITES this header (e.g. nginx `proxy_set_header X-Real-IP $remote_addr`) is
  // authoritative; otherwise the right-most X-Forwarded-For hop (Next sets it from the socket when absent)
  const edge = process.env.JEV_CLIENT_IP_HEADER;
  const raw = edge ? headers.get(edge) : headers.get("x-forwarded-for")?.split(",").pop();
  const ip = raw?.trim() ?? "";
  return ip && ip.length <= 45 ? ip : null;
}

export async function signClient(ip: string, secret: string, ts: number = Math.floor(Date.now() / 1000)): Promise<string> {
  const key = await crypto.subtle.importKey("raw", new TextEncoder().encode(secret), { name: "HMAC", hash: "SHA-256" }, false, ["sign"]);
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(`v1|${ip}|${ts}`));
  const hex = Array.from(new Uint8Array(mac), (b) => b.toString(16).padStart(2, "0")).join("");
  return `v1:${ip}:${ts}:${hex}`;
}
