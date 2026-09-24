/**
 * API client. All calls go to /api/* and are proxied to FastAPI by next.config rewrites, so the
 * httpOnly session cookie stays same-origin. Mutations carry the CSRF header the backend requires
 * for cookie-authenticated writes.
 */

export class ApiError extends Error {
  constructor(public status: number, message: string, public requestId?: string, public details?: unknown) {
    super(message);
  }
}

const BASE = "/api";

export async function api<T>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const { json, headers, ...rest } = init;
  const method = (rest.method ?? (json !== undefined ? "POST" : "GET")).toUpperCase();
  const res = await fetch(`${BASE}${path}`, {
    ...rest,
    method,
    credentials: "same-origin",
    headers: {
      Accept: "application/json",
      ...(json !== undefined ? { "Content-Type": "application/json" } : {}),
      ...(method !== "GET" ? { "X-JEV-CSRF": "1" } : {}),
      ...headers,
    },
    body: json !== undefined ? JSON.stringify(json) : rest.body,
  });
  if (res.status === 204) return undefined as T;
  let body: unknown = null;
  const text = await res.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!res.ok) {
    const b = (body ?? {}) as { detail?: unknown; request_id?: string; errors?: unknown; blockers?: unknown };
    // governance answers 409 with {detail, blockers}; older builds nested {message, blockers} in detail
    const obj = b.detail && typeof b.detail === "object" && !Array.isArray(b.detail) ? (b.detail as { message?: unknown; blockers?: unknown }) : null;
    const detail = typeof b.detail === "string" ? b.detail : typeof obj?.message === "string" ? obj.message : res.statusText || "request failed";
    const blockers = Array.isArray(b.blockers) ? b.blockers : obj && Array.isArray(obj.blockers) ? obj.blockers : null;
    throw new ApiError(res.status, detail, b.request_id, blockers ?? b.errors);
  }
  return body as T;
}

export const swrFetcher = <T,>(path: string) => api<T>(path);

export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 422 && Array.isArray(err.details) && err.details.length) {
      const first = err.details[0] as { msg?: string; loc?: unknown[] };
      const field = Array.isArray(first.loc) ? String(first.loc[first.loc.length - 1]) : "";
      return `${field ? `${field}: ` : ""}${first.msg ?? err.message}`;
    }
    if (err.status === 429) return "Too many requests — give it a minute.";
    if (err.status === 503) return "The recommendation model isn't loaded yet. Train a model and restart the API.";
    return err.message;
  }
  return "Something went wrong. Check your connection and try again.";
}

export function qs(params: Record<string, string | number | boolean | string[] | null | undefined>): string {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "" || v === false) continue;
    if (Array.isArray(v)) v.forEach((x) => sp.append(k, x));
    else sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
}
