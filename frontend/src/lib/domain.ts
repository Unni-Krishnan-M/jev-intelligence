/**
 * The console's domain dimension (docs/platform.md §8–§9): which domain adapter the operator is
 * looking at, how it travels in `?domain=`, and which sections that domain produces. Pure
 * functions only; the React side lives in components/jev/intel/domain-context.tsx.
 */

import type { DomainCapabilities, DomainCapability, DomainInfo } from "./intel-types";

export const DEFAULT_DOMAIN = "movie";
export const DOMAIN_PARAM = "domain";

/** Adapter keys look like "movie" or "generic:us-unemployment". Anything else falls back to movie. */
const DOMAIN_RE = /^[a-z0-9][a-z0-9:_.-]{0,63}$/i;

export function normalizeDomain(raw: string | null | undefined): string {
  const v = (raw ?? "").trim();
  return v && DOMAIN_RE.test(v) ? v : DEFAULT_DOMAIN;
}

function splitHash(path: string): [string, string] {
  const i = path.indexOf("#");
  return i < 0 ? [path, ""] : [path.slice(0, i), path.slice(i)];
}

function setParam(path: string, key: string, value: string | null): string {
  const [base, hash] = splitHash(path);
  const q = base.indexOf("?");
  const pathname = q < 0 ? base : base.slice(0, q);
  const sp = new URLSearchParams(q < 0 ? "" : base.slice(q + 1));
  if (value === null) sp.delete(key);
  else sp.set(key, value);
  const s = sp.toString();
  return `${pathname}${s ? `?${s}` : ""}${hash}`;
}

/**
 * An /intel/* API path with `domain=` set (replacing any existing value). Every console read and
 * write carries it, including the default, so the request says which adapter it means.
 */
export function withDomain(path: string, domain: string): string {
  return setParam(path, DOMAIN_PARAM, normalizeDomain(domain));
}

/** Is this an in-app console link (/intel or /intel/…)? */
export function isConsolePath(href: string): boolean {
  return href === "/intel" || href.startsWith("/intel/") || href.startsWith("/intel?") || href.startsWith("/intel#");
}

/**
 * A console link that keeps the chosen domain. The default domain is left out of the URL (it is
 * what the API assumes anyway); links outside the console are returned unchanged.
 */
export function intelHref(href: string, domain: string): string {
  if (!isConsolePath(href)) return href;
  const d = normalizeDomain(domain);
  return setParam(href, DOMAIN_PARAM, d === DEFAULT_DOMAIN ? null : d);
}

/**
 * Where the domain picker goes: the same section in the other domain. A detail page (a warning,
 * decision or series id) belongs to one domain, so it falls back to its section's list, and
 * section filters (batch ids, series ids) are dropped for the same reason.
 */
export function switchDomainHref(pathname: string, domain: string): string {
  const parts = pathname.split("/").filter(Boolean);
  let path = parts[0] === "intel" ? `/${parts.slice(0, 2).join("/")}` : "/intel";
  if (parts[1] === "series") path = "/intel/trends";
  return intelHref(path, domain);
}

// ---- capabilities ----------------------------------------------------------------------------

/** Stages only the movie adapter contributes (raters, lapse, model governance, the recommender). */
export const MOVIE_ONLY: DomainCapability[] = ["recommendation", "user_intelligence", "lapse", "raters", "model_governance"];

export const CAPABILITY_LABEL: Record<DomainCapability, string> = {
  recommendation: "Recommender monitoring",
  user_intelligence: "Per-user intelligence",
  lapse: "Audience-lapse model",
  raters: "Rater-behaviour anomalies",
  model_governance: "Model governance",
  scenarios: "Scenarios",
};

const CAPABILITY_WHY: Record<DomainCapability, string> = {
  recommendation: "there is no recommender to monitor",
  user_intelligence: "it has no per-user histories",
  lapse: "it has no per-member activity to model lapse from",
  raters: "it has no individual raters to profile",
  model_governance: "it serves no trained model to govern",
  scenarios: "its adapter does not support what-if projections",
};

export interface CapabilityState {
  available: boolean;
  /** one line for the operator when the section is hidden; null when available */
  reason: string | null;
}

/**
 * Whether the chosen domain produces a section. The domain's declared capabilities win; without
 * them (GET /intel/domains unavailable) the movie domain keeps every section it had before the
 * migration and any other domain hides the movie-only ones.
 */
export function capabilityState(domain: string, info: Pick<DomainInfo, "name" | "capabilities"> | null | undefined, cap: DomainCapability): CapabilityState {
  const declared: DomainCapabilities | undefined = info?.capabilities;
  const known = declared && typeof declared[cap] === "boolean";
  const available = known ? Boolean(declared[cap]) : normalizeDomain(domain) === DEFAULT_DOMAIN || !MOVIE_ONLY.includes(cap);
  if (available) return { available, reason: null };
  const name = info?.name ?? domain;
  return {
    available,
    reason: `${CAPABILITY_LABEL[cap]} is not shown for ${name}: ${CAPABILITY_WHY[cap]}${known ? "" : " (capabilities not reported, so movie-only sections are hidden)"}.`,
  };
}

/** A domain's display name, from the list when it is known. */
export function domainName(domain: string, domains: Pick<DomainInfo, "key" | "name">[] | null | undefined): string {
  const d = normalizeDomain(domain);
  return domains?.find((x) => x.key === d)?.name ?? (d === DEFAULT_DOMAIN ? "Movies" : d);
}
