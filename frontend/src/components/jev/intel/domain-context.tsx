"use client";

import { Info } from "lucide-react";
import NextLink from "next/link";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { createContext, useContext, useMemo, type ComponentProps } from "react";
import useSWR from "swr";

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { capabilityState, DEFAULT_DOMAIN, DOMAIN_PARAM, domainName, intelHref, normalizeDomain, switchDomainHref, withDomain, type CapabilityState } from "@/lib/domain";
import { isNotDeployed } from "@/lib/intel";
import type { DomainCapability, DomainInfo, DomainList } from "@/lib/intel-types";
import { cn } from "@/lib/utils";

export type DomainListState = "loading" | "ok" | "unavailable" | "error";

interface DomainContextValue {
  /** the adapter key in ?domain= (default "movie") */
  domain: string;
  /** display name of the chosen domain */
  name: string;
  /** GET /intel/domains items; null until loaded or when the endpoint is missing */
  domains: DomainInfo[] | null;
  info: DomainInfo | null;
  listState: DomainListState;
  can: (cap: DomainCapability) => CapabilityState;
  /** an /intel/* API path with the domain param */
  q: (path: string) => string;
}

function makeValue(domain: string, domains: DomainInfo[] | null, listState: DomainListState): DomainContextValue {
  const info = domains?.find((d) => d.key === domain) ?? null;
  return {
    domain,
    name: domainName(domain, domains),
    domains,
    info,
    listState,
    can: (cap) => capabilityState(domain, info, cap),
    q: (path) => withDomain(path, domain),
  };
}

const DomainContext = createContext<DomainContextValue>(makeValue(DEFAULT_DOMAIN, null, "unavailable"));

/** The console's chosen domain. Outside the console it is the default (movie). */
export function useIntelDomain(): DomainContextValue {
  return useContext(DomainContext);
}

/** Reads ?domain= and the domain list once for the whole console. Needs a Suspense boundary. */
export function DomainProvider({ children }: { children: React.ReactNode }) {
  const domain = normalizeDomain(useSearchParams().get(DOMAIN_PARAM));
  const { data, error } = useSWR<DomainList>("/intel/domains", { revalidateOnFocus: false });
  const listState: DomainListState = data ? "ok" : error ? (isNotDeployed(error) ? "unavailable" : "error") : "loading";
  const domains = data?.items ?? null;
  const value = useMemo(() => makeValue(domain, domains, listState), [domain, domains, listState]);
  return <DomainContext.Provider value={value}>{children}</DomainContext.Provider>;
}

/** next/link that keeps ?domain= on console links; any other href passes through unchanged. */
export function IntelLink({ href, ...props }: ComponentProps<typeof NextLink>) {
  const { domain } = useIntelDomain();
  return <NextLink href={typeof href === "string" ? intelHref(href, domain) : href} {...props} />;
}

export default IntelLink;

/** One line saying why a section is hidden for this domain. Renders nothing when it is available. */
export function CapabilityNotice({ cap, className }: { cap: DomainCapability; className?: string }) {
  const { can } = useIntelDomain();
  const s = can(cap);
  if (s.available) return null;
  return (
    <p className={cn("flex items-start gap-2 rounded border border-dashed hairline px-3 py-2 text-sm text-muted-foreground", className)}>
      <Info className="mt-0.5 size-4 shrink-0" aria-hidden />
      <span>{s.reason}</span>
    </p>
  );
}

/** Domain switcher for the console rail / phone header. The choice lives in ?domain=. */
export function DomainPicker({ className }: { className?: string }) {
  const { domain, domains, info, listState } = useIntelDomain();
  const router = useRouter();
  const pathname = usePathname();
  const options: { key: string; name: string; available: boolean; reason: string | null }[] = domains?.length
    ? domains.map((d) => ({ key: d.key, name: d.name, available: d.available, reason: d.reason }))
    : [{ key: DEFAULT_DOMAIN, name: "Movies", available: true, reason: null }];
  if (!options.some((o) => o.key === domain)) options.push({ key: domain, name: domain, available: false, reason: "not in the domain list" });

  return (
    <div className={cn("min-w-0", className)}>
      <p className="eyebrow mb-1.5" id="domain-picker-label">Domain</p>
      <Select value={domain} onValueChange={(v) => router.push(switchDomainHref(pathname, v))} disabled={listState === "loading" && !domains}>
        <SelectTrigger className="h-8 w-full text-sm" aria-labelledby="domain-picker-label">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.key} value={o.key} disabled={!o.available && o.key !== domain}>
              {o.name}
              {!o.available && <span className="ml-1 text-xs text-muted-foreground">(unavailable)</span>}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <p className="mt-1.5 text-xs leading-snug text-muted-foreground">
        {listState === "unavailable"
          ? "Domain list not available on this API yet; showing Movies."
          : listState === "error"
            ? "Could not load the domain list."
            : info && !info.available
              ? `Unavailable here: ${info.reason ?? "no reason given"}.`
              : info
                ? `${info.frequency} · ${info.warnings_open.toLocaleString()} open warning${info.warnings_open === 1 ? "" : "s"}`
                : null}
      </p>
    </div>
  );
}
