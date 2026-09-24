import type { Metadata, Viewport } from "next";
import { headers } from "next/headers";
import { IBM_Plex_Mono, IBM_Plex_Sans, Instrument_Serif } from "next/font/google";

import { Providers } from "@/components/jev/providers";
import { SiteHeader } from "@/components/jev/site-header";
import { shortVersion } from "@/lib/version";
import "./globals.css";

const display = Instrument_Serif({ variable: "--font-display", subsets: ["latin"], weight: "400", style: ["normal", "italic"] });
const sans = IBM_Plex_Sans({ variable: "--font-plex-sans", subsets: ["latin"], weight: ["400", "500", "600"] });
const mono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  title: { default: "JEV — Intelligent Decision & Early-Warning Engine", template: "%s · JEV" },
  description:
    "JEV turns changing data into signals, trends, anomalies, forecasts and risk, then into typed, explained decisions, early warnings, recommendations and actions. Movies is its first domain.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0f0e0c" },
    { media: "(prefers-color-scheme: light)", color: "#f3eee4" },
  ],
};

export default async function RootLayout({ children }: LayoutProps<"/">) {
  // set by src/proxy.ts with the Content-Security-Policy; next-themes' inline script needs it
  const nonce = (await headers()).get("x-nonce") ?? undefined;
  return (
    <html lang="en" suppressHydrationWarning className={`${display.variable} ${sans.variable} ${mono.variable} h-full`}>
      <body className="min-h-full flex flex-col">
        <Providers nonce={nonce}>
          <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-3 focus:z-50 focus:rounded focus:bg-primary focus:px-3 focus:py-2 focus:text-primary-foreground">
            Skip to content
          </a>
          <SiteHeader />
          <main id="main" className="flex-1">{children}</main>
          <footer className="border-t hairline mt-24">
            <div className="mx-auto flex max-w-[1320px] flex-wrap items-center justify-between gap-3 px-5 py-6 sm:px-8">
              <p className="eyebrow">JEV · decision &amp; early-warning engine · {shortVersion()}</p>
              <p className="text-xs text-muted-foreground">
                Movies: MovieLens (GroupLens, research use) and Wikidata (CC0); every cover is typeset. US unemployment: BLS via FRED (public domain).
              </p>
            </div>
          </footer>
        </Providers>
      </body>
    </html>
  );
}
