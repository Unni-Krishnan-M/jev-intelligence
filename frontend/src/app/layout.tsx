import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Instrument_Serif } from "next/font/google";

import { Providers } from "@/components/jev/providers";
import { SiteHeader } from "@/components/jev/site-header";
import "./globals.css";

const display = Instrument_Serif({ variable: "--font-display", subsets: ["latin"], weight: "400", style: ["normal", "italic"] });
const sans = IBM_Plex_Sans({ variable: "--font-plex-sans", subsets: ["latin"], weight: ["400", "500", "600"] });
const mono = IBM_Plex_Mono({ variable: "--font-plex-mono", subsets: ["latin"], weight: ["400", "500"] });

export const metadata: Metadata = {
  title: { default: "JEV — a film programme that learns you", template: "%s · JEV" },
  description:
    "JEV is a hybrid movie recommender: content, collaborative filtering, matrix factorization and popularity, blended and explained.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#0f0e0c" },
    { media: "(prefers-color-scheme: light)", color: "#f3eee4" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={`${display.variable} ${sans.variable} ${mono.variable} h-full`}>
      <body className="min-h-full flex flex-col">
        <Providers>
          <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-3 focus:z-50 focus:rounded focus:bg-primary focus:px-3 focus:py-2 focus:text-primary-foreground">
            Skip to content
          </a>
          <SiteHeader />
          <main id="main" className="flex-1">{children}</main>
          <footer className="border-t hairline mt-24">
            <div className="mx-auto flex max-w-[1320px] flex-wrap items-center justify-between gap-3 px-5 py-6 sm:px-8">
              <p className="eyebrow">JEV · hybrid recommendation engine · v1.0</p>
              <p className="text-xs text-muted-foreground">
                Ratings: MovieLens (GroupLens, research use). Metadata: Wikidata (CC0). No posters are used; every cover is typeset.
              </p>
            </div>
          </footer>
        </Providers>
      </body>
    </html>
  );
}
