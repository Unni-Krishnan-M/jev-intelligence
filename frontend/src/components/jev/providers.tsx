"use client";

import { ThemeProvider } from "next-themes";
import { SWRConfig } from "swr";

import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { swrFetcher } from "@/lib/api";

export function Providers({ children, nonce }: { children: React.ReactNode; nonce?: string }) {
  return (
    <ThemeProvider nonce={nonce} attribute="class" defaultTheme="dark" enableSystem={false} disableTransitionOnChange>
      <SWRConfig value={{ fetcher: swrFetcher, revalidateOnFocus: false, shouldRetryOnError: false, keepPreviousData: true }}>
        <TooltipProvider delayDuration={250}>
          {children}
          <Toaster position="bottom-right" />
        </TooltipProvider>
      </SWRConfig>
    </ThemeProvider>
  );
}
