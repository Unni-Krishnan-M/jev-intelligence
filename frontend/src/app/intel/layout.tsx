import type { Metadata } from "next";

import { ConsoleShell } from "@/components/jev/intel/console-shell";

export const metadata: Metadata = {
  title: "Intelligence console",
  description: "JEV operator console: signals, trends, anomalies, forecasts, risk, JEV decisions, early warnings and actions per domain.",
};

export default function IntelLayout({ children }: LayoutProps<"/intel">) {
  return <ConsoleShell>{children}</ConsoleShell>;
}
