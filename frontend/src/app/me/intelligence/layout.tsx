import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "My intelligence",
  description: "What JEV sees in your taste: preference history, the drift test, the strategy decision behind your recommendations and what-if projections.",
};

export default function MyIntelligenceLayout({ children }: LayoutProps<"/me/intelligence">) {
  return children;
}
