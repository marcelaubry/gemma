import "./globals.css";

import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Gemma Compute Monitor",
  description:
    "Real-time per-transformer-layer CPU/GPU/memory telemetry for a locally running Gemma 3 4B model.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
