import type { Metadata } from "next";
import { Geist, Instrument_Serif, JetBrains_Mono } from "next/font/google";
import "./globals.css";

/* ── Marketing font (Stripe-inspired pages) ─────────────────────────────── */
const geist = Geist({
  variable: "--font-geist-loaded",
  subsets: ["latin"],
  weight: ["300", "400"],
  display: "swap",
});

/* ── App fonts (Cursor-inspired pages) ──────────────────────────────────── */
const instrumentSerif = Instrument_Serif({
  variable: "--font-instrument-serif-loaded",
  subsets: ["latin"],
  weight: ["400"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono-loaded",
  subsets: ["latin"],
  weight: ["400", "500"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "SynthFlow — Data that understands the real world",
  description:
    "Autonomous synthetic data generation. Describe your dataset in plain English and get causally realistic, statistically accurate, privacy-safe data.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={[
        geist.variable,
        instrumentSerif.variable,
        jetbrainsMono.variable,
      ].join(" ")}
    >
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
