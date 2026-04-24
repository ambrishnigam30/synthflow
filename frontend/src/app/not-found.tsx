import Link from "next/link";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Page not found — SynthFlow",
};

export default function NotFound() {
  return (
    <div
      className="min-h-screen flex flex-col items-center justify-center px-4"
      style={{ background: "#f2f1ed" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 mb-12">
        <div
          className="w-8 h-8 rounded-[6px] flex items-center justify-center"
          style={{ background: "#3d4043" }}
        >
          <span className="text-white text-[15px] font-[500]">S</span>
        </div>
        <span
          className="text-[16px] font-[400]"
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            color: "#3d4043",
            letterSpacing: "-0.15px",
          }}
        >
          SynthFlow
        </span>
      </div>

      {/* 404 number */}
      <p
        className="mb-3 font-[400] leading-none"
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "96px",
          letterSpacing: "-2.88px",
          color: "rgba(38,37,30,0.12)",
        }}
      >
        404
      </p>

      {/* Heading */}
      <h1
        className="mb-3 text-center"
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "26px",
          fontWeight: 400,
          letterSpacing: "-0.325px",
          color: "#26251e",
        }}
      >
        Page not found
      </h1>

      {/* Body */}
      <p
        className="mb-10 text-center max-w-[360px] leading-relaxed"
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "16px",
          fontWeight: 400,
          color: "rgba(38,37,30,0.55)",
        }}
      >
        The page you&apos;re looking for doesn&apos;t exist or has been moved.
      </p>

      {/* Actions */}
      <div className="flex items-center gap-3">
        <Link
          href="/"
          className="inline-flex items-center px-4 py-2.5 rounded-[8px] text-[14px] font-[400] transition-colors hover:text-[#cf2d56]"
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            background: "#ebeae5",
            color: "#26251e",
            textDecoration: "none",
          }}
        >
          Go home
        </Link>

        <Link
          href="/dashboard"
          className="inline-flex items-center px-4 py-2.5 rounded-[8px] text-[14px] font-[400] opacity-100 hover:opacity-90 transition-opacity"
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            background: "#f54e00",
            color: "#ffffff",
            textDecoration: "none",
          }}
        >
          Open dashboard
        </Link>
      </div>

      {/* Decorative divider */}
      <div
        className="mt-16 w-px h-16"
        style={{ background: "rgba(38,37,30,0.1)" }}
      />
      <p
        className="mt-4 text-center"
        style={{
          fontFamily: "system-ui",
          fontSize: "11px",
          fontWeight: 500,
          letterSpacing: "0.048px",
          textTransform: "uppercase",
          color: "rgba(38,37,30,0.3)",
        }}
      >
        SynthFlow — Autonomous Synthetic Data
      </p>
    </div>
  );
}
