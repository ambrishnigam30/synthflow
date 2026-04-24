"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { label: "Profile", href: "/app/settings" },
  { label: "LLM Providers", href: "/app/settings/providers" },
  { label: "Team", href: "/app/settings/team" },
  { label: "Billing", href: "/app/settings/billing" },
  { label: "API Keys", href: "/app/settings/api-keys" },
  { label: "Webhooks", href: "/app/settings/webhooks" },
  { label: "Preferences", href: "/app/settings/preferences" },
];

export default function SettingsLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  function isActive(href: string): boolean {
    if (href === "/app/settings") return pathname === "/app/settings";
    return pathname?.startsWith(href) ?? false;
  }

  return (
    <div className="flex flex-col h-full">
      {/* Page header */}
      <div
        className="px-6 pt-6 pb-0 flex-shrink-0"
        style={{ background: "#f2f1ed" }}
      >
        <h1
          className="mb-4"
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "22px",
            fontWeight: 400,
            color: "#26251e",
            letterSpacing: "-0.22px",
          }}
        >
          Settings
        </h1>

        {/* Tab bar */}
        <div
          className="flex items-end gap-0 overflow-x-auto"
          style={{ borderBottom: "1px solid rgba(38,37,30,0.1)" }}
        >
          {TABS.map(({ label, href }) => {
            const active = isActive(href);
            return (
              <Link
                key={href}
                href={href}
                className="flex-shrink-0 px-4 py-2.5 transition-colors"
                style={{
                  fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                  fontSize: "13px",
                  fontWeight: 400,
                  color: active ? "#26251e" : "rgba(38,37,30,0.55)",
                  textDecoration: "none",
                  borderBottom: active ? "2px solid #f54e00" : "2px solid transparent",
                  marginBottom: "-1px",
                  whiteSpace: "nowrap",
                }}
                onMouseEnter={(e) => {
                  if (!active) (e.currentTarget as HTMLElement).style.color = "#26251e";
                }}
                onMouseLeave={(e) => {
                  if (!active) (e.currentTarget as HTMLElement).style.color = "rgba(38,37,30,0.55)";
                }}
              >
                {label}
              </Link>
            );
          })}
        </div>
      </div>

      {/* Tab content */}
      <div className="flex-1 overflow-auto px-6 py-6">
        {children}
      </div>
    </div>
  );
}
