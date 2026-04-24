"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  LayoutDashboard,
  Zap,
  Search,
  Clock,
  Settings,
  LogOut,
  ChevronRight,
  X,
} from "lucide-react";
import { useAuthStore } from "@/lib/stores/authStore";

// ── Nav item definition ──────────────────────────────────────────────────────

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number; className?: string }>;
}

const NAV_ITEMS: NavItem[] = [
  { href: "/app/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/app/generate", label: "Generate", icon: Zap },
  { href: "/app/explore", label: "Explore", icon: Search },
  { href: "/app/history", label: "History", icon: Clock },
];

// ── Logo mark ────────────────────────────────────────────────────────────────

function AppLogo() {
  return (
    <Link href="/app/dashboard" className="flex items-center gap-2.5">
      <div
        className="w-7 h-7 rounded-[6px] flex items-center justify-center flex-shrink-0"
        style={{ backgroundColor: "#3d4043" }}
      >
        <span className="text-white text-[13px] font-[500]">S</span>
      </div>
      <span
        className="text-[15px] font-[400]"
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          color: "#3d4043",
          letterSpacing: "-0.15px",
        }}
      >
        SynthFlow
      </span>
    </Link>
  );
}

// ── Single nav link ──────────────────────────────────────────────────────────

function NavLink({ item, active }: { item: NavItem; active: boolean }) {
  const Icon = item.icon;
  return (
    <Link
      href={item.href}
      className="flex items-center gap-2.5 px-3 py-2 transition-all duration-150"
      style={{
        borderRadius: "0 6px 6px 0",
        borderLeft: active ? "2px solid #f54e00" : "2px solid transparent",
        background: active ? "#f2f1ed" : "transparent",
        color: active ? "#26251e" : "rgba(38,37,30,0.55)",
        fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
        fontSize: "14px",
        fontWeight: 400,
        textDecoration: "none",
      }}
      onMouseEnter={(e) => {
        if (!active) {
          (e.currentTarget as HTMLElement).style.background = "#f7f7f4";
        }
      }}
      onMouseLeave={(e) => {
        if (!active) {
          (e.currentTarget as HTMLElement).style.background = "transparent";
        }
      }}
    >
      <Icon size={18} strokeWidth={1.5} />
      <span>{item.label}</span>
    </Link>
  );
}

// ── Plan badge ───────────────────────────────────────────────────────────────

function PlanIndicator({ plan }: { plan: string }) {
  return (
    <div className="px-4 pb-4">
      <div
        className="rounded-[6px] p-3"
        style={{
          background: "#f7f7f4",
          border: "1px solid rgba(38,37,30,0.08)",
        }}
      >
        <div className="flex items-center justify-between mb-1.5">
          <span
            className="uppercase"
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 500,
              letterSpacing: "0.048px",
              color: "rgba(38,37,30,0.55)",
            }}
          >
            {plan} Plan
          </span>
          <Link
            href="/pricing"
            className="transition-opacity hover:opacity-80"
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "12px",
              fontWeight: 400,
              color: "#f54e00",
              textDecoration: "none",
            }}
          >
            Upgrade
          </Link>
        </div>
        {plan === "free" && (
          <>
            <div
              className="rounded-full overflow-hidden mb-1"
              style={{ height: "4px", background: "rgba(38,37,30,0.1)" }}
            >
              <div
                className="h-full rounded-full"
                style={{ width: "34%", background: "#f54e00" }}
              />
            </div>
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                color: "rgba(38,37,30,0.45)",
              }}
            >
              340K / 1M rows used
            </p>
          </>
        )}
      </div>
    </div>
  );
}

// ── User avatar row ──────────────────────────────────────────────────────────

function UserRow({ onLogout }: { onLogout: () => void }) {
  const user = useAuthStore((s) => s.user);
  const initials = user?.name
    .split(" ")
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase() ?? "SF";

  return (
    <div
      className="px-4 py-3 flex items-center gap-2.5"
      style={{ borderTop: "1px solid rgba(38,37,30,0.08)" }}
    >
      {user?.avatarUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={user.avatarUrl}
          alt={user.name}
          className="w-7 h-7 rounded-full flex-shrink-0"
        />
      ) : (
        <div
          className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ background: "#ebeae5" }}
        >
          <span
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 600,
              color: "#26251e",
            }}
          >
            {initials}
          </span>
        </div>
      )}
      <div className="flex-1 min-w-0">
        <p
          className="truncate"
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            fontWeight: 400,
            color: "#26251e",
          }}
        >
          {user?.name ?? "User"}
        </p>
        <p
          className="truncate"
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            color: "rgba(38,37,30,0.45)",
          }}
        >
          {user?.email ?? ""}
        </p>
      </div>
      <button
        onClick={onLogout}
        title="Log out"
        className="flex-shrink-0 transition-opacity hover:opacity-70"
        style={{ color: "rgba(38,37,30,0.4)", background: "none", border: "none", cursor: "pointer" }}
      >
        <LogOut size={16} strokeWidth={1.5} />
      </button>
    </div>
  );
}

// ── Sidebar ──────────────────────────────────────────────────────────────────

function Sidebar({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const pathname = usePathname();
  const { user, logout } = useAuthStore();

  return (
    <>
      {/* Mobile overlay */}
      {open && (
        <div
          className="fixed inset-0 z-30 md:hidden"
          style={{ background: "rgba(38,37,30,0.3)" }}
          onClick={onClose}
        />
      )}

      {/* Sidebar panel */}
      <aside
        className="fixed inset-y-0 left-0 z-40 flex flex-col transition-transform duration-200 md:translate-x-0 md:static md:z-auto"
        style={{
          width: "260px",
          background: "#ffffff",
          borderRight: "1px solid rgba(38,37,30,0.1)",
          transform: open ? "translateX(0)" : undefined,
        }}
        data-mobile-open={open}
      >
        {/* Logo */}
        <div className="px-4 py-5 flex items-center justify-between">
          <AppLogo />
          <button
            onClick={onClose}
            className="md:hidden transition-opacity hover:opacity-70"
            style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(38,37,30,0.45)" }}
          >
            <X size={18} strokeWidth={1.5} />
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-2 space-y-0.5 overflow-y-auto">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.href}
              item={item}
              active={pathname?.startsWith(item.href) ?? false}
            />
          ))}

          <div className="pt-4" style={{ borderTop: "1px solid rgba(38,37,30,0.06)", marginTop: "8px" }}>
            <NavLink
              item={{ href: "/app/settings", label: "Settings", icon: Settings }}
              active={pathname?.startsWith("/app/settings") ?? false}
            />
          </div>
        </nav>

        {/* Plan + user */}
        <div>
          <PlanIndicator plan={user?.plan ?? "free"} />
          <UserRow onLogout={logout} />
        </div>
      </aside>
    </>
  );
}

// ── Mobile top bar ────────────────────────────────────────────────────────────

function MobileTopBar({ onOpen }: { onOpen: () => void }) {
  return (
    <header
      className="md:hidden flex items-center justify-between px-4 h-12 flex-shrink-0"
      style={{
        background: "#ffffff",
        borderBottom: "1px solid rgba(38,37,30,0.1)",
        position: "sticky",
        top: 0,
        zIndex: 20,
      }}
    >
      <AppLogo />
      <button
        onClick={onOpen}
        className="transition-opacity hover:opacity-70"
        style={{ background: "none", border: "none", cursor: "pointer", color: "#26251e" }}
      >
        <ChevronRight size={20} strokeWidth={1.5} />
      </button>
    </header>
  );
}

// ── Root app layout ───────────────────────────────────────────────────────────

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: "#f2f1ed" }}>
      {/* Sidebar — hidden on mobile by default, shown via CSS on md+ */}
      <div className="hidden md:flex md:flex-shrink-0">
        <Sidebar open={false} onClose={() => undefined} />
      </div>

      {/* Mobile sidebar (portal-style overlay) */}
      <div className="md:hidden">
        {sidebarOpen && (
          <Sidebar open={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        )}
      </div>

      {/* Main content area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <MobileTopBar onOpen={() => setSidebarOpen(true)} />
        <main className="flex-1 overflow-auto" style={{ background: "#f2f1ed" }}>
          {children}
        </main>
      </div>
    </div>
  );
}
