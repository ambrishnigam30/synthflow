"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  BarChart2,
  Zap,
  Database,
  Clock,
  ChevronRight,
  CheckCircle2,
  Circle,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useAuthStore } from "@/lib/stores/authStore";
import { usageApi, generateApi, llmConfigApi, type UsageSummary, type GenerationSummary } from "@/lib/api";

function getGreeting(): string {
  const h = new Date().getHours();
  if (h >= 5 && h < 12) return "Good morning";
  if (h >= 12 && h < 17) return "Good afternoon";
  if (h >= 17 && h < 21) return "Good evening";
  return "Hello";
}

// ── Fallback data (shown while loading or on API error) ──────────────────────

const FALLBACK_USAGE: { month: string; rows: number }[] = [
  { month: "Nov", rows: 0 },
  { month: "Dec", rows: 0 },
  { month: "Jan", rows: 0 },
  { month: "Feb", rows: 0 },
  { month: "Mar", rows: 0 },
  { month: "Apr", rows: 0 },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

function formatRows(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function qualityColor(score: number): string {
  if (score > 80) return "#1f8a65";
  if (score >= 60) return "#c08532";
  return "#cf2d56";
}

// ── Stat card ────────────────────────────────────────────────────────────────

interface StatCardProps {
  label: string;
  value: string;
  sub?: string;
  icon: React.ComponentType<{ size?: number; strokeWidth?: number }>;
  accent?: string;
}

function StatCard({ label, value, sub, icon: Icon, accent = "#f54e00" }: StatCardProps) {
  return (
    <div
      className="rounded-[8px] p-5"
      style={{
        background: "#ffffff",
        border: "1px solid rgba(38,37,30,0.1)",
      }}
    >
      <div className="flex items-start justify-between mb-3">
        <div
          className="w-8 h-8 rounded-[6px] flex items-center justify-center"
          style={{ background: `${accent}15` }}
        >
          <Icon size={16} strokeWidth={1.5} />
        </div>
      </div>
      <p
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "28px",
          fontWeight: 400,
          color: "#26251e",
          letterSpacing: "-0.56px",
          lineHeight: 1.1,
        }}
      >
        {value}
      </p>
      <p
        className="mt-1"
        style={{
          fontFamily: "system-ui",
          fontSize: "12px",
          fontWeight: 500,
          color: "rgba(38,37,30,0.55)",
        }}
      >
        {label}
      </p>
      {sub && (
        <p
          className="mt-0.5"
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            color: "rgba(38,37,30,0.35)",
          }}
        >
          {sub}
        </p>
      )}
    </div>
  );
}

// ── Custom tooltip ────────────────────────────────────────────────────────────

interface TooltipPayload {
  value: number;
}

function UsageTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div
      className="rounded-[6px] px-3 py-2"
      style={{
        background: "#ffffff",
        border: "1px solid rgba(38,37,30,0.1)",
        boxShadow: "rgba(0,0,0,0.1) 0px 4px 12px",
      }}
    >
      <p
        style={{
          fontFamily: "system-ui",
          fontSize: "11px",
          fontWeight: 500,
          color: "rgba(38,37,30,0.55)",
          marginBottom: "2px",
        }}
      >
        {label}
      </p>
      <p
        style={{
          fontFamily: "var(--font-mono, monospace)",
          fontSize: "13px",
          color: "#26251e",
        }}
      >
        {formatRows(payload[0].value)} rows
      </p>
    </div>
  );
}

// ── Quick-start input ─────────────────────────────────────────────────────────

function QuickStart({ hasProvider }: { hasProvider: boolean | null }) {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;
    if (hasProvider === false) {
      router.push("/app/settings/providers");
      return;
    }
    router.push(`/app/generate?q=${encodeURIComponent(prompt.trim())}`);
  }

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <input
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Describe your dataset…  e.g. 5000 patient records in Mumbai"
        className="flex-1"
        style={{
          background: "transparent",
          border: "1px solid rgba(38,37,30,0.15)",
          borderRadius: "8px",
          padding: "10px 14px",
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "14px",
          color: "#26251e",
          outline: "none",
        }}
        onFocus={(e) => {
          e.target.style.borderColor = "rgba(38,37,30,0.3)";
        }}
        onBlur={(e) => {
          e.target.style.borderColor = "rgba(38,37,30,0.15)";
        }}
      />
      <button
        type="submit"
        className="flex items-center gap-1.5 transition-opacity hover:opacity-90"
        style={{
          background: "#f54e00",
          color: "#ffffff",
          border: "none",
          borderRadius: "8px",
          padding: "10px 16px",
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "14px",
          fontWeight: 400,
          cursor: "pointer",
          whiteSpace: "nowrap",
        }}
      >
        <Zap size={14} strokeWidth={1.5} />
        Generate
      </button>
    </form>
  );
}

// ── Onboarding checklist ──────────────────────────────────────────────────────

interface OnboardingState {
  providerAdded: boolean;
  generated: boolean;
  explored: boolean;
}

interface ChecklistItemProps {
  done: boolean;
  label: string;
  href: string;
  linkLabel: string;
}

function ChecklistItem({ done, label, href, linkLabel }: ChecklistItemProps) {
  return (
    <div className="flex items-center gap-3">
      {done ? (
        <CheckCircle2 size={16} strokeWidth={1.5} style={{ color: "#1f8a65", flexShrink: 0 }} />
      ) : (
        <Circle size={16} strokeWidth={1.5} style={{ color: "rgba(38,37,30,0.3)", flexShrink: 0 }} />
      )}
      <span
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "14px",
          color: done ? "rgba(38,37,30,0.4)" : "#26251e",
          textDecoration: done ? "line-through" : "none",
          flex: 1,
        }}
      >
        {label}
      </span>
      {!done && (
        <Link
          href={href}
          style={{
            fontFamily: "system-ui",
            fontSize: "12px",
            color: "#f54e00",
            textDecoration: "none",
          }}
        >
          {linkLabel} →
        </Link>
      )}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const firstName = user?.name?.split(" ")[0] ?? "";

  const [usageData, setUsageData] = useState<{ month: string; rows: number }[]>(FALLBACK_USAGE);
  const [usageSummary, setUsageSummary] = useState<UsageSummary | null>(null);
  const [recentGens, setRecentGens] = useState<GenerationSummary[]>([]);
  const [loadingData, setLoadingData] = useState(true);

  // Onboarding checklist state
  const [onboarding, setOnboarding] = useState<OnboardingState>({
    providerAdded: false,
    generated: false,
    explored: false,
  });
  const [onboardingDismissed, setOnboardingDismissed] = useState(true); // default hidden until loaded

  useEffect(() => {
    const dismissed = localStorage.getItem("sf_onboarding_dismissed") === "true";
    setOnboardingDismissed(dismissed);
    if (!dismissed) {
      llmConfigApi.list().then((configs) => {
        setOnboarding((prev) => ({ ...prev, providerAdded: configs.length > 0 }));
      }).catch(() => {});
      const explored = localStorage.getItem("sf_visited_explore") === "true";
      setOnboarding((prev) => ({ ...prev, explored }));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [historyRes, summaryRes, gensRes] = await Promise.allSettled([
          usageApi.history(),
          usageApi.summary(),
          generateApi.list({ limit: 5 }),
        ]);
        if (cancelled) return;
        if (historyRes.status === "fulfilled") {
          const mapped = historyRes.value.map((p) => ({
            month: p.month.slice(0, 3),
            rows: p.rows_generated,
          }));
          setUsageData(mapped.length > 0 ? mapped : FALLBACK_USAGE);
        }
        if (summaryRes.status === "fulfilled") setUsageSummary(summaryRes.value);
        if (gensRes.status === "fulfilled") {
          const items = gensRes.value.items;
          setRecentGens(items);
          if (items.length > 0) {
            setOnboarding((prev) => ({ ...prev, generated: true }));
          }
        }
      } catch {
        // Silent — fallback data already shown
      } finally {
        if (!cancelled) setLoadingData(false);
      }
    }
    void load();
    return () => { cancelled = true; };
  }, []);

  // Auto-dismiss checklist when all items are complete
  useEffect(() => {
    if (!onboardingDismissed && onboarding.providerAdded && onboarding.generated && onboarding.explored) {
      localStorage.setItem("sf_onboarding_dismissed", "true");
      setOnboardingDismissed(true);
    }
  }, [onboarding, onboardingDismissed]);

  const showOnboarding = !onboardingDismissed && !loadingData;

  return (
    <div className="p-6 max-w-[1200px] mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "26px",
            fontWeight: 400,
            color: "#26251e",
            letterSpacing: "-0.325px",
            lineHeight: 1.25,
          }}
        >
          {getGreeting()}{firstName ? `, ${firstName}` : ""}
        </h1>
        <p
          className="mt-1"
          style={{
            fontFamily: "var(--font-serif, Georgia, serif)",
            fontSize: "17px",
            color: "rgba(38,37,30,0.55)",
            lineHeight: 1.35,
          }}
        >
          Your synthetic data platform is ready.
        </p>
      </div>

      {/* Onboarding checklist */}
      {showOnboarding && (
        <div
          className="rounded-[8px] p-5 mb-8"
          style={{
            background: "#ffffff",
            border: "1px solid rgba(38,37,30,0.1)",
          }}
        >
          <div className="flex items-center justify-between mb-4">
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 600,
                letterSpacing: "0.048px",
                color: "rgba(38,37,30,0.45)",
                textTransform: "uppercase",
              }}
            >
              Getting started
            </p>
            <button
              onClick={() => {
                localStorage.setItem("sf_onboarding_dismissed", "true");
                setOnboardingDismissed(true);
              }}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontFamily: "system-ui",
                fontSize: "12px",
                color: "rgba(38,37,30,0.35)",
              }}
            >
              Dismiss
            </button>
          </div>
          <div className="space-y-3">
            <ChecklistItem
              done={onboarding.providerAdded}
              label="Add an AI provider"
              href="/app/settings/providers"
              linkLabel="Go to Settings"
            />
            <ChecklistItem
              done={onboarding.generated}
              label="Generate your first dataset"
              href="/app/generate"
              linkLabel="Generate now"
            />
            <ChecklistItem
              done={onboarding.explored}
              label="Upload and explore data"
              href="/app/explore"
              linkLabel="Open Explorer"
            />
          </div>
        </div>
      )}

      {/* Quick start */}
      <div
        className="rounded-[8px] p-5 mb-8"
        style={{
          background: "#ffffff",
          border: "1px solid rgba(38,37,30,0.1)",
        }}
      >
        <p
          className="mb-3"
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            fontWeight: 600,
            letterSpacing: "0.048px",
            color: "rgba(38,37,30,0.45)",
            textTransform: "uppercase",
          }}
        >
          Quick Start
        </p>
        <QuickStart hasProvider={onboarding.providerAdded} />
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Rows generated this month"
          value={loadingData ? "…" : formatRows(usageSummary?.rows_this_month ?? 0)}
          sub={usageSummary ? `of ${formatRows(usageSummary.rows_limit)} limit` : undefined}
          icon={BarChart2}
          accent="#f54e00"
        />
        <StatCard
          label="Generations total"
          value={loadingData ? "…" : String(usageSummary?.total_generations ?? 0)}
          icon={Zap}
          accent="#c08532"
        />
        <StatCard
          label="Active datasets"
          value={loadingData ? "…" : String(usageSummary?.active_datasets ?? 0)}
          icon={Database}
          accent="#1f8a65"
        />
        <StatCard
          label="API calls today"
          value={loadingData ? "…" : String(usageSummary?.api_calls_today ?? 0)}
          icon={Clock}
          accent="#9fc9a2"
        />
      </div>

      {/* Charts + Recent */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_400px] gap-6">
        {/* Usage chart */}
        <div
          className="rounded-[8px] p-5"
          style={{
            background: "#ffffff",
            border: "1px solid rgba(38,37,30,0.1)",
          }}
        >
          <p
            className="mb-5"
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 600,
              letterSpacing: "0.048px",
              color: "rgba(38,37,30,0.45)",
              textTransform: "uppercase",
            }}
          >
            Rows Generated — Last 6 Months
          </p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={usageData} barSize={32}>
              <CartesianGrid
                vertical={false}
                stroke="rgba(38,37,30,0.06)"
              />
              <XAxis
                dataKey="month"
                axisLine={false}
                tickLine={false}
                tick={{
                  fontFamily: "system-ui",
                  fontSize: 11,
                  fill: "rgba(38,37,30,0.45)",
                }}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tickFormatter={(v: number) => formatRows(v)}
                tick={{
                  fontFamily: "system-ui",
                  fontSize: 11,
                  fill: "rgba(38,37,30,0.45)",
                }}
                width={40}
              />
              <Tooltip content={<UsageTooltip />} cursor={{ fill: "rgba(38,37,30,0.04)" }} />
              <Bar dataKey="rows" fill="#f54e00" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Recent generations */}
        <div
          className="rounded-[8px] overflow-hidden"
          style={{
            background: "#ffffff",
            border: "1px solid rgba(38,37,30,0.1)",
          }}
        >
          <div
            className="px-5 py-4 flex items-center justify-between"
            style={{ borderBottom: "1px solid rgba(38,37,30,0.08)" }}
          >
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 600,
                letterSpacing: "0.048px",
                color: "rgba(38,37,30,0.45)",
                textTransform: "uppercase",
              }}
            >
              Recent Generations
            </p>
            <button
              onClick={() => router.push("/app/history")}
              className="flex items-center gap-0.5 transition-opacity hover:opacity-70"
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontFamily: "system-ui",
                fontSize: "12px",
                color: "#f54e00",
              }}
            >
              View all <ChevronRight size={12} strokeWidth={2} />
            </button>
          </div>

          {loadingData ? (
            <div className="px-5 py-8 text-center" style={{ color: "rgba(38,37,30,0.35)", fontSize: "13px", fontFamily: "system-ui" }}>
              Loading…
            </div>
          ) : recentGens.length === 0 ? (
            <div className="px-5 py-8 text-center" style={{ color: "rgba(38,37,30,0.35)", fontSize: "13px", fontFamily: "system-ui" }}>
              No generations yet. Try the quick start above.
            </div>
          ) : (
            <ul>
              {recentGens.map((gen, i) => (
                <li
                  key={gen.id}
                  onClick={() => router.push(`/history?id=${gen.id}`)}
                  className="px-5 py-3.5 cursor-pointer transition-colors"
                  style={{
                    borderBottom:
                      i < recentGens.length - 1
                        ? "1px solid rgba(38,37,30,0.06)"
                        : "none",
                  }}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.background = "#f7f7f4";
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.background = "transparent";
                  }}
                >
                  <div className="flex items-start justify-between gap-2">
                    <p
                      className="flex-1 line-clamp-1"
                      style={{
                        fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                        fontSize: "13px",
                        color: "#26251e",
                      }}
                    >
                      {gen.prompt}
                    </p>
                    {gen.quality_score > 0 && (
                      <span
                        style={{
                          fontFamily: "var(--font-mono, monospace)",
                          fontSize: "13px",
                          fontWeight: 400,
                          color: qualityColor(gen.quality_score),
                          flexShrink: 0,
                        }}
                      >
                        {gen.quality_score.toFixed(1)}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-3 mt-1">
                    <span style={{ fontFamily: "system-ui", fontSize: "11px", color: "rgba(38,37,30,0.4)" }}>
                      {gen.domain}
                    </span>
                    <span style={{ fontFamily: "var(--font-mono, monospace)", fontSize: "11px", color: "rgba(38,37,30,0.4)" }}>
                      {formatRows(gen.row_count)} rows
                    </span>
                    <span style={{ fontFamily: "system-ui", fontSize: "11px", color: "rgba(38,37,30,0.35)" }}>
                      {gen.created_at.slice(0, 10)}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
