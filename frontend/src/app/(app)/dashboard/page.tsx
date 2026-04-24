"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import {
  BarChart2,
  Zap,
  Database,
  Clock,
  ChevronRight,
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

// ── Mock data ────────────────────────────────────────────────────────────────

const USAGE_DATA = [
  { month: "Nov", rows: 420000 },
  { month: "Dec", rows: 680000 },
  { month: "Jan", rows: 520000 },
  { month: "Feb", rows: 910000 },
  { month: "Mar", rows: 760000 },
  { month: "Apr", rows: 340000 },
];

const RECENT_GENERATIONS = [
  {
    id: "g1",
    prompt: "Generate 10,000 healthcare patient records for Maharashtra",
    domain: "Healthcare",
    rows: 10000,
    quality: 94,
    status: "done",
    date: "2026-04-24",
  },
  {
    id: "g2",
    prompt: "Banking transaction dataset with fraud patterns for HDFC",
    domain: "Banking",
    rows: 50000,
    quality: 91,
    status: "done",
    date: "2026-04-23",
  },
  {
    id: "g3",
    prompt: "Retail customer purchase history, e-commerce India",
    domain: "Retail",
    rows: 25000,
    quality: 88,
    status: "done",
    date: "2026-04-22",
  },
  {
    id: "g4",
    prompt: "Agricultural yield dataset for Punjab wheat season",
    domain: "Agriculture",
    rows: 8000,
    quality: 96,
    status: "done",
    date: "2026-04-21",
  },
  {
    id: "g5",
    prompt: "IoT sensor readings for smart factory monitoring",
    domain: "IoT",
    rows: 100000,
    quality: 89,
    status: "done",
    date: "2026-04-20",
  },
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

function QuickStart() {
  const router = useRouter();
  const [prompt, setPrompt] = useState("");

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!prompt.trim()) return;
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

// ── Page ─────────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const firstName = user?.name?.split(" ")[0] ?? "there";

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
          Good morning, {firstName}
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
        <QuickStart />
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Rows generated this month"
          value="3.6M"
          sub="of 10M limit"
          icon={BarChart2}
          accent="#f54e00"
        />
        <StatCard
          label="Generations total"
          value="47"
          sub="5 this week"
          icon={Zap}
          accent="#c08532"
        />
        <StatCard
          label="Active datasets"
          value="12"
          sub="2.3 GB stored"
          icon={Database}
          accent="#1f8a65"
        />
        <StatCard
          label="Avg. quality score"
          value="91.4"
          sub="↑ 2.1 pts vs last month"
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
            <BarChart data={USAGE_DATA} barSize={32}>
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

          <ul>
            {RECENT_GENERATIONS.map((gen, i) => (
              <li
                key={gen.id}
                onClick={() => router.push(`/app/history?id=${gen.id}`)}
                className="px-5 py-3.5 cursor-pointer transition-colors"
                style={{
                  borderBottom:
                    i < RECENT_GENERATIONS.length - 1
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
                  <span
                    style={{
                      fontFamily: "var(--font-mono, monospace)",
                      fontSize: "13px",
                      fontWeight: 400,
                      color: qualityColor(gen.quality),
                      flexShrink: 0,
                    }}
                  >
                    {gen.quality}
                  </span>
                </div>
                <div className="flex items-center gap-3 mt-1">
                  <span
                    style={{
                      fontFamily: "system-ui",
                      fontSize: "11px",
                      color: "rgba(38,37,30,0.4)",
                    }}
                  >
                    {gen.domain}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--font-mono, monospace)",
                      fontSize: "11px",
                      color: "rgba(38,37,30,0.4)",
                    }}
                  >
                    {formatRows(gen.rows)} rows
                  </span>
                  <span
                    style={{
                      fontFamily: "system-ui",
                      fontSize: "11px",
                      color: "rgba(38,37,30,0.35)",
                    }}
                  >
                    {gen.date}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
