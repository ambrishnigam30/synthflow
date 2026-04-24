"use client";

import { useState } from "react";
import { Download } from "lucide-react";
import { useAuthStore } from "@/lib/stores/authStore";

// ── Data ──────────────────────────────────────────────────────────────────────

const PLANS = [
  {
    id: "free",
    name: "Free",
    price: "$0",
    period: "forever",
    highlight: false,
    features: ["10 generations / month", "1,000 rows max", "1 dataset upload"],
  },
  {
    id: "pro",
    name: "Pro",
    price: "$19",
    period: "/ month",
    highlight: true,
    features: ["200 generations / month", "100K rows max", "10 dataset uploads", "API access"],
  },
  {
    id: "business",
    name: "Business",
    price: "$49",
    period: "/ month",
    highlight: false,
    features: ["Unlimited generations", "1M rows max", "50 datasets", "Teams + Webhooks"],
  },
];

interface Invoice {
  id: string;
  date: string;
  amount: string;
  status: "paid" | "pending" | "failed";
  description: string;
}

const MOCK_INVOICES: Invoice[] = [
  { id: "inv_001", date: "2026-04-01", amount: "$19.00", status: "paid", description: "Pro plan — April 2026" },
  { id: "inv_002", date: "2026-03-01", amount: "$19.00", status: "paid", description: "Pro plan — March 2026" },
  { id: "inv_003", date: "2026-02-01", amount: "$19.00", status: "paid", description: "Pro plan — February 2026" },
];

const STATUS_STYLE: Record<Invoice["status"], { bg: string; color: string }> = {
  paid: { bg: "rgba(21,190,83,0.1)", color: "#108c3d" },
  pending: { bg: "rgba(192,133,50,0.1)", color: "#c08532" },
  failed: { bg: "rgba(207,45,86,0.1)", color: "#cf2d56" },
};

// ── Usage meter ───────────────────────────────────────────────────────────────

function UsageMeter({
  label,
  used,
  total,
  unit,
}: {
  label: string;
  used: number;
  total: number;
  unit: string;
}) {
  const pct = Math.min((used / total) * 100, 100);
  const color = pct > 90 ? "#cf2d56" : pct > 70 ? "#c08532" : "#1f8a65";

  function fmt(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
    return String(n);
  }

  return (
    <div>
      <div className="flex justify-between mb-1.5">
        <span
          style={{
            fontFamily: "system-ui",
            fontSize: "12px",
            fontWeight: 500,
            color: "rgba(38,37,30,0.6)",
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "12px",
            color: "rgba(38,37,30,0.5)",
          }}
        >
          {fmt(used)} / {fmt(total)} {unit}
        </span>
      </div>
      <div
        className="rounded-full overflow-hidden"
        style={{ height: "6px", background: "rgba(38,37,30,0.08)" }}
      >
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

// ── Plan comparison card ──────────────────────────────────────────────────────

function PlanCard({
  plan,
  current,
  onSelect,
}: {
  plan: (typeof PLANS)[number];
  current: boolean;
  onSelect: () => void;
}) {
  return (
    <div
      className="rounded-[8px] p-5 flex flex-col"
      style={{
        background: current ? "rgba(245,78,0,0.04)" : "#ffffff",
        border: `1px solid ${current ? "#f54e00" : "rgba(38,37,30,0.1)"}`,
      }}
    >
      <div className="flex items-start justify-between mb-3">
        <div>
          <p
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "15px",
              fontWeight: 400,
              color: "#26251e",
            }}
          >
            {plan.name}
          </p>
          <div className="flex items-baseline gap-1">
            <span
              style={{
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "22px",
                fontWeight: 400,
                color: "#26251e",
                letterSpacing: "-0.44px",
              }}
            >
              {plan.price}
            </span>
            {plan.period && (
              <span
                style={{
                  fontFamily: "system-ui",
                  fontSize: "12px",
                  color: "rgba(38,37,30,0.45)",
                }}
              >
                {plan.period}
              </span>
            )}
          </div>
        </div>
        {current && (
          <span
            className="px-2 py-0.5 rounded-[4px]"
            style={{
              background: "rgba(245,78,0,0.1)",
              color: "#f54e00",
              fontFamily: "system-ui",
              fontSize: "10px",
              fontWeight: 600,
            }}
          >
            CURRENT
          </span>
        )}
      </div>

      <ul className="flex-1 space-y-1.5 mb-4">
        {plan.features.map((f) => (
          <li
            key={f}
            className="flex items-center gap-1.5"
            style={{
              fontFamily: "system-ui",
              fontSize: "12px",
              color: "rgba(38,37,30,0.6)",
            }}
          >
            <span style={{ color: "#1f8a65" }}>✓</span> {f}
          </li>
        ))}
      </ul>

      {!current && (
        <button
          onClick={onSelect}
          className="w-full transition-all"
          style={{
            background: "#ebeae5",
            border: "none",
            borderRadius: "8px",
            padding: "8px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            color: "#26251e",
            cursor: "pointer",
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.color = "#cf2d56";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = "#26251e";
          }}
        >
          {plan.id === "free" ? "Downgrade" : "Upgrade"}
        </button>
      )}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function BillingPage() {
  const user = useAuthStore((s) => s.user);
  const currentPlan = user?.plan ?? "free";

  const [invoices] = useState<Invoice[]>(MOCK_INVOICES);

  return (
    <div className="max-w-[700px] space-y-5">
      {/* Current plan summary */}
      <div
        className="rounded-[8px] p-5"
        style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
      >
        <div className="flex items-start justify-between mb-5">
          <div>
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.048px",
                color: "rgba(38,37,30,0.45)",
                marginBottom: "4px",
              }}
            >
              Current plan
            </p>
            <p
              style={{
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "20px",
                fontWeight: 400,
                color: "#26251e",
                letterSpacing: "-0.2px",
              }}
            >
              {currentPlan.charAt(0).toUpperCase() + currentPlan.slice(1)}
            </p>
            {currentPlan !== "free" && (
              <p
                style={{
                  fontFamily: "system-ui",
                  fontSize: "12px",
                  color: "rgba(38,37,30,0.45)",
                  marginTop: "2px",
                }}
              >
                Renews on May 1, 2026
              </p>
            )}
          </div>
          <button
            className="transition-opacity hover:opacity-90"
            style={{
              background: "#f54e00",
              color: "#ffffff",
              border: "none",
              borderRadius: "8px",
              padding: "9px 16px",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "13px",
              cursor: "pointer",
            }}
          >
            {currentPlan === "free" ? "Upgrade plan" : "Manage subscription"}
          </button>
        </div>

        {/* Usage meters */}
        <div className="space-y-4">
          <UsageMeter label="Generations" used={47} total={200} unit="" />
          <UsageMeter label="Rows generated" used={3_600_000} total={10_000_000} unit="rows" />
          <UsageMeter label="Storage" used={15_200_000} total={10 * 1024 * 1024 * 1024} unit="GB" />
        </div>
      </div>

      {/* Plan comparison */}
      <div>
        <p
          className="mb-3"
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.048px",
            color: "rgba(38,37,30,0.45)",
          }}
        >
          Available plans
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {PLANS.map((plan) => (
            <PlanCard
              key={plan.id}
              plan={plan}
              current={plan.id === currentPlan}
              onSelect={() => {
                window.location.href = "/pricing";
              }}
            />
          ))}
        </div>
      </div>

      {/* Invoice history */}
      <div
        className="rounded-[8px] overflow-hidden"
        style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
      >
        <div
          className="px-5 py-4"
          style={{ borderBottom: "1px solid rgba(38,37,30,0.08)" }}
        >
          <p
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.048px",
              color: "rgba(38,37,30,0.45)",
            }}
          >
            Invoice history
          </p>
        </div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid rgba(38,37,30,0.08)" }}>
              {["Date", "Description", "Amount", "Status", ""].map((h) => (
                <th
                  key={h}
                  style={{
                    fontFamily: "system-ui",
                    fontSize: "11px",
                    fontWeight: 600,
                    color: "rgba(38,37,30,0.45)",
                    textTransform: "uppercase",
                    letterSpacing: "0.048px",
                    padding: "10px 14px",
                    textAlign: "left",
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {invoices.map((inv, i) => {
              const st = STATUS_STYLE[inv.status];
              return (
                <tr
                  key={inv.id}
                  style={{
                    borderBottom: i < invoices.length - 1 ? "1px solid rgba(38,37,30,0.06)" : "none",
                  }}
                >
                  <td
                    style={{
                      padding: "12px 14px",
                      fontFamily: "var(--font-mono, monospace)",
                      fontSize: "12px",
                      color: "rgba(38,37,30,0.5)",
                    }}
                  >
                    {inv.date}
                  </td>
                  <td
                    style={{
                      padding: "12px 14px",
                      fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                      fontSize: "13px",
                      color: "#26251e",
                    }}
                  >
                    {inv.description}
                  </td>
                  <td
                    style={{
                      padding: "12px 14px",
                      fontFamily: "var(--font-mono, monospace)",
                      fontSize: "13px",
                      color: "#26251e",
                      fontVariantNumeric: "tabular-nums",
                    }}
                  >
                    {inv.amount}
                  </td>
                  <td style={{ padding: "12px 14px" }}>
                    <span
                      className="px-2 py-0.5 rounded-[4px] capitalize"
                      style={{
                        background: st.bg,
                        color: st.color,
                        fontFamily: "system-ui",
                        fontSize: "11px",
                        fontWeight: 500,
                      }}
                    >
                      {inv.status}
                    </span>
                  </td>
                  <td style={{ padding: "12px 14px", textAlign: "right" }}>
                    <button
                      className="flex items-center gap-1 ml-auto transition-all"
                      style={{
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        fontFamily: "system-ui",
                        fontSize: "12px",
                        color: "rgba(38,37,30,0.4)",
                      }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.color = "#f54e00";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.color = "rgba(38,37,30,0.4)";
                      }}
                    >
                      <Download size={12} strokeWidth={1.5} />
                      PDF
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
