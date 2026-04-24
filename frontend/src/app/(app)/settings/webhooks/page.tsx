"use client";

import { useState } from "react";
import { Plus, X, ChevronDown, ChevronRight, Send } from "lucide-react";
import { useAuthStore } from "@/lib/stores/authStore";

// ── Types ─────────────────────────────────────────────────────────────────────

type WebhookEvent = "generation.completed" | "generation.failed" | "dataset.uploaded";

interface WebhookDelivery {
  id: string;
  timestamp: string;
  event: WebhookEvent;
  statusCode: number;
  success: boolean;
}

interface Webhook {
  id: string;
  url: string;
  events: WebhookEvent[];
  active: boolean;
  secret: string;
  lastTriggered: string | null;
  failureCount: number;
  deliveries: WebhookDelivery[];
}

// ── Mock data ─────────────────────────────────────────────────────────────────

const MOCK_WEBHOOKS: Webhook[] = [
  {
    id: "wh1",
    url: "https://api.myapp.com/webhooks/synthflow",
    events: ["generation.completed"],
    active: true,
    secret: "whsec_abc123",
    lastTriggered: "2026-04-24T14:32:00",
    failureCount: 0,
    deliveries: [
      {
        id: "d1",
        timestamp: "2026-04-24T14:32:00",
        event: "generation.completed",
        statusCode: 200,
        success: true,
      },
      {
        id: "d2",
        timestamp: "2026-04-23T10:15:00",
        event: "generation.completed",
        statusCode: 200,
        success: true,
      },
    ],
  },
  {
    id: "wh2",
    url: "https://hooks.slack.com/services/T000/B000/xxx",
    events: ["generation.completed", "generation.failed"],
    active: false,
    secret: "whsec_xyz789",
    lastTriggered: "2026-04-20T09:00:00",
    failureCount: 2,
    deliveries: [
      {
        id: "d3",
        timestamp: "2026-04-20T09:00:00",
        event: "generation.failed",
        statusCode: 500,
        success: false,
      },
    ],
  },
];

const ALL_EVENTS: { id: WebhookEvent; label: string; description: string }[] = [
  { id: "generation.completed", label: "generation.completed", description: "Fires when data generation succeeds" },
  { id: "generation.failed", label: "generation.failed", description: "Fires when data generation fails" },
  { id: "dataset.uploaded", label: "dataset.uploaded", description: "Fires when a dataset is uploaded" },
];

// ── Add webhook form ──────────────────────────────────────────────────────────

function AddWebhookModal({
  onClose,
  onAdd,
}: {
  onClose: () => void;
  onAdd: (url: string, events: WebhookEvent[], secret: string) => void;
}) {
  const [url, setUrl] = useState("");
  const [events, setEvents] = useState<WebhookEvent[]>(["generation.completed"]);
  const [creating, setCreating] = useState(false);
  const [newSecret] = useState(`whsec_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`);
  const [secretCopied, setSecretCopied] = useState(false);

  function toggleEvent(e: WebhookEvent) {
    setEvents((prev) =>
      prev.includes(e) ? prev.filter((x) => x !== e) : [...prev, e]
    );
  }

  async function handleSubmit(ev: React.FormEvent) {
    ev.preventDefault();
    if (!url.trim() || !events.length) return;
    setCreating(true);
    await new Promise((r) => setTimeout(r, 700));
    onAdd(url.trim(), events, newSecret);
    setCreating(false);
  }

  function copySecret() {
    void navigator.clipboard.writeText(newSecret);
    setSecretCopied(true);
    setTimeout(() => setSecretCopied(false), 2000);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(38,37,30,0.4)" }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-[480px] rounded-[8px] p-6"
        style={{
          background: "#ffffff",
          boxShadow: "rgba(0,0,0,0.14) 0px 28px 70px, rgba(0,0,0,0.1) 0px 14px 32px",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-5">
          <h2
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "16px",
              fontWeight: 400,
              color: "#26251e",
            }}
          >
            Add webhook
          </h2>
          <button
            onClick={onClose}
            style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(38,37,30,0.4)" }}
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          {/* URL */}
          <div>
            <label
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.048px",
                color: "rgba(38,37,30,0.45)",
                display: "block",
                marginBottom: "6px",
              }}
            >
              Endpoint URL
            </label>
            <input
              type="url"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              required
              placeholder="https://yourapp.com/webhooks"
              style={{
                width: "100%",
                background: "transparent",
                border: "1px solid rgba(38,37,30,0.12)",
                borderRadius: "8px",
                padding: "9px 12px",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "13px",
                color: "#26251e",
                outline: "none",
                boxSizing: "border-box",
              }}
              onFocus={(e) => { e.target.style.borderColor = "rgba(38,37,30,0.25)"; }}
              onBlur={(e) => { e.target.style.borderColor = "rgba(38,37,30,0.12)"; }}
            />
          </div>

          {/* Events */}
          <div>
            <label
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.048px",
                color: "rgba(38,37,30,0.45)",
                display: "block",
                marginBottom: "6px",
              }}
            >
              Events
            </label>
            <div className="space-y-2">
              {ALL_EVENTS.map((ev) => (
                <label
                  key={ev.id}
                  className="flex items-center gap-3 cursor-pointer px-3 py-2 rounded-[6px]"
                  style={{
                    border: `1px solid ${events.includes(ev.id) ? "rgba(245,78,0,0.3)" : "rgba(38,37,30,0.1)"}`,
                    background: events.includes(ev.id) ? "rgba(245,78,0,0.04)" : "transparent",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={events.includes(ev.id)}
                    onChange={() => toggleEvent(ev.id)}
                    style={{ accentColor: "#f54e00" }}
                  />
                  <div>
                    <p
                      style={{
                        fontFamily: "var(--font-mono, monospace)",
                        fontSize: "12px",
                        color: "#26251e",
                      }}
                    >
                      {ev.label}
                    </p>
                    <p
                      style={{
                        fontFamily: "system-ui",
                        fontSize: "11px",
                        color: "rgba(38,37,30,0.45)",
                      }}
                    >
                      {ev.description}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Secret (shown once) */}
          <div
            className="rounded-[8px] p-3"
            style={{ background: "#f7f7f4", border: "1px solid rgba(38,37,30,0.08)" }}
          >
            <p
              className="mb-1"
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 600,
                color: "rgba(38,37,30,0.5)",
                textTransform: "uppercase",
                letterSpacing: "0.048px",
              }}
            >
              Signing secret — save this now
            </p>
            <div className="flex items-center gap-2">
              <code
                style={{
                  flex: 1,
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "12px",
                  color: "#26251e",
                  wordBreak: "break-all",
                }}
              >
                {newSecret}
              </code>
              <button
                type="button"
                onClick={copySecret}
                style={{
                  background: "#ebeae5",
                  border: "none",
                  borderRadius: "6px",
                  padding: "5px 10px",
                  fontFamily: "system-ui",
                  fontSize: "11px",
                  color: secretCopied ? "#1f8a65" : "#26251e",
                  cursor: "pointer",
                  flexShrink: 0,
                }}
              >
                {secretCopied ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>

          <button
            type="submit"
            disabled={creating || !url.trim() || !events.length}
            className="w-full transition-opacity hover:opacity-90"
            style={{
              background: "#f54e00",
              color: "#ffffff",
              border: "none",
              borderRadius: "8px",
              padding: "10px",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "14px",
              cursor: creating ? "not-allowed" : "pointer",
              opacity: (!url.trim() || !events.length) ? 0.5 : 1,
            }}
          >
            {creating ? "Adding…" : "Add webhook"}
          </button>
        </form>
      </div>
    </div>
  );
}

// ── Webhook row ───────────────────────────────────────────────────────────────

function WebhookRow({
  webhook,
  onToggle,
  onDelete,
  onTest,
}: {
  webhook: Webhook;
  onToggle: () => void;
  onDelete: () => void;
  onTest: () => void;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      style={{
        borderBottom: "1px solid rgba(38,37,30,0.08)",
      }}
    >
      {/* Main row */}
      <div
        className="flex items-start gap-3 px-4 py-4 cursor-pointer"
        onClick={() => setExpanded((v) => !v)}
      >
        <button
          onClick={(e) => { e.stopPropagation(); setExpanded((v) => !v); }}
          style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(38,37,30,0.4)", padding: "2px", flexShrink: 0 }}
        >
          {expanded ? <ChevronDown size={14} strokeWidth={1.5} /> : <ChevronRight size={14} strokeWidth={1.5} />}
        </button>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <code
              className="truncate"
              style={{
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "13px",
                color: "#26251e",
              }}
            >
              {webhook.url}
            </code>
            <span
              className="px-1.5 py-0.5 rounded-[4px]"
              style={{
                background: webhook.active ? "rgba(21,190,83,0.1)" : "rgba(38,37,30,0.06)",
                color: webhook.active ? "#108c3d" : "rgba(38,37,30,0.45)",
                fontFamily: "system-ui",
                fontSize: "10px",
                fontWeight: 500,
                flexShrink: 0,
              }}
            >
              {webhook.active ? "Active" : "Inactive"}
            </span>
            {webhook.failureCount > 0 && (
              <span
                style={{
                  fontFamily: "system-ui",
                  fontSize: "11px",
                  color: "#cf2d56",
                  flexShrink: 0,
                }}
              >
                {webhook.failureCount} failure{webhook.failureCount > 1 ? "s" : ""}
              </span>
            )}
          </div>
          <div className="flex flex-wrap gap-1.5 mt-1.5">
            {webhook.events.map((ev) => (
              <span
                key={ev}
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "10px",
                  color: "#f54e00",
                  background: "rgba(245,78,0,0.08)",
                  borderRadius: "3px",
                  padding: "1px 5px",
                }}
              >
                {ev}
              </span>
            ))}
          </div>
          {webhook.lastTriggered && (
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                color: "rgba(38,37,30,0.35)",
                marginTop: "4px",
              }}
            >
              Last triggered: {new Date(webhook.lastTriggered).toLocaleString("en-IN")}
            </p>
          )}
        </div>

        <div className="flex items-center gap-2 flex-shrink-0" onClick={(e) => e.stopPropagation()}>
          <button
            onClick={onTest}
            className="flex items-center gap-1 transition-all"
            style={{
              background: "#ebeae5",
              border: "none",
              borderRadius: "6px",
              padding: "5px 10px",
              fontFamily: "system-ui",
              fontSize: "11px",
              color: "#26251e",
              cursor: "pointer",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = "#cf2d56"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = "#26251e"; }}
          >
            <Send size={10} strokeWidth={1.5} />
            Test
          </button>
          <button
            onClick={onToggle}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              fontFamily: "system-ui",
              fontSize: "11px",
              color: "rgba(38,37,30,0.45)",
            }}
          >
            {webhook.active ? "Disable" : "Enable"}
          </button>
          <button
            onClick={onDelete}
            style={{
              background: "none",
              border: "none",
              cursor: "pointer",
              color: "rgba(38,37,30,0.35)",
            }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = "#cf2d56"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = "rgba(38,37,30,0.35)"; }}
          >
            <X size={14} strokeWidth={1.5} />
          </button>
        </div>
      </div>

      {/* Deliveries */}
      {expanded && webhook.deliveries.length > 0 && (
        <div
          className="px-12 pb-3"
          style={{ borderTop: "1px solid rgba(38,37,30,0.06)" }}
        >
          <p
            className="py-2"
            style={{
              fontFamily: "system-ui",
              fontSize: "10px",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.048px",
              color: "rgba(38,37,30,0.4)",
            }}
          >
            Recent deliveries
          </p>
          {webhook.deliveries.map((d) => (
            <div
              key={d.id}
              className="flex items-center gap-3 py-1.5"
              style={{ borderTop: "1px solid rgba(38,37,30,0.04)" }}
            >
              <span
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "11px",
                  color: "rgba(38,37,30,0.4)",
                  width: "140px",
                  flexShrink: 0,
                }}
              >
                {new Date(d.timestamp).toLocaleString("en-IN")}
              </span>
              <span
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "11px",
                  color: "#f54e00",
                  flex: 1,
                }}
              >
                {d.event}
              </span>
              <span
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "11px",
                  color: d.success ? "#1f8a65" : "#cf2d56",
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {d.statusCode}
              </span>
              <span
                style={{
                  fontFamily: "system-ui",
                  fontSize: "11px",
                  color: d.success ? "#1f8a65" : "#cf2d56",
                  fontWeight: 500,
                  width: "50px",
                  textAlign: "right",
                }}
              >
                {d.success ? "✓" : "✗"}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function WebhooksPage() {
  const user = useAuthStore((s) => s.user);
  const isBusinessPlan = user?.plan === "enterprise";

  const [webhooks, setWebhooks] = useState<Webhook[]>(isBusinessPlan ? MOCK_WEBHOOKS : []);
  const [showAdd, setShowAdd] = useState(false);

  function handleAdd(url: string, events: WebhookEvent[], secret: string) {
    const wh: Webhook = {
      id: crypto.randomUUID(),
      url,
      events,
      active: true,
      secret,
      lastTriggered: null,
      failureCount: 0,
      deliveries: [],
    };
    setWebhooks((prev) => [wh, ...prev]);
    setShowAdd(false);
  }

  function handleToggle(id: string) {
    setWebhooks((prev) =>
      prev.map((w) => w.id === id ? { ...w, active: !w.active } : w)
    );
  }

  function handleDelete(id: string) {
    setWebhooks((prev) => prev.filter((w) => w.id !== id));
  }

  function handleTest(id: string) {
    setWebhooks((prev) =>
      prev.map((w) =>
        w.id === id
          ? {
              ...w,
              lastTriggered: new Date().toISOString(),
              deliveries: [
                {
                  id: crypto.randomUUID(),
                  timestamp: new Date().toISOString(),
                  event: w.events[0] ?? "generation.completed",
                  statusCode: 200,
                  success: true,
                },
                ...w.deliveries.slice(0, 4),
              ],
            }
          : w
      )
    );
  }

  if (!isBusinessPlan) {
    return (
      <div className="max-w-[560px]">
        <div
          className="rounded-[8px] p-6"
          style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
        >
          <p
            className="mb-2"
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.048px",
              color: "#c08532",
            }}
          >
            Business Plan feature
          </p>
          <p
            className="mb-4"
            style={{
              fontFamily: "var(--font-serif, Georgia, serif)",
              fontSize: "16px",
              color: "rgba(38,37,30,0.6)",
              lineHeight: 1.5,
            }}
          >
            Webhooks let you receive real-time notifications when datasets are generated.
            Available on the Business plan.
          </p>
          <a
            href="/app/settings/billing"
            style={{
              display: "inline-block",
              background: "#f54e00",
              color: "#ffffff",
              borderRadius: "8px",
              padding: "10px 20px",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "13px",
              textDecoration: "none",
            }}
          >
            View plans
          </a>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-[700px]">
      <div className="flex items-center justify-between mb-5">
        <p
          style={{
            fontFamily: "var(--font-serif, Georgia, serif)",
            fontSize: "15px",
            color: "rgba(38,37,30,0.55)",
          }}
        >
          Receive HTTP POST notifications when SynthFlow events occur.
        </p>
        <button
          onClick={() => setShowAdd(true)}
          className="flex items-center gap-1.5 flex-shrink-0 ml-4 transition-opacity hover:opacity-90"
          style={{
            background: "#f54e00",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            padding: "9px 14px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            cursor: "pointer",
            whiteSpace: "nowrap",
          }}
        >
          <Plus size={14} strokeWidth={2} />
          Add webhook
        </button>
      </div>

      {webhooks.length === 0 ? (
        <div
          className="rounded-[8px] flex items-center justify-center py-12"
          style={{
            background: "#ffffff",
            border: "1px dashed rgba(38,37,30,0.2)",
            fontFamily: "system-ui",
            fontSize: "13px",
            color: "rgba(38,37,30,0.35)",
          }}
        >
          No webhooks configured
        </div>
      ) : (
        <div
          className="rounded-[8px] overflow-hidden"
          style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
        >
          {webhooks.map((wh) => (
            <WebhookRow
              key={wh.id}
              webhook={wh}
              onToggle={() => handleToggle(wh.id)}
              onDelete={() => handleDelete(wh.id)}
              onTest={() => handleTest(wh.id)}
            />
          ))}
        </div>
      )}

      {showAdd && (
        <AddWebhookModal onClose={() => setShowAdd(false)} onAdd={handleAdd} />
      )}
    </div>
  );
}
