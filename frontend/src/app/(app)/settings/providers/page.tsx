"use client";

import { useState } from "react";
import { Eye, EyeOff, CheckCircle, Circle, RefreshCw } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Provider {
  id: string;
  name: string;
  logo: string;
  models: string[];
  placeholder: string;
  docsUrl: string;
}

interface ProviderState {
  apiKey: string;
  model: string;
  isDefault: boolean;
  status: "unconfigured" | "connected" | "error";
  testing: boolean;
  saved: boolean;
}

// ── Provider definitions ──────────────────────────────────────────────────────

const PROVIDERS: Provider[] = [
  {
    id: "gemini",
    name: "Google Gemini",
    logo: "G",
    models: ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash"],
    placeholder: "AIza…",
    docsUrl: "#",
  },
  {
    id: "openai",
    name: "OpenAI",
    logo: "⬡",
    models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"],
    placeholder: "sk-…",
    docsUrl: "#",
  },
  {
    id: "groq",
    name: "Groq",
    logo: "G",
    models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    placeholder: "gsk_…",
    docsUrl: "#",
  },
];

// ── Logo mark ─────────────────────────────────────────────────────────────────

function ProviderLogo({ provider }: { provider: Provider }) {
  const colors: Record<string, { bg: string; text: string }> = {
    gemini: { bg: "#4285F4", text: "#ffffff" },
    openai: { bg: "#000000", text: "#ffffff" },
    groq: { bg: "#f54e00", text: "#ffffff" },
  };
  const c = colors[provider.id] ?? { bg: "#ebeae5", text: "#26251e" };
  return (
    <div
      className="w-9 h-9 rounded-[8px] flex items-center justify-center flex-shrink-0"
      style={{ background: c.bg }}
    >
      <span style={{ color: c.text, fontSize: "16px", fontWeight: 600, lineHeight: 1 }}>
        {provider.logo}
      </span>
    </div>
  );
}

// ── Provider card ─────────────────────────────────────────────────────────────

function ProviderCard({
  provider,
  state,
  onChange,
  onTestConnection,
  onSave,
  onSetDefault,
}: {
  provider: Provider;
  state: ProviderState;
  onChange: (patch: Partial<ProviderState>) => void;
  onTestConnection: () => void;
  onSave: () => void;
  onSetDefault: () => void;
}) {
  const [showKey, setShowKey] = useState(false);

  const statusDot =
    state.status === "connected" ? (
      <span
        className="flex items-center gap-1.5"
        style={{ fontFamily: "system-ui", fontSize: "11px", color: "#1f8a65", fontWeight: 500 }}
      >
        <CheckCircle size={12} strokeWidth={2} /> Connected
      </span>
    ) : (
      <span
        className="flex items-center gap-1.5"
        style={{ fontFamily: "system-ui", fontSize: "11px", color: "rgba(38,37,30,0.4)" }}
      >
        <Circle size={12} strokeWidth={1.5} /> Not configured
      </span>
    );

  return (
    <div
      className="rounded-[8px] p-5 mb-4"
      style={{
        background: "#ffffff",
        border: `1px solid ${state.isDefault ? "#f54e00" : "rgba(38,37,30,0.1)"}`,
      }}
    >
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <ProviderLogo provider={provider} />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <p
              style={{
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "15px",
                fontWeight: 400,
                color: "#26251e",
              }}
            >
              {provider.name}
            </p>
            {state.isDefault && (
              <span
                className="px-1.5 py-0.5 rounded-[4px]"
                style={{
                  background: "rgba(245,78,0,0.1)",
                  color: "#f54e00",
                  fontFamily: "system-ui",
                  fontSize: "10px",
                  fontWeight: 600,
                  letterSpacing: "0.04px",
                }}
              >
                DEFAULT
              </span>
            )}
          </div>
          {statusDot}
        </div>
        <button
          onClick={onSetDefault}
          disabled={state.isDefault}
          style={{
            background: "none",
            border: "1px solid rgba(38,37,30,0.12)",
            borderRadius: "6px",
            padding: "5px 10px",
            fontFamily: "system-ui",
            fontSize: "11px",
            color: state.isDefault ? "rgba(38,37,30,0.3)" : "rgba(38,37,30,0.55)",
            cursor: state.isDefault ? "not-allowed" : "pointer",
          }}
        >
          {state.isDefault ? "Default" : "Set as default"}
        </button>
      </div>

      {/* API key */}
      <div className="mb-3">
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
          API Key
        </label>
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <input
              type={showKey ? "text" : "password"}
              value={state.apiKey}
              onChange={(e) => onChange({ apiKey: e.target.value, status: "unconfigured", saved: false })}
              placeholder={provider.placeholder}
              style={{
                width: "100%",
                background: "transparent",
                border: "1px solid rgba(38,37,30,0.12)",
                borderRadius: "8px",
                padding: "9px 36px 9px 12px",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "13px",
                color: "#26251e",
                outline: "none",
                boxSizing: "border-box",
              }}
              onFocus={(e) => { e.target.style.borderColor = "rgba(38,37,30,0.25)"; }}
              onBlur={(e) => { e.target.style.borderColor = "rgba(38,37,30,0.12)"; }}
            />
            <button
              type="button"
              onClick={() => setShowKey((v) => !v)}
              style={{
                position: "absolute",
                right: "10px",
                top: "50%",
                transform: "translateY(-50%)",
                background: "none",
                border: "none",
                cursor: "pointer",
                color: "rgba(38,37,30,0.35)",
                lineHeight: 0,
              }}
            >
              {showKey ? <EyeOff size={14} strokeWidth={1.5} /> : <Eye size={14} strokeWidth={1.5} />}
            </button>
          </div>
        </div>
        <p
          className="mt-1.5"
          style={{ fontFamily: "system-ui", fontSize: "11px", color: "rgba(38,37,30,0.4)" }}
        >
          🔒 Your API keys are encrypted with AES-256 and never shared.
        </p>
      </div>

      {/* Model selector */}
      <div className="mb-4">
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
          Model
        </label>
        <select
          value={state.model}
          onChange={(e) => onChange({ model: e.target.value })}
          style={{
            width: "100%",
            background: "transparent",
            border: "1px solid rgba(38,37,30,0.12)",
            borderRadius: "8px",
            padding: "9px 12px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "14px",
            color: "#26251e",
            outline: "none",
            cursor: "pointer",
          }}
        >
          {provider.models.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={onTestConnection}
          disabled={!state.apiKey || state.testing}
          className="flex items-center gap-1.5 transition-all"
          style={{
            background: "#ebeae5",
            border: "none",
            borderRadius: "8px",
            padding: "9px 14px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            color: !state.apiKey ? "rgba(38,37,30,0.3)" : "#26251e",
            cursor: !state.apiKey ? "not-allowed" : "pointer",
          }}
          onMouseEnter={(e) => {
            if (state.apiKey) (e.currentTarget as HTMLElement).style.color = "#cf2d56";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = !state.apiKey ? "rgba(38,37,30,0.3)" : "#26251e";
          }}
        >
          <RefreshCw size={13} strokeWidth={1.5} className={state.testing ? "animate-spin" : ""} />
          {state.testing ? "Testing…" : "Test connection"}
        </button>

        <button
          onClick={onSave}
          disabled={!state.apiKey}
          className="transition-opacity hover:opacity-90"
          style={{
            background: !state.apiKey ? "rgba(38,37,30,0.2)" : "#f54e00",
            border: "none",
            borderRadius: "8px",
            padding: "9px 16px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            color: "#ffffff",
            cursor: !state.apiKey ? "not-allowed" : "pointer",
          }}
        >
          {state.saved ? "Saved ✓" : "Save"}
        </button>

        {state.status === "error" && (
          <span
            style={{ fontFamily: "system-ui", fontSize: "12px", color: "#cf2d56" }}
          >
            Connection failed
          </span>
        )}
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function ProvidersPage() {
  const [states, setStates] = useState<Record<string, ProviderState>>(
    Object.fromEntries(
      PROVIDERS.map((p, i) => [
        p.id,
        {
          apiKey: "",
          model: p.models[0],
          isDefault: i === 0,
          status: "unconfigured",
          testing: false,
          saved: false,
        },
      ])
    )
  );

  function patch(id: string, update: Partial<ProviderState>) {
    setStates((prev) => ({ ...prev, [id]: { ...prev[id], ...update } }));
  }

  async function handleTest(id: string) {
    patch(id, { testing: true, status: "unconfigured" });
    await new Promise((r) => setTimeout(r, 1200));
    patch(id, { testing: false, status: "connected" });
  }

  async function handleSave(id: string) {
    await new Promise((r) => setTimeout(r, 400));
    patch(id, { saved: true });
    setTimeout(() => patch(id, { saved: false }), 2500);
  }

  function handleSetDefault(id: string) {
    setStates((prev) =>
      Object.fromEntries(
        Object.entries(prev).map(([k, v]) => [k, { ...v, isDefault: k === id }])
      )
    );
  }

  return (
    <div className="max-w-[580px]">
      <p
        className="mb-6"
        style={{
          fontFamily: "var(--font-serif, Georgia, serif)",
          fontSize: "16px",
          color: "rgba(38,37,30,0.55)",
          lineHeight: 1.5,
        }}
      >
        Configure the LLM providers SynthFlow uses for intent analysis and causal reasoning.
        At least one provider must be configured and set as default.
      </p>

      {PROVIDERS.map((provider) => (
        <ProviderCard
          key={provider.id}
          provider={provider}
          state={states[provider.id]}
          onChange={(update) => patch(provider.id, update)}
          onTestConnection={() => void handleTest(provider.id)}
          onSave={() => void handleSave(provider.id)}
          onSetDefault={() => handleSetDefault(provider.id)}
        />
      ))}
    </div>
  );
}
