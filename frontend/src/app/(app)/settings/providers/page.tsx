"use client";

import { useState, useEffect } from "react";
import { Eye, EyeOff, CheckCircle, Circle, RefreshCw } from "lucide-react";
import { llmConfigApi, type LLMProviderConfig } from "@/lib/api";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Provider {
  id: string;
  name: string;
  logo: string;
  models: string[];
  placeholder: string;
}

interface ProviderState {
  /** Key typed by user. Empty = use stored key (don't re-save). */
  apiKey: string;
  /** Masked key from DB, e.g. "••••••••••••abcd". Shown as placeholder when saved. */
  maskedKey: string;
  model: string;
  isDefault: boolean;
  /** Whether a saved config exists in the DB for this provider. */
  isSaved: boolean;
  status: "unconfigured" | "connected" | "error";
  testing: boolean;
  testResult: { success: boolean; message: string } | null;
  saving: boolean;
  /** Brief "Saved ✓" flash after successful save. */
  savedFlash: boolean;
  settingDefault: boolean;
}

// ── Provider definitions ──────────────────────────────────────────────────────

const PROVIDERS: Provider[] = [
  {
    id: "gemini",
    name: "Google Gemini",
    logo: "G",
    models: ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-pro", "gemini-1.5-flash"],
    placeholder: "AIza…",
  },
  {
    id: "openai",
    name: "OpenAI",
    logo: "⬡",
    models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-mini"],
    placeholder: "sk-…",
  },
  {
    id: "groq",
    name: "Groq",
    logo: "G",
    models: ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    placeholder: "gsk_…",
  },
];

function initialState(p: Provider, i: number): ProviderState {
  return {
    apiKey: "",
    maskedKey: "",
    model: p.models[0],
    isDefault: i === 0,
    isSaved: false,
    status: "unconfigured",
    testing: false,
    testResult: null,
    saving: false,
    savedFlash: false,
    settingDefault: false,
  };
}

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
  onTest,
  onSave,
  onSetDefault,
}: {
  provider: Provider;
  state: ProviderState;
  onChange: (patch: Partial<ProviderState>) => void;
  onTest: () => void;
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

  // Determine whether "Save" is valid: requires a key in the input field
  const canSave = state.apiKey.trim().length > 0;
  // "Test" uses the stored DB key — only works if provider is saved
  const canTest = state.isSaved && !state.testing;
  // "Set as default" only works if saved and not already default
  const canSetDefault = state.isSaved && !state.isDefault;

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
          disabled={!canSetDefault || state.settingDefault}
          style={{
            background: "none",
            border: "1px solid rgba(38,37,30,0.12)",
            borderRadius: "6px",
            padding: "5px 10px",
            fontFamily: "system-ui",
            fontSize: "11px",
            color: canSetDefault ? "rgba(38,37,30,0.55)" : "rgba(38,37,30,0.3)",
            cursor: canSetDefault ? "pointer" : "not-allowed",
          }}
        >
          {state.settingDefault ? "Setting…" : state.isDefault ? "Default" : "Set as default"}
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
              onChange={(e) => onChange({ apiKey: e.target.value, testResult: null })}
              placeholder={
                state.isSaved && state.maskedKey
                  ? state.maskedKey
                  : state.isSaved
                  ? "Enter new key to update"
                  : provider.placeholder
              }
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

      {/* Test result */}
      {state.testResult && (
        <div
          className="mb-3 px-3 py-2 rounded-[6px]"
          style={{
            background: state.testResult.success
              ? "rgba(31,138,101,0.08)"
              : "rgba(207,45,86,0.06)",
            border: `1px solid ${state.testResult.success ? "rgba(31,138,101,0.2)" : "rgba(207,45,86,0.2)"}`,
          }}
        >
          <p
            style={{
              fontFamily: "system-ui",
              fontSize: "12px",
              color: state.testResult.success ? "#1f8a65" : "#cf2d56",
            }}
          >
            {state.testResult.success ? "✓ " : "✗ "}
            {state.testResult.message}
          </p>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-2 flex-wrap">
        <button
          onClick={onTest}
          disabled={!canTest}
          className="flex items-center gap-1.5 transition-all"
          style={{
            background: "#ebeae5",
            border: "none",
            borderRadius: "8px",
            padding: "9px 14px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            color: !canTest ? "rgba(38,37,30,0.3)" : "#26251e",
            cursor: !canTest ? "not-allowed" : "pointer",
          }}
          onMouseEnter={(e) => {
            if (canTest) (e.currentTarget as HTMLElement).style.color = "#cf2d56";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = !canTest ? "rgba(38,37,30,0.3)" : "#26251e";
          }}
          title={!state.isSaved ? "Save a key first to test it" : undefined}
        >
          <RefreshCw size={13} strokeWidth={1.5} className={state.testing ? "animate-spin" : ""} />
          {state.testing ? "Testing…" : "Test connection"}
        </button>

        <button
          onClick={onSave}
          disabled={!canSave || state.saving}
          className="transition-opacity hover:opacity-90"
          style={{
            background: !canSave || state.saving ? "rgba(38,37,30,0.2)" : "#f54e00",
            border: "none",
            borderRadius: "8px",
            padding: "9px 16px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            color: "#ffffff",
            cursor: !canSave || state.saving ? "not-allowed" : "pointer",
          }}
        >
          {state.saving ? "Saving…" : state.savedFlash ? "Saved ✓" : "Save"}
        </button>
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function ProvidersPage() {
  const [states, setStates] = useState<Record<string, ProviderState>>(
    Object.fromEntries(PROVIDERS.map((p, i) => [p.id, initialState(p, i)]))
  );
  const [loading, setLoading] = useState(true);

  // Load existing configs from DB on mount
  useEffect(() => {
    llmConfigApi.list().then((configs: LLMProviderConfig[]) => {
      setStates((prev) => {
        const next = { ...prev };
        // Reset all to unconfigured first
        for (const p of PROVIDERS) {
          next[p.id] = { ...next[p.id], isSaved: false, status: "unconfigured", isDefault: false };
        }
        // Apply loaded configs
        for (const cfg of configs) {
          if (next[cfg.provider]) {
            const prov = PROVIDERS.find((p) => p.id === cfg.provider);
            next[cfg.provider] = {
              ...next[cfg.provider],
              model: cfg.model_name ?? prov?.models[0] ?? "",
              isDefault: cfg.is_default,
              isSaved: true,
              status: "connected",
              maskedKey: cfg.masked_key ?? "",
            };
          }
        }
        // If no provider is default but some are saved, make the first saved one default
        const hasSavedDefault = configs.some((c) => c.is_default);
        if (!hasSavedDefault && configs.length > 0) {
          const firstSaved = configs[0].provider;
          if (next[firstSaved]) next[firstSaved].isDefault = true;
        }
        return next;
      });
    }).catch(() => {
      // Silently fall back to unconfigured state
    }).finally(() => setLoading(false));
  }, []);

  function patch(id: string, update: Partial<ProviderState>) {
    setStates((prev) => ({ ...prev, [id]: { ...prev[id], ...update } }));
  }

  async function handleTest(id: string) {
    patch(id, { testing: true, testResult: null });
    try {
      const result = await llmConfigApi.test(id);
      patch(id, {
        testing: false,
        testResult: { success: result.success, message: result.message },
        status: result.success ? "connected" : "error",
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Test failed";
      patch(id, { testing: false, testResult: { success: false, message: msg }, status: "error" });
    }
  }

  async function handleSave(id: string) {
    const state = states[id];
    if (!state.apiKey.trim()) return;
    patch(id, { saving: true });
    try {
      const saved = await llmConfigApi.save(id, state.apiKey.trim(), state.model, state.isDefault);
      patch(id, {
        saving: false,
        savedFlash: true,
        isSaved: true,
        status: "connected",
        apiKey: "", // clear input after save; stored key is now in DB
        maskedKey: saved.masked_key ?? "",
      });
      setTimeout(() => patch(id, { savedFlash: false }), 2500);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Save failed";
      patch(id, { saving: false, testResult: { success: false, message: msg } });
    }
  }

  async function handleSetDefault(id: string) {
    patch(id, { settingDefault: true });
    try {
      await llmConfigApi.setDefault(id);
      // Update all providers: only this one is default now
      setStates((prev) => {
        const next = { ...prev };
        for (const key of Object.keys(next)) {
          next[key] = { ...next[key], isDefault: key === id, settingDefault: false };
        }
        return next;
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Could not set default";
      patch(id, { settingDefault: false, testResult: { success: false, message: msg } });
    }
  }

  if (loading) {
    return (
      <div className="max-w-[580px]">
        <p style={{ fontFamily: "system-ui", fontSize: "13px", color: "rgba(38,37,30,0.4)" }}>
          Loading provider configurations…
        </p>
      </div>
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
          onTest={() => void handleTest(provider.id)}
          onSave={() => void handleSave(provider.id)}
          onSetDefault={() => void handleSetDefault(provider.id)}
        />
      ))}
    </div>
  );
}
