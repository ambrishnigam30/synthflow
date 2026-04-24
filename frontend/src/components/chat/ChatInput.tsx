"use client";

import { useState, useRef, useEffect } from "react";
import { Send, ChevronDown, ChevronRight } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface ChatInputOptions {
  rowCount?: number;
  seed?: number;
  outputFormat?: "csv" | "json" | "parquet" | "excel";
  scenario?: string;
}

interface ChatInputProps {
  onSend: (content: string, options: ChatInputOptions) => void;
  disabled?: boolean;
  placeholder?: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const FORMAT_OPTIONS = ["csv", "json", "parquet", "excel"] as const;

// ── Component ─────────────────────────────────────────────────────────────────

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Describe the dataset you need…",
}: ChatInputProps) {
  const [value, setValue] = useState("");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [options, setOptions] = useState<ChatInputOptions>({
    rowCount: undefined,
    seed: undefined,
    outputFormat: "csv",
    scenario: "",
  });
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 180)}px`;
  }, [value]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed, options);
    setValue("");
    // Reset height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  function setOpt<K extends keyof ChatInputOptions>(key: K, val: ChatInputOptions[K]) {
    setOptions((prev) => ({ ...prev, [key]: val }));
  }

  const canSend = value.trim().length > 0 && !disabled;

  return (
    <div
      className="rounded-[8px] overflow-hidden"
      style={{
        background: "#ffffff",
        border: "1px solid rgba(38,37,30,0.12)",
        boxShadow: "rgba(0,0,0,0.04) 0px 2px 8px",
      }}
    >
      {/* Textarea row */}
      <div className="flex items-end gap-2 px-4 pt-3 pb-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          className="flex-1 resize-none"
          style={{
            background: "transparent",
            border: "none",
            outline: "none",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "15px",
            fontWeight: 400,
            color: "#26251e",
            lineHeight: 1.5,
            minHeight: "24px",
          }}
        />
        <button
          onClick={submit}
          disabled={!canSend}
          className="flex-shrink-0 flex items-center justify-center transition-opacity"
          style={{
            width: "34px",
            height: "34px",
            borderRadius: "8px",
            background: canSend ? "#f54e00" : "rgba(38,37,30,0.08)",
            border: "none",
            cursor: canSend ? "pointer" : "not-allowed",
            color: canSend ? "#ffffff" : "rgba(38,37,30,0.3)",
            opacity: disabled ? 0.5 : 1,
          }}
        >
          <Send size={15} strokeWidth={1.5} />
        </button>
      </div>

      {/* Toolbar row */}
      <div
        className="flex items-center justify-between px-4 pb-2.5"
        style={{ borderTop: "1px solid rgba(38,37,30,0.06)" }}
      >
        <button
          onClick={() => setShowAdvanced((v) => !v)}
          className="flex items-center gap-1 transition-opacity hover:opacity-70"
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            fontFamily: "system-ui",
            fontSize: "11px",
            fontWeight: 500,
            color: "rgba(38,37,30,0.45)",
            letterSpacing: "0.02px",
            padding: "4px 0",
          }}
        >
          {showAdvanced ? (
            <ChevronDown size={12} strokeWidth={2} />
          ) : (
            <ChevronRight size={12} strokeWidth={2} />
          )}
          Advanced options
        </button>
        <span
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            color: "rgba(38,37,30,0.3)",
          }}
        >
          ↵ Send · ⇧↵ Newline
        </span>
      </div>

      {/* Advanced panel */}
      {showAdvanced && (
        <div
          className="px-4 pb-4 grid grid-cols-2 gap-3"
          style={{ borderTop: "1px solid rgba(38,37,30,0.06)" }}
        >
          {/* Row count */}
          <label className="flex flex-col gap-1">
            <span
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 500,
                color: "rgba(38,37,30,0.55)",
                letterSpacing: "0.048px",
                textTransform: "uppercase",
              }}
            >
              Row count
            </span>
            <input
              type="number"
              min={1}
              max={10000000}
              placeholder="e.g. 5000"
              value={options.rowCount ?? ""}
              onChange={(e) =>
                setOpt("rowCount", e.target.value ? Number(e.target.value) : undefined)
              }
              style={{
                background: "transparent",
                border: "1px solid rgba(38,37,30,0.12)",
                borderRadius: "6px",
                padding: "6px 8px",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "13px",
                color: "#26251e",
                outline: "none",
              }}
              onFocus={(e) => {
                e.target.style.borderColor = "rgba(38,37,30,0.25)";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "rgba(38,37,30,0.12)";
              }}
            />
          </label>

          {/* Seed */}
          <label className="flex flex-col gap-1">
            <span
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 500,
                color: "rgba(38,37,30,0.55)",
                letterSpacing: "0.048px",
                textTransform: "uppercase",
              }}
            >
              Seed
            </span>
            <input
              type="number"
              min={0}
              placeholder="e.g. 42"
              value={options.seed ?? ""}
              onChange={(e) =>
                setOpt("seed", e.target.value ? Number(e.target.value) : undefined)
              }
              style={{
                background: "transparent",
                border: "1px solid rgba(38,37,30,0.12)",
                borderRadius: "6px",
                padding: "6px 8px",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "13px",
                color: "#26251e",
                outline: "none",
              }}
              onFocus={(e) => {
                e.target.style.borderColor = "rgba(38,37,30,0.25)";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "rgba(38,37,30,0.12)";
              }}
            />
          </label>

          {/* Output format */}
          <label className="flex flex-col gap-1">
            <span
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 500,
                color: "rgba(38,37,30,0.55)",
                letterSpacing: "0.048px",
                textTransform: "uppercase",
              }}
            >
              Output format
            </span>
            <select
              value={options.outputFormat ?? "csv"}
              onChange={(e) =>
                setOpt("outputFormat", e.target.value as ChatInputOptions["outputFormat"])
              }
              style={{
                background: "transparent",
                border: "1px solid rgba(38,37,30,0.12)",
                borderRadius: "6px",
                padding: "6px 8px",
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "13px",
                color: "#26251e",
                outline: "none",
                cursor: "pointer",
              }}
            >
              {FORMAT_OPTIONS.map((f) => (
                <option key={f} value={f}>
                  {f.toUpperCase()}
                </option>
              ))}
            </select>
          </label>

          {/* Scenario */}
          <label className="flex flex-col gap-1">
            <span
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 500,
                color: "rgba(38,37,30,0.55)",
                letterSpacing: "0.048px",
                textTransform: "uppercase",
              }}
            >
              Scenario
            </span>
            <input
              type="text"
              placeholder="e.g. high fraud rate"
              value={options.scenario ?? ""}
              onChange={(e) => setOpt("scenario", e.target.value)}
              style={{
                background: "transparent",
                border: "1px solid rgba(38,37,30,0.12)",
                borderRadius: "6px",
                padding: "6px 8px",
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "13px",
                color: "#26251e",
                outline: "none",
              }}
              onFocus={(e) => {
                e.target.style.borderColor = "rgba(38,37,30,0.25)";
              }}
              onBlur={(e) => {
                e.target.style.borderColor = "rgba(38,37,30,0.12)";
              }}
            />
          </label>
        </div>
      )}
    </div>
  );
}
