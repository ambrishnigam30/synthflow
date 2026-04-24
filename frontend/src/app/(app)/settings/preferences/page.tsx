"use client";

import { useState, useEffect } from "react";

type DefaultFormat = "csv" | "json" | "parquet" | "excel";

interface Preferences {
  defaultRowCount: number;
  defaultFormat: DefaultFormat;
  emailOnComplete: boolean;
  emailOnError: boolean;
  showPhaseAnimations: boolean;
  compactMode: boolean;
}

const STORAGE_KEY = "sf_preferences";

function loadPrefs(): Preferences {
  if (typeof window === "undefined") return defaultPrefs();
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...defaultPrefs(), ...JSON.parse(raw) } as Preferences;
  } catch {
    // ignore
  }
  return defaultPrefs();
}

function defaultPrefs(): Preferences {
  return {
    defaultRowCount: 5000,
    defaultFormat: "csv",
    emailOnComplete: false,
    emailOnError: true,
    showPhaseAnimations: true,
    compactMode: false,
  };
}

export default function PreferencesPage() {
  const [prefs, setPrefs] = useState<Preferences>(defaultPrefs);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setPrefs(loadPrefs());
  }, []);

  function setP<K extends keyof Preferences>(key: K, val: Preferences[K]) {
    setPrefs((p) => ({ ...p, [key]: val }));
  }

  function handleSave() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prefs));
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  }

  return (
    <div className="max-w-[520px] space-y-4">
      {/* Generation defaults */}
      <SectionCard title="Generation defaults">
        <Field label="Default row count">
          <input
            type="number"
            min={1}
            max={10000000}
            value={prefs.defaultRowCount}
            onChange={(e) => setP("defaultRowCount", Number(e.target.value))}
            style={{
              background: "transparent",
              border: "1px solid rgba(38,37,30,0.12)",
              borderRadius: "8px",
              padding: "8px 12px",
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "14px",
              color: "#26251e",
              outline: "none",
              width: "140px",
            }}
          />
        </Field>
        <Field label="Default output format">
          <select
            value={prefs.defaultFormat}
            onChange={(e) => setP("defaultFormat", e.target.value as DefaultFormat)}
            style={{
              background: "transparent",
              border: "1px solid rgba(38,37,30,0.12)",
              borderRadius: "8px",
              padding: "8px 12px",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "14px",
              color: "#26251e",
              outline: "none",
              cursor: "pointer",
            }}
          >
            {["csv", "json", "parquet", "excel"].map((f) => (
              <option key={f} value={f}>{f.toUpperCase()}</option>
            ))}
          </select>
        </Field>
      </SectionCard>

      {/* Display */}
      <SectionCard title="Display">
        <Field label="Compact mode">
          <Toggle
            value={prefs.compactMode}
            onChange={(v) => setP("compactMode", v)}
            label="Reduce spacing and font sizes"
          />
        </Field>
        <Field label="Phase animations">
          <Toggle
            value={prefs.showPhaseAnimations}
            onChange={(v) => setP("showPhaseAnimations", v)}
            label="Show generation phase progress animations"
          />
        </Field>
      </SectionCard>

      {/* Notifications */}
      <SectionCard title="Notifications">
        <Field label="On generation complete">
          <Toggle
            value={prefs.emailOnComplete}
            onChange={(v) => setP("emailOnComplete", v)}
            label="Send email when dataset finishes"
          />
        </Field>
        <Field label="On generation error">
          <Toggle
            value={prefs.emailOnError}
            onChange={(v) => setP("emailOnError", v)}
            label="Send email when generation fails"
          />
        </Field>
      </SectionCard>

      <button
        onClick={handleSave}
        className="transition-opacity hover:opacity-90"
        style={{
          background: "#f54e00",
          color: "#ffffff",
          border: "none",
          borderRadius: "8px",
          padding: "10px 20px",
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "13px",
          cursor: "pointer",
        }}
      >
        {saved ? "Saved ✓" : "Save preferences"}
      </button>
    </div>
  );
}

// ── Shared sub-components ─────────────────────────────────────────────────────

function SectionCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div
      className="rounded-[8px] p-5"
      style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
    >
      <p
        className="mb-4"
        style={{
          fontFamily: "system-ui",
          fontSize: "11px",
          fontWeight: 600,
          textTransform: "uppercase",
          letterSpacing: "0.048px",
          color: "rgba(38,37,30,0.45)",
        }}
      >
        {title}
      </p>
      <div className="space-y-4">{children}</div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "14px",
          color: "#26251e",
          flex: 1,
        }}
      >
        {label}
      </span>
      {children}
    </div>
  );
}

function Toggle({
  value,
  onChange,
  label,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  label?: string;
}) {
  return (
    <div className="flex items-center gap-3">
      {label && (
        <span
          style={{
            fontFamily: "system-ui",
            fontSize: "12px",
            color: "rgba(38,37,30,0.5)",
          }}
        >
          {label}
        </span>
      )}
      <button
        role="switch"
        aria-checked={value}
        onClick={() => onChange(!value)}
        style={{
          width: "36px",
          height: "20px",
          borderRadius: "9999px",
          background: value ? "#f54e00" : "rgba(38,37,30,0.15)",
          border: "none",
          cursor: "pointer",
          position: "relative",
          flexShrink: 0,
          transition: "background 150ms ease",
        }}
      >
        <span
          style={{
            position: "absolute",
            top: "2px",
            left: value ? "18px" : "2px",
            width: "16px",
            height: "16px",
            borderRadius: "50%",
            background: "#ffffff",
            transition: "left 150ms ease",
            boxShadow: "0 1px 3px rgba(0,0,0,0.2)",
          }}
        />
      </button>
    </div>
  );
}
