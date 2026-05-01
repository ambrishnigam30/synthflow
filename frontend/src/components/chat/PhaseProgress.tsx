"use client";

import type { GenerationPhase } from "@/lib/stores/generationStore";

// ── Styles ────────────────────────────────────────────────────────────────────

const PHASE_STATUS_STYLES: Record<
  GenerationPhase["status"],
  { icon: string; color: string; animation?: string }
> = {
  done: { icon: "✓", color: "#1f8a65" },
  active: { icon: "●", color: "#f54e00", animation: "pulse 1.5s ease-in-out infinite" },
  pending: { icon: "○", color: "rgba(38,37,30,0.3)" },
  failed: { icon: "✗", color: "#cf2d56" },
};

// ── Pulse animation ───────────────────────────────────────────────────────────

const PULSE_CSS = `
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
`;

// ── Props ─────────────────────────────────────────────────────────────────────

interface PhaseProgressProps {
  phases: GenerationPhase[];
  progress: number; // 0–1
  compact?: boolean;
  isFailed?: boolean;
  failedMessage?: string;
  onRetry?: () => void;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function PhaseProgress({
  phases,
  progress,
  compact = false,
  isFailed = false,
  failedMessage,
  onRetry,
}: PhaseProgressProps) {
  const pct = Math.min(Math.max(progress, 0), 1) * 100;
  const barColor = isFailed ? "#cf2d56" : "#f54e00";
  const labelColor = isFailed ? "#cf2d56" : "rgba(38,37,30,0.5)";

  return (
    <div
      className="rounded-[8px] p-4"
      style={{
        background: "#ebeae5",
        border: `1px solid ${isFailed ? "rgba(207,45,86,0.2)" : "rgba(38,37,30,0.1)"}`,
      }}
    >
      <style>{PULSE_CSS}</style>

      {/* Progress bar */}
      <div className="mb-4">
        <div
          className="rounded-full overflow-hidden"
          style={{ height: "4px", background: "rgba(38,37,30,0.1)" }}
        >
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, background: barColor }}
          />
        </div>
        <div className="flex justify-between mt-1">
          <span
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "11px",
              fontWeight: 500,
              color: labelColor,
              textTransform: "uppercase",
              letterSpacing: "0.048px",
            }}
          >
            {isFailed ? "Generation Failed" : "Generating"}
          </span>
          <span
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "11px",
              color: barColor,
            }}
          >
            {Math.round(pct)}%
          </span>
        </div>
      </div>

      {/* Phase list */}
      <ol
        className={compact ? "grid grid-cols-3 gap-x-4 gap-y-2" : "space-y-2"}
      >
        {phases.map((phase) => {
          const s = PHASE_STATUS_STYLES[phase.status];
          return (
            <li key={phase.index} className="flex items-center gap-2">
              <span
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "12px",
                  color: s.color,
                  animation: s.animation ?? "none",
                  width: "14px",
                  textAlign: "center",
                  flexShrink: 0,
                  lineHeight: 1,
                }}
              >
                {s.icon}
              </span>
              <span
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "11px",
                  fontWeight: 500,
                  color: phase.status === "pending" ? "rgba(38,37,30,0.35)" : s.color,
                  textTransform: "uppercase",
                  letterSpacing: "0.048px",
                  lineHeight: 1.27,
                }}
              >
                {phase.name}
              </span>
            </li>
          );
        })}
      </ol>

      {/* Failure details */}
      {isFailed && failedMessage && (
        <div
          className="mt-4 pt-3"
          style={{ borderTop: "1px solid rgba(207,45,86,0.15)" }}
        >
          <p
            style={{
              fontFamily: "system-ui",
              fontSize: "12px",
              color: "#cf2d56",
              lineHeight: 1.5,
              marginBottom: onRetry ? "10px" : 0,
            }}
          >
            {failedMessage}
          </p>
          {onRetry && (
            <button
              onClick={onRetry}
              style={{
                background: "#ebeae5",
                border: "1px solid rgba(207,45,86,0.3)",
                borderRadius: "6px",
                padding: "6px 12px",
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "12px",
                fontWeight: 400,
                color: "#cf2d56",
                cursor: "pointer",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.background = "rgba(207,45,86,0.08)";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.background = "#ebeae5";
              }}
            >
              Retry
            </button>
          )}
        </div>
      )}
    </div>
  );
}
