"use client";

import { useEffect, useState, useCallback } from "react";
import { X, CheckCircle, AlertCircle, Info } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

export type ToastVariant = "success" | "error" | "info";

export interface Toast {
  id: string;
  message: string;
  variant: ToastVariant;
}

// ── Singleton event bus ───────────────────────────────────────────────────────

type ToastListener = (toast: Toast) => void;
type DismissListener = (id: string) => void;

const addListeners = new Set<ToastListener>();
const dismissListeners = new Set<DismissListener>();

let _counter = 0;

function emit(message: string, variant: ToastVariant): string {
  const id = `toast-${++_counter}`;
  const toast: Toast = { id, message, variant };
  addListeners.forEach((fn) => fn(toast));
  return id;
}

// ── Public API ────────────────────────────────────────────────────────────────

export const toast = {
  success: (message: string) => emit(message, "success"),
  error: (message: string) => emit(message, "error"),
  info: (message: string) => emit(message, "info"),
  dismiss: (id: string) => dismissListeners.forEach((fn) => fn(id)),
};

// ── Single toast item ─────────────────────────────────────────────────────────

const VARIANT_STYLES: Record<
  ToastVariant,
  { bg: string; border: string; text: string; icon: React.ReactNode }
> = {
  success: {
    bg: "#f0fdf4",
    border: "rgba(21,190,83,0.3)",
    text: "#1f8a65",
    icon: <CheckCircle size={16} strokeWidth={1.5} />,
  },
  error: {
    bg: "#fff5f6",
    border: "rgba(207,45,86,0.3)",
    text: "#cf2d56",
    icon: <AlertCircle size={16} strokeWidth={1.5} />,
  },
  info: {
    bg: "#f7f7f4",
    border: "rgba(38,37,30,0.15)",
    text: "#26251e",
    icon: <Info size={16} strokeWidth={1.5} />,
  },
};

function ToastItem({
  toast: t,
  onDismiss,
}: {
  toast: Toast;
  onDismiss: (id: string) => void;
}) {
  const s = VARIANT_STYLES[t.variant];

  useEffect(() => {
    const timer = setTimeout(() => onDismiss(t.id), 5000);
    return () => clearTimeout(timer);
  }, [t.id, onDismiss]);

  return (
    <div
      role="alert"
      className="flex items-start gap-2.5 px-4 py-3 rounded-[8px] shadow-md animate-fade-in-up"
      style={{
        background: s.bg,
        border: `1px solid ${s.border}`,
        color: s.text,
        minWidth: "280px",
        maxWidth: "420px",
        boxShadow:
          "rgba(0,0,0,0.1) 0px 4px 12px, rgba(0,0,0,0.06) 0px 2px 4px",
      }}
    >
      <span className="mt-[1px] flex-shrink-0">{s.icon}</span>
      <p
        className="flex-1 text-[13px] leading-[1.4]"
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontWeight: 400,
        }}
      >
        {t.message}
      </p>
      <button
        onClick={() => onDismiss(t.id)}
        className="flex-shrink-0 mt-[1px] transition-opacity hover:opacity-60"
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          color: "inherit",
          padding: 0,
        }}
        aria-label="Dismiss"
      >
        <X size={14} strokeWidth={2} />
      </button>
    </div>
  );
}

// ── Provider (render at root) ─────────────────────────────────────────────────

export function ToastProvider() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  useEffect(() => {
    const addFn: ToastListener = (t) =>
      setToasts((prev) => [...prev, t]);
    const dismissFn: DismissListener = (id) => dismiss(id);

    addListeners.add(addFn);
    dismissListeners.add(dismissFn);
    return () => {
      addListeners.delete(addFn);
      dismissListeners.delete(dismissFn);
    };
  }, [dismiss]);

  if (toasts.length === 0) return null;

  return (
    <div
      aria-live="polite"
      className="fixed bottom-5 right-5 z-[9999] flex flex-col gap-2 items-end"
    >
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
      ))}
    </div>
  );
}
