import React from "react";

type BadgeVariant = "success" | "error" | "warning" | "info" | "neutral";
type BadgeContext = "marketing" | "app";

interface BadgeProps {
  variant?: BadgeVariant;
  context?: BadgeContext;
  children: React.ReactNode;
  className?: string;
}

const MARKETING_STYLES: Record<BadgeVariant, string> = {
  success: "bg-[rgba(21,190,83,0.2)] text-[#108c3d] border border-[rgba(21,190,83,0.4)]",
  error:   "bg-[rgba(234,34,97,0.12)] text-[#ea2261] border border-[rgba(234,34,97,0.3)]",
  warning: "bg-[rgba(155,104,41,0.12)] text-[#9b6829] border border-[rgba(155,104,41,0.3)]",
  info:    "bg-[rgba(83,58,253,0.1)] text-[#533afd] border border-[rgba(83,58,253,0.3)]",
  neutral: "bg-[rgba(6,27,49,0.06)] text-[#64748d] border border-[#e5edf5]",
};

const APP_STYLES: Record<BadgeVariant, string> = {
  success: "bg-[rgba(31,138,101,0.15)] text-[#1f8a65] border border-[rgba(31,138,101,0.3)]",
  error:   "bg-[rgba(207,45,86,0.12)] text-[#cf2d56] border border-[rgba(207,45,86,0.3)]",
  warning: "bg-[rgba(192,133,50,0.15)] text-[#c08532] border border-[rgba(192,133,50,0.3)]",
  info:    "bg-[rgba(159,187,224,0.3)] text-[#26251e] border border-[rgba(38,37,30,0.1)]",
  neutral: "bg-[#e6e5e0] text-[rgba(38,37,30,0.55)] border border-[rgba(38,37,30,0.1)]",
};

export function Badge({
  variant = "neutral",
  context = "marketing",
  children,
  className = "",
}: BadgeProps) {
  const base =
    context === "marketing"
      ? "inline-flex items-center px-[6px] py-[1px] rounded-[4px] text-[10px] font-[300]"
      : "inline-flex items-center px-[6px] py-[1px] rounded-[4px] text-[11px] font-[500]";

  const colors = context === "marketing" ? MARKETING_STYLES[variant] : APP_STYLES[variant];

  return (
    <span className={[base, colors, className].join(" ")}>
      {children}
    </span>
  );
}
