import React from "react";

type ButtonContext = "marketing" | "app";

type ButtonVariant =
  // Marketing (Stripe-inspired)
  | "m-primary"    // purple fill
  | "m-ghost"      // purple outline
  | "m-neutral"    // neutral outline
  // App (Cursor-inspired)
  | "a-primary"    // warm surface
  | "a-pill"       // pill tag
  | "a-ghost"      // transparent
  | "a-accent";    // orange fill (AI / CTAs)

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  context?: ButtonContext;
  size?: "sm" | "md" | "lg";
  children: React.ReactNode;
}

const VARIANT_STYLES: Record<ButtonVariant, string> = {
  "m-primary":
    "bg-[#533afd] text-white border-transparent hover:bg-[#4434d4] rounded-[4px] font-[400] text-[16px]",
  "m-ghost":
    "bg-transparent text-[#533afd] border border-[#b9b9f9] hover:bg-[rgba(83,58,253,0.05)] rounded-[4px] font-[400] text-[16px]",
  "m-neutral":
    "bg-transparent text-[rgba(16,16,16,0.3)] border border-[#d4dee9] hover:bg-[rgba(0,0,0,0.03)] rounded-[4px] font-[400] text-[16px]",
  "a-primary":
    "bg-[#ebeae5] text-[#26251e] border-transparent hover:text-[#cf2d56] rounded-[8px] font-[400] text-[14px]",
  "a-pill":
    "bg-[#e6e5e0] text-[rgba(38,37,30,0.6)] border-transparent hover:text-[#cf2d56] rounded-full font-[400] text-[14px]",
  "a-ghost":
    "bg-[rgba(38,37,30,0.06)] text-[rgba(38,37,30,0.55)] border-transparent hover:text-[#cf2d56] rounded-[8px] font-[400] text-[14px]",
  "a-accent":
    "bg-[#f54e00] text-white border-transparent hover:opacity-90 rounded-[8px] font-[400] text-[14px]",
};

const SIZE_STYLES: Record<"sm" | "md" | "lg", string> = {
  sm: "px-3 py-1.5",
  md: "px-4 py-2",
  lg: "px-5 py-2.5",
};

export function Button({
  variant = "m-primary",
  context,
  size = "md",
  className = "",
  children,
  ...props
}: ButtonProps) {
  // Infer context from variant prefix
  const resolvedVariant =
    context === "app" && !variant.startsWith("a-")
      ? ("a-primary" as ButtonVariant)
      : context === "marketing" && !variant.startsWith("m-")
        ? ("m-primary" as ButtonVariant)
        : variant;

  return (
    <button
      className={[
        "inline-flex items-center justify-center gap-2 cursor-pointer transition-all duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-1",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        VARIANT_STYLES[resolvedVariant],
        SIZE_STYLES[size],
        className,
      ].join(" ")}
      {...props}
    >
      {children}
    </button>
  );
}
