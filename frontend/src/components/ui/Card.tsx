import React from "react";

type CardVariant = "marketing" | "app" | "app-elevated" | "dark";

interface CardProps {
  variant?: CardVariant;
  className?: string;
  children: React.ReactNode;
  onClick?: () => void;
  hover?: boolean;
}

const VARIANT_STYLES: Record<CardVariant, string> = {
  marketing:
    "bg-white border border-[#e5edf5] rounded-[6px]",
  app:
    "bg-[#e6e5e0] border border-[rgba(38,37,30,0.1)] rounded-[8px]",
  "app-elevated":
    "bg-white border border-[rgba(38,37,30,0.1)] rounded-[8px]",
  dark:
    "bg-[#1c1e54] border border-[rgba(255,255,255,0.1)] rounded-[6px]",
};

const HOVER_STYLES: Record<CardVariant, string> = {
  marketing: "hover:shadow-[rgba(50,50,93,0.35)_0px_30px_45px_-30px,rgba(0,0,0,0.15)_0px_18px_36px_-18px] cursor-pointer",
  app: "hover:border-[rgba(38,37,30,0.2)] cursor-pointer transition-all duration-150",
  "app-elevated": "hover:shadow-[rgba(0,0,0,0.18)_0px_28px_70px,rgba(0,0,0,0.12)_0px_14px_32px] cursor-pointer",
  dark: "hover:border-[rgba(255,255,255,0.2)] cursor-pointer",
};

export function Card({
  variant = "marketing",
  className = "",
  hover = false,
  onClick,
  children,
}: CardProps) {
  return (
    <div
      className={[
        VARIANT_STYLES[variant],
        hover ? HOVER_STYLES[variant] : "",
        className,
      ].join(" ")}
      style={
        variant === "marketing" && !hover
          ? {
              boxShadow:
                "rgba(50,50,93,0.25) 0px 30px 45px -30px, rgba(0,0,0,0.1) 0px 18px 36px -18px",
            }
          : variant === "app-elevated"
            ? {
                boxShadow:
                  "rgba(0,0,0,0.14) 0px 28px 70px, rgba(0,0,0,0.1) 0px 14px 32px",
              }
            : undefined
      }
      onClick={onClick}
    >
      {children}
    </div>
  );
}
