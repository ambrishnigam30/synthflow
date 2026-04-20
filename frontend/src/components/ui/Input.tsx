import React from "react";

type InputContext = "marketing" | "app";

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  context?: InputContext;
  label?: string;
  error?: string;
}

export function Input({
  context = "marketing",
  label,
  error,
  className = "",
  id,
  ...props
}: InputProps) {
  const inputId = id ?? (label ? label.toLowerCase().replace(/\s+/g, "-") : undefined);

  const inputClass =
    context === "marketing"
      ? "w-full border border-[#e5edf5] rounded-[4px] px-3 py-2 text-[#061b31] text-[16px] font-[300] bg-white placeholder:text-[#64748d] focus:outline-none focus:border-[#533afd] transition-colors"
      : "w-full border border-[rgba(38,37,30,0.1)] rounded-[8px] px-3 py-2 text-[#26251e] text-[14px] font-[400] bg-transparent placeholder:text-[rgba(38,37,30,0.4)] focus:outline-none focus:border-[rgba(38,37,30,0.2)] transition-colors";

  const labelClass =
    context === "marketing"
      ? "block text-[14px] font-[400] text-[#273951] mb-1.5"
      : "block text-[11px] font-[500] uppercase tracking-[0.05em] text-[rgba(38,37,30,0.55)] mb-1";

  return (
    <div className="w-full">
      {label && (
        <label htmlFor={inputId} className={labelClass}>
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={[inputClass, error ? (context === "marketing" ? "border-[#ea2261]" : "border-[#cf2d56]") : "", className].join(" ")}
        {...props}
      />
      {error && (
        <p className={`mt-1 text-[12px] ${context === "marketing" ? "text-[#ea2261]" : "text-[#cf2d56]"}`}>
          {error}
        </p>
      )}
    </div>
  );
}
