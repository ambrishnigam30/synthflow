"use client";

import { useState, useRef, useCallback } from "react";
import { Upload, X } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface UploadZoneProps {
  onUpload: (file: File) => Promise<void>;
  accept?: string[];
  maxSizeMb?: number;
}

type UploadState = "idle" | "dragging" | "uploading" | "error";

// ── Helpers ───────────────────────────────────────────────────────────────────

const DEFAULT_ACCEPT = [".csv", ".xlsx", ".parquet", ".json"];

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function UploadZone({
  onUpload,
  accept = DEFAULT_ACCEPT,
  maxSizeMb = 500,
}: UploadZoneProps) {
  const [state, setState] = useState<UploadState>("idle");
  const [progress, setProgress] = useState(0);
  const [errorMsg, setErrorMsg] = useState("");
  const [fileName, setFileName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const validate = useCallback(
    (file: File): string | null => {
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!accept.includes(ext)) {
        return `Unsupported format. Accepted: ${accept.join(", ")}`;
      }
      if (file.size > maxSizeMb * 1024 * 1024) {
        return `File too large. Max ${maxSizeMb} MB.`;
      }
      return null;
    },
    [accept, maxSizeMb]
  );

  async function handleFile(file: File) {
    const err = validate(file);
    if (err) {
      setState("error");
      setErrorMsg(err);
      return;
    }

    setFileName(file.name);
    setState("uploading");
    setProgress(0);

    // Simulate upload progress
    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= 90) {
          clearInterval(interval);
          return 90;
        }
        return p + 10;
      });
    }, 150);

    try {
      await onUpload(file);
      clearInterval(interval);
      setProgress(100);
      setTimeout(() => {
        setState("idle");
        setProgress(0);
        setFileName("");
      }, 800);
    } catch (e: unknown) {
      clearInterval(interval);
      setState("error");
      setErrorMsg(e instanceof Error ? e.message : "Upload failed");
    }
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    setState("idle");
    const file = e.dataTransfer.files[0];
    if (file) void handleFile(file);
  }

  function onDragOver(e: React.DragEvent) {
    e.preventDefault();
    setState("dragging");
  }

  function onDragLeave() {
    setState("idle");
  }

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) void handleFile(file);
    e.target.value = "";
  }

  const borderColor =
    state === "dragging"
      ? "#f54e00"
      : state === "error"
      ? "#cf2d56"
      : "rgba(38,37,30,0.2)";

  const bgColor =
    state === "dragging" ? "rgba(245,78,0,0.04)" : "#f7f7f4";

  return (
    <div
      className="rounded-[8px] p-8 flex flex-col items-center gap-4 transition-all duration-150"
      style={{
        border: `2px dashed ${borderColor}`,
        background: bgColor,
        cursor: state === "uploading" ? "not-allowed" : "pointer",
      }}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onClick={() => state !== "uploading" && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept.join(",")}
        onChange={onInputChange}
        style={{ display: "none" }}
      />

      {state === "uploading" ? (
        <>
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center"
            style={{ background: "rgba(245,78,0,0.1)" }}
          >
            <Upload size={18} strokeWidth={1.5} style={{ color: "#f54e00" }} />
          </div>
          <div className="w-full max-w-[280px]">
            <div className="flex justify-between mb-1.5">
              <span
                style={{
                  fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                  fontSize: "13px",
                  color: "#26251e",
                }}
              >
                {fileName}
              </span>
              <span
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "12px",
                  color: "#f54e00",
                }}
              >
                {progress}%
              </span>
            </div>
            <div
              className="rounded-full overflow-hidden"
              style={{ height: "4px", background: "rgba(38,37,30,0.1)" }}
            >
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{ width: `${progress}%`, background: "#f54e00" }}
              />
            </div>
          </div>
        </>
      ) : state === "error" ? (
        <>
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center"
            style={{ background: "rgba(207,45,86,0.1)" }}
          >
            <X size={18} strokeWidth={1.5} style={{ color: "#cf2d56" }} />
          </div>
          <div className="text-center">
            <p
              style={{
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "14px",
                color: "#cf2d56",
                marginBottom: "4px",
              }}
            >
              {errorMsg}
            </p>
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "12px",
                color: "rgba(38,37,30,0.45)",
              }}
            >
              Click to try again
            </p>
          </div>
        </>
      ) : (
        <>
          <div
            className="w-10 h-10 rounded-full flex items-center justify-center"
            style={{ background: "rgba(38,37,30,0.06)" }}
          >
            <Upload
              size={18}
              strokeWidth={1.5}
              style={{ color: "rgba(38,37,30,0.45)" }}
            />
          </div>
          <div className="text-center">
            <p
              style={{
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "15px",
                fontWeight: 400,
                color: "#26251e",
                marginBottom: "4px",
              }}
            >
              {state === "dragging" ? "Drop to upload" : "Drop a file or click to browse"}
            </p>
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "12px",
                color: "rgba(38,37,30,0.45)",
              }}
            >
              {accept.join(", ")} · up to {maxSizeMb} MB
            </p>
          </div>
        </>
      )}
    </div>
  );
}

// ── Compact variant ───────────────────────────────────────────────────────────

export function UploadButton({
  onUpload,
  accept = DEFAULT_ACCEPT,
}: {
  onUpload: (file: File) => Promise<void>;
  accept?: string[];
}) {
  const inputRef = useRef<HTMLInputElement>(null);

  function onInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) void onUpload(file);
    e.target.value = "";
  }

  return (
    <>
      <input
        ref={inputRef}
        type="file"
        accept={accept.join(",")}
        onChange={onInputChange}
        style={{ display: "none" }}
      />
      <button
        onClick={() => inputRef.current?.click()}
        className="flex items-center gap-1.5 transition-opacity hover:opacity-80"
        style={{
          background: "#f54e00",
          color: "#ffffff",
          border: "none",
          borderRadius: "8px",
          padding: "8px 14px",
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "13px",
          fontWeight: 400,
          cursor: "pointer",
        }}
      >
        <Upload size={14} strokeWidth={1.5} />
        Upload
      </button>
    </>
  );
}

// ── Re-export size formatter ──────────────────────────────────────────────────

export { formatBytes };
