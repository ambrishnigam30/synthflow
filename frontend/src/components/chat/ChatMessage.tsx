"use client";

import type { ChatMessage as ChatMessageType } from "@/lib/stores/chatStore";

// ── Very minimal inline markdown ─────────────────────────────────────────────
// Handles: **bold**, `code`, line breaks. Full MD lib is overkill for chat.

function InlineMarkdown({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((part, i) => {
        if (part.startsWith("**") && part.endsWith("**")) {
          return <strong key={i}>{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith("`") && part.endsWith("`")) {
          return (
            <code
              key={i}
              style={{
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "0.9em",
                background: "rgba(38,37,30,0.08)",
                borderRadius: "3px",
                padding: "1px 4px",
              }}
            >
              {part.slice(1, -1)}
            </code>
          );
        }
        // Render newlines as <br />
        return part.split("\n").map((line, j, arr) => (
          <span key={`${i}-${j}`}>
            {line}
            {j < arr.length - 1 && <br />}
          </span>
        ));
      })}
    </>
  );
}

// ── Timestamp ────────────────────────────────────────────────────────────────

function Timestamp({ date }: { date: Date }) {
  const t = new Date(date);
  const label = t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  return (
    <span
      style={{
        fontFamily: "system-ui",
        fontSize: "11px",
        color: "rgba(38,37,30,0.35)",
        lineHeight: 1,
      }}
    >
      {label}
    </span>
  );
}

// ── User message ─────────────────────────────────────────────────────────────

function UserMessage({ msg }: { msg: ChatMessageType }) {
  return (
    <div className="flex flex-col items-end gap-1">
      <div
        className="max-w-[75%] px-4 py-3"
        style={{
          background: "#26251e",
          color: "#ffffff",
          borderRadius: "8px 8px 2px 8px",
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "16px",
          fontWeight: 400,
          lineHeight: 1.5,
        }}
      >
        <InlineMarkdown text={msg.content} />
      </div>
      <Timestamp date={msg.timestamp} />
    </div>
  );
}

// ── Assistant message ─────────────────────────────────────────────────────────

function AssistantMessage({ msg }: { msg: ChatMessageType }) {
  return (
    <div className="flex flex-col items-start gap-1">
      <div
        className="max-w-[80%] px-4 py-3"
        style={{
          background: "#ffffff",
          color: "#26251e",
          border: "1px solid rgba(38,37,30,0.1)",
          borderRadius: "8px 8px 8px 2px",
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "16px",
          fontWeight: 400,
          lineHeight: 1.5,
        }}
      >
        <InlineMarkdown text={msg.content} />
      </div>
      <Timestamp date={msg.timestamp} />
    </div>
  );
}

// ── System / progress message ─────────────────────────────────────────────────

function SystemMessage({
  msg,
  onRetry,
}: {
  msg: ChatMessageType;
  onRetry?: () => void;
}) {
  if (msg.isError) {
    return (
      <div className="flex justify-center">
        <div
          className="px-4 py-3"
          style={{
            background: "rgba(207,45,86,0.06)",
            border: "1px solid rgba(207,45,86,0.25)",
            borderRadius: "8px",
            fontFamily: "system-ui",
            fontSize: "13px",
            color: "#cf2d56",
            maxWidth: "520px",
            textAlign: "center",
          }}
        >
          <div style={{ marginBottom: onRetry ? "10px" : "0" }}>{msg.content}</div>
          {onRetry && (
            <button
              onClick={onRetry}
              style={{
                background: "#cf2d56",
                color: "#ffffff",
                border: "none",
                borderRadius: "6px",
                padding: "6px 14px",
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "12px",
                fontWeight: 400,
                cursor: "pointer",
              }}
              onMouseEnter={(e) => {
                (e.currentTarget as HTMLElement).style.opacity = "0.85";
              }}
              onMouseLeave={(e) => {
                (e.currentTarget as HTMLElement).style.opacity = "1";
              }}
            >
              Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-center">
      <div
        className="px-4 py-2"
        style={{
          background: "#ebeae5",
          border: "1px solid rgba(38,37,30,0.1)",
          borderRadius: "8px",
          fontFamily: "system-ui",
          fontSize: "12px",
          color: "rgba(38,37,30,0.55)",
          maxWidth: "480px",
          textAlign: "center",
        }}
      >
        {msg.content}
      </div>
    </div>
  );
}

// ── Streaming assistant bubble ────────────────────────────────────────────────

export function StreamingBubble({ content }: { content: string }) {
  return (
    <div className="flex flex-col items-start gap-1">
      <div
        className="max-w-[80%] px-4 py-3"
        style={{
          background: "#ffffff",
          color: "#26251e",
          border: "1px solid rgba(38,37,30,0.1)",
          borderRadius: "8px 8px 8px 2px",
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "16px",
          lineHeight: 1.5,
        }}
      >
        <InlineMarkdown text={content} />
        {/* Cursor blink */}
        <span
          style={{
            display: "inline-block",
            width: "2px",
            height: "1em",
            background: "#f54e00",
            marginLeft: "2px",
            verticalAlign: "text-bottom",
            animation: "blink 1s step-end infinite",
          }}
        />
        <style>{`@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }`}</style>
      </div>
    </div>
  );
}

// ── ChatMessage (router) ──────────────────────────────────────────────────────

export default function ChatMessage({
  msg,
  onRetry,
}: {
  msg: ChatMessageType;
  onRetry?: () => void;
}) {
  if (msg.role === "user") return <UserMessage msg={msg} />;
  if (msg.role === "system") return <SystemMessage msg={msg} onRetry={onRetry} />;
  return <AssistantMessage msg={msg} />;
}
