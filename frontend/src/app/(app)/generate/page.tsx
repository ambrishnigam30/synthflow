"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Plus, ChevronDown } from "lucide-react";
import ChatMessage, { StreamingBubble } from "@/components/chat/ChatMessage";
import ChatInput, { type ChatInputOptions } from "@/components/chat/ChatInput";
import PhaseProgress from "@/components/chat/PhaseProgress";
import GenerationCard from "@/components/chat/GenerationCard";
import { useChatStore, type ChatMessage as MsgType } from "@/lib/stores/chatStore";
import {
  useGenerationStore,
  type GenerationResult,
} from "@/lib/stores/generationStore";
import { SynthFlowWS } from "@/lib/ws";
import type { WsGenerationDone } from "@/lib/ws";

// ── Example cards ─────────────────────────────────────────────────────────────

const EXAMPLES = [
  {
    domain: "Healthcare",
    icon: "🏥",
    prompt: "Generate 10,000 patient records for a cardiology department in Mumbai",
  },
  {
    domain: "Banking",
    icon: "🏦",
    prompt: "Create a banking transaction dataset with 5% fraud rate for HDFC",
  },
  {
    domain: "Retail",
    icon: "🛒",
    prompt: "E-commerce order history with seasonal patterns, India, 50K rows",
  },
  {
    domain: "Agriculture",
    icon: "🌾",
    prompt: "Crop yield data for Punjab wheat season with weather correlation",
  },
  {
    domain: "IoT",
    icon: "📡",
    prompt: "Smart factory sensor readings with anomaly injection, 100K rows",
  },
  {
    domain: "Education",
    icon: "🎓",
    prompt: "Student performance dataset across 12 subjects, Maharashtra Board",
  },
];

// ── Helpers ───────────────────────────────────────────────────────────────────

function newMsgId(): string {
  return crypto.randomUUID();
}

function formatDate(d: Date): string {
  return d.toLocaleDateString("en-IN", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

// ── Empty state ───────────────────────────────────────────────────────────────

function EmptyState({ onExample }: { onExample: (prompt: string) => void }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-6 py-12">
      {/* Logo mark */}
      <div
        className="w-10 h-10 rounded-[8px] flex items-center justify-center mb-6"
        style={{ backgroundColor: "#3d4043" }}
      >
        <span className="text-white text-[18px] font-[500]">S</span>
      </div>

      <h1
        className="text-center mb-2"
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "32px",
          fontWeight: 400,
          color: "#26251e",
          letterSpacing: "-0.96px",
          lineHeight: 1.15,
        }}
      >
        What would you like to generate?
      </h1>
      <p
        className="text-center mb-10"
        style={{
          fontFamily: "var(--font-serif, Georgia, serif)",
          fontSize: "17px",
          color: "rgba(38,37,30,0.5)",
          lineHeight: 1.5,
          maxWidth: "420px",
        }}
      >
        Describe your dataset in plain English. SynthFlow handles the rest.
      </p>

      {/* Example grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 w-full max-w-[680px]">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.domain}
            onClick={() => onExample(ex.prompt)}
            className="text-left rounded-[8px] p-4 transition-all duration-150"
            style={{
              background: "#e6e5e0",
              border: "1px solid rgba(38,37,30,0.08)",
              cursor: "pointer",
            }}
            onMouseEnter={(e) => {
              const el = e.currentTarget as HTMLElement;
              el.style.background = "#dddcd7";
              el.style.borderColor = "rgba(38,37,30,0.15)";
            }}
            onMouseLeave={(e) => {
              const el = e.currentTarget as HTMLElement;
              el.style.background = "#e6e5e0";
              el.style.borderColor = "rgba(38,37,30,0.08)";
            }}
          >
            <div className="text-xl mb-2">{ex.icon}</div>
            <p
              style={{
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "13px",
                fontWeight: 400,
                color: "#26251e",
                marginBottom: "4px",
              }}
            >
              {ex.domain}
            </p>
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                color: "rgba(38,37,30,0.5)",
                lineHeight: 1.4,
              }}
            >
              {ex.prompt.slice(0, 65)}…
            </p>
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Conversation sidebar item ─────────────────────────────────────────────────

interface ConvItemProps {
  id: string;
  title: string;
  updatedAt: Date;
  active: boolean;
  onClick: () => void;
}

function ConvItem({ title, updatedAt, active, onClick }: ConvItemProps) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left px-3 py-2.5 rounded-[6px] transition-colors"
      style={{
        background: active ? "#f2f1ed" : "transparent",
        border: active ? "1px solid rgba(38,37,30,0.1)" : "1px solid transparent",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        if (!active) (e.currentTarget as HTMLElement).style.background = "#f7f7f4";
      }}
      onMouseLeave={(e) => {
        if (!active) (e.currentTarget as HTMLElement).style.background = "transparent";
      }}
    >
      <p
        className="truncate"
        style={{
          fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
          fontSize: "13px",
          fontWeight: 400,
          color: active ? "#26251e" : "rgba(38,37,30,0.65)",
        }}
      >
        {title}
      </p>
      <p
        style={{
          fontFamily: "system-ui",
          fontSize: "11px",
          color: "rgba(38,37,30,0.35)",
          marginTop: "1px",
        }}
      >
        {formatDate(updatedAt)}
      </p>
    </button>
  );
}

// ── Conversation sidebar ──────────────────────────────────────────────────────

interface ConvSidebarProps {
  onNewChat: () => void;
}

function ConvSidebar({ onNewChat }: ConvSidebarProps) {
  const { conversations, activeConversationId, setActiveConversation } =
    useChatStore();

  return (
    <aside
      className="hidden lg:flex flex-col flex-shrink-0"
      style={{
        width: "240px",
        background: "#f7f7f4",
        borderRight: "1px solid rgba(38,37,30,0.08)",
      }}
    >
      {/* New chat */}
      <div className="p-3">
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-1.5 transition-opacity hover:opacity-90"
          style={{
            background: "#f54e00",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            padding: "9px 12px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            fontWeight: 400,
            cursor: "pointer",
          }}
        >
          <Plus size={14} strokeWidth={2} />
          New Chat
        </button>
      </div>

      {/* Conversation list */}
      <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
        {conversations.length === 0 && (
          <p
            className="px-3 py-4 text-center"
            style={{
              fontFamily: "system-ui",
              fontSize: "12px",
              color: "rgba(38,37,30,0.35)",
            }}
          >
            No conversations yet
          </p>
        )}
        {conversations.map((c) => (
          <ConvItem
            key={c.id}
            id={c.id}
            title={c.title}
            updatedAt={c.updatedAt}
            active={c.id === activeConversationId}
            onClick={() => setActiveConversation(c.id)}
          />
        ))}
      </div>

      {/* Provider selector */}
      <div
        className="p-3"
        style={{ borderTop: "1px solid rgba(38,37,30,0.08)" }}
      >
        <div
          className="flex items-center justify-between px-3 py-2 rounded-[6px] cursor-pointer"
          style={{
            background: "#ebeae5",
            border: "1px solid rgba(38,37,30,0.08)",
          }}
        >
          <span
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 500,
              color: "rgba(38,37,30,0.55)",
            }}
          >
            SynthFlow Engine v2
          </span>
          <ChevronDown size={12} strokeWidth={2} style={{ color: "rgba(38,37,30,0.4)" }} />
        </div>
      </div>
    </aside>
  );
}

// ── Chat thread ───────────────────────────────────────────────────────────────

interface ChatThreadProps {
  messages: MsgType[];
  isStreaming: boolean;
  streamingContent: string;
  activeGenerationId: string | null;
}

function ChatThread({
  messages,
  isStreaming,
  streamingContent,
  activeGenerationId,
}: ChatThreadProps) {
  const { generations } = useGenerationStore();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  if (messages.length === 0) return null;

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
      {messages.map((msg) => {
        // Inject phase progress / generation card for messages that have a generationId
        const genId = msg.generationId;
        const gen = genId ? generations.find((g) => g.id === genId) : undefined;

        return (
          <div key={msg.id}>
            <ChatMessage msg={msg} />
            {gen && gen.status === "generating" && (
              <div className="mt-3 ml-0">
                <PhaseProgress phases={gen.phases} progress={gen.progress} />
              </div>
            )}
            {gen && gen.status === "done" && gen.result && (
              <div className="mt-3">
                <GenerationCard result={gen.result} />
              </div>
            )}
          </div>
        );
      })}

      {/* Active generation not yet linked to a message */}
      {activeGenerationId &&
        !messages.some((m) => m.generationId === activeGenerationId) && (
          <div>
            {(() => {
              const gen = generations.find((g) => g.id === activeGenerationId);
              if (!gen || gen.status !== "generating") return null;
              return <PhaseProgress phases={gen.phases} progress={gen.progress} />;
            })()}
          </div>
        )}

      {isStreaming && streamingContent && (
        <StreamingBubble content={streamingContent} />
      )}

      <div ref={bottomRef} />
    </div>
  );
}

// ── Inner page (needs useSearchParams) ───────────────────────────────────────

function GenerateInner() {
  const searchParams = useSearchParams();
  const initialPrompt = searchParams.get("q") ?? "";

  const {
    conversations,
    activeConversationId,
    messages,
    isStreaming,
    streamingContent,
    createConversation,
    setActiveConversation,
    addMessage,
    setStreaming,
    appendStreamChunk,
    clearStreamingContent,
  } = useChatStore();

  const {
    generations,
    activeGenerationId,
    startGeneration,
    setPhase,
    setProgress,
    setResult,
    setError,
  } = useGenerationStore();

  const wsRef = useRef<SynthFlowWS | null>(null);
  const pendingGenIdRef = useRef<string | null>(null);

  // On mount: connect WS for active conversation or pre-fill prompt
  useEffect(() => {
    if (initialPrompt && conversations.length === 0) {
      handleSend(initialPrompt, {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function ensureConversation(): string {
    if (activeConversationId) return activeConversationId;
    const conv = createConversation("New conversation");
    setActiveConversation(conv.id);
    return conv.id;
  }

  function connectWs(convId: string) {
    wsRef.current?.disconnect();

    wsRef.current = new SynthFlowWS(convId, {
      onTextChunk: (chunk) => {
        setStreaming(true);
        appendStreamChunk(chunk);
      },
      onGenerationStart: (genId) => {
        startGeneration(genId, convId, "");
        pendingGenIdRef.current = genId;
        // Add system message with linked genId
        addMessage({
          id: newMsgId(),
          role: "system",
          content: "Starting data generation…",
          timestamp: new Date(),
          generationId: genId,
        });
      },
      onPhaseUpdate: (phase, progress) => {
        if (pendingGenIdRef.current) {
          setPhase(pendingGenIdRef.current, phase);
          setProgress(pendingGenIdRef.current, progress);
        }
      },
      onGenerationDone: (payload: WsGenerationDone) => {
        const genId = payload.generation_id;
        const result: GenerationResult = {
          generationId: genId,
          domain: payload.domain,
          rowCount: payload.row_count,
          colCount: payload.col_count,
          qualityScore: payload.quality_score,
          previewRows: payload.preview_rows,
          schema: payload.schema,
          downloadUrls: payload.download_urls,
          createdAt: new Date(),
        };
        setResult(genId, result);
        pendingGenIdRef.current = null;
      },
      onError: (message) => {
        if (pendingGenIdRef.current) {
          setError(pendingGenIdRef.current, message);
          pendingGenIdRef.current = null;
        }
        addMessage({
          id: newMsgId(),
          role: "system",
          content: `Error: ${message}`,
          timestamp: new Date(),
        });
        setStreaming(false);
        clearStreamingContent();
      },
      onClose: () => {
        // Flush streaming content to a message
        const content = useChatStore.getState().streamingContent;
        if (content.trim()) {
          useChatStore.getState().addMessage({
            id: newMsgId(),
            role: "assistant",
            content,
            timestamp: new Date(),
          });
        }
        setStreaming(false);
        clearStreamingContent();
      },
    });

    wsRef.current.connect();
  }

  function handleNewChat() {
    wsRef.current?.disconnect();
    wsRef.current = null;
    const conv = createConversation("New conversation");
    setActiveConversation(conv.id);
  }

  function handleSend(content: string, opts: ChatInputOptions) {
    const convId = ensureConversation();

    // Add user message
    addMessage({
      id: newMsgId(),
      role: "user",
      content,
      timestamp: new Date(),
    });

    // Ensure WS connected
    if (!wsRef.current?.connected) {
      connectWs(convId);
    }

    // Small delay to let WS open
    setTimeout(() => {
      wsRef.current?.send({
        type: "message",
        content,
        conversation_id: convId,
        options: {
          row_count: opts.rowCount,
          seed: opts.seed,
          output_format: opts.outputFormat,
          scenario: opts.scenario,
        },
      });
    }, 100);
  }

  const hasMessages = messages.length > 0;
  const _ = conversations; // suppress unused warning
  const __ = activeGenerationId;
  void _;
  void __;

  return (
    <div className="flex h-full">
      {/* Conversation sidebar */}
      <ConvSidebar onNewChat={handleNewChat} />

      {/* Main chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {hasMessages ? (
          <ChatThread
            messages={messages}
            isStreaming={isStreaming}
            streamingContent={streamingContent}
            activeGenerationId={activeGenerationId}
          />
        ) : (
          <EmptyState onExample={(p) => handleSend(p, {})} />
        )}

        {/* Input */}
        <div
          className="flex-shrink-0 px-6 py-4"
          style={{ borderTop: "1px solid rgba(38,37,30,0.08)", background: "#f2f1ed" }}
        >
          <ChatInput
            onSend={handleSend}
            disabled={isStreaming}
            placeholder={
              hasMessages
                ? "Ask a follow-up or request changes…"
                : "Describe the dataset you need…"
            }
          />
        </div>
      </div>
    </div>
  );
}

// ── Page export ───────────────────────────────────────────────────────────────

export default function GeneratePage() {
  return (
    <Suspense>
      <GenerateInner />
    </Suspense>
  );
}
