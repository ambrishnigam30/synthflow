"use client";

import { useState, useMemo } from "react";
import { Search, RefreshCw, X, ChevronDown, Zap, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";

// ── Types ─────────────────────────────────────────────────────────────────────

interface HistoryRow {
  id: string;
  prompt: string;
  domain: string;
  rows: number;
  cols: number;
  qualityScore: number;
  status: "done" | "error" | "generating";
  date: string;
  conversationId: string;
}

// ── Mock data ─────────────────────────────────────────────────────────────────

const MOCK_HISTORY: HistoryRow[] = [
  {
    id: "g1",
    prompt: "Generate 10,000 patient records for a cardiology department in Mumbai",
    domain: "Healthcare",
    rows: 10000,
    cols: 18,
    qualityScore: 94.2,
    status: "done",
    date: "2026-04-24",
    conversationId: "c1",
  },
  {
    id: "g2",
    prompt: "Banking transaction dataset with 5% fraud rate, HDFC",
    domain: "Banking",
    rows: 50000,
    cols: 22,
    qualityScore: 91.5,
    status: "done",
    date: "2026-04-23",
    conversationId: "c2",
  },
  {
    id: "g3",
    prompt: "E-commerce order history with seasonal patterns, India",
    domain: "Retail",
    rows: 25000,
    cols: 15,
    qualityScore: 88.1,
    status: "done",
    date: "2026-04-22",
    conversationId: "c3",
  },
  {
    id: "g4",
    prompt: "Crop yield data Punjab wheat season with weather correlation",
    domain: "Agriculture",
    rows: 8000,
    cols: 12,
    qualityScore: 96.0,
    status: "done",
    date: "2026-04-21",
    conversationId: "c4",
  },
  {
    id: "g5",
    prompt: "Smart factory sensor readings with anomaly injection 100K rows",
    domain: "IoT",
    rows: 100000,
    cols: 28,
    qualityScore: 89.3,
    status: "done",
    date: "2026-04-20",
    conversationId: "c5",
  },
  {
    id: "g6",
    prompt: "Student performance dataset across 12 subjects Maharashtra Board",
    domain: "Education",
    rows: 5000,
    cols: 16,
    qualityScore: 92.7,
    status: "done",
    date: "2026-04-19",
    conversationId: "c6",
  },
  {
    id: "g7",
    prompt: "HR attrition dataset with salary bands and performance ratings",
    domain: "HR",
    rows: 12000,
    cols: 20,
    qualityScore: 0,
    status: "error",
    date: "2026-04-18",
    conversationId: "c7",
  },
];

const ALL_DOMAINS = [...new Set(MOCK_HISTORY.map((r) => r.domain))].sort();

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatRows(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

function qualityColor(score: number): string {
  if (score > 80) return "#1f8a65";
  if (score >= 60) return "#c08532";
  return "#cf2d56";
}

function statusLabel(s: HistoryRow["status"]) {
  const map: Record<HistoryRow["status"], { label: string; color: string; bg: string }> = {
    done: { label: "Done", color: "#108c3d", bg: "rgba(21,190,83,0.12)" },
    error: { label: "Error", color: "#cf2d56", bg: "rgba(207,45,86,0.1)" },
    generating: { label: "Running", color: "#f54e00", bg: "rgba(245,78,0,0.1)" },
  };
  return map[s];
}

// ── Detail panel ──────────────────────────────────────────────────────────────

function DetailPanel({
  row,
  onClose,
  onRegenerate,
  onOpenInChat,
}: {
  row: HistoryRow;
  onClose: () => void;
  onRegenerate: (id: string) => void;
  onOpenInChat: (conversationId: string) => void;
}) {
  const st = statusLabel(row.status);

  return (
    <div
      className="flex flex-col h-full"
      style={{
        width: "340px",
        background: "#ffffff",
        borderLeft: "1px solid rgba(38,37,30,0.1)",
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-4"
        style={{ borderBottom: "1px solid rgba(38,37,30,0.08)" }}
      >
        <span
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            fontWeight: 600,
            textTransform: "uppercase",
            letterSpacing: "0.048px",
            color: "rgba(38,37,30,0.45)",
          }}
        >
          Generation Detail
        </span>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            color: "rgba(38,37,30,0.4)",
          }}
        >
          <X size={16} strokeWidth={1.5} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-4 space-y-5">
        {/* Quality */}
        {row.status === "done" && (
          <div className="flex items-center gap-3">
            <span
              style={{
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "36px",
                fontWeight: 400,
                color: qualityColor(row.qualityScore),
                letterSpacing: "-0.72px",
                lineHeight: 1,
              }}
            >
              {row.qualityScore.toFixed(1)}
            </span>
            <div>
              <p
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "10px",
                  fontWeight: 500,
                  textTransform: "uppercase",
                  letterSpacing: "0.5px",
                  color: "rgba(38,37,30,0.45)",
                }}
              >
                Quality Score
              </p>
            </div>
          </div>
        )}

        {/* Status badge */}
        <div>
          <span
            className="inline-block px-2 py-1 rounded-[4px]"
            style={{
              background: st.bg,
              color: st.color,
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 500,
            }}
          >
            {st.label}
          </span>
        </div>

        {/* Prompt */}
        <div>
          <p
            className="mb-1"
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.048px",
              color: "rgba(38,37,30,0.4)",
            }}
          >
            Prompt
          </p>
          <p
            style={{
              fontFamily: "var(--font-serif, Georgia, serif)",
              fontSize: "15px",
              color: "#26251e",
              lineHeight: 1.5,
            }}
          >
            {row.prompt}
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-3">
          {[
            { label: "Rows", value: formatRows(row.rows) },
            { label: "Columns", value: String(row.cols) },
            { label: "Domain", value: row.domain },
            { label: "Date", value: row.date },
          ].map(({ label, value }) => (
            <div
              key={label}
              className="rounded-[6px] p-3"
              style={{ background: "#f7f7f4", border: "1px solid rgba(38,37,30,0.06)" }}
            >
              <p
                style={{
                  fontFamily: "system-ui",
                  fontSize: "10px",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.048px",
                  color: "rgba(38,37,30,0.4)",
                  marginBottom: "2px",
                }}
              >
                {label}
              </p>
              <p
                style={{
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "13px",
                  color: "#26251e",
                }}
              >
                {value}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Actions */}
      <div
        className="px-5 py-4 flex flex-col gap-2"
        style={{ borderTop: "1px solid rgba(38,37,30,0.08)" }}
      >
        <button
          onClick={() => onOpenInChat(row.conversationId)}
          className="w-full flex items-center justify-center gap-1.5 transition-opacity hover:opacity-90"
          style={{
            background: "#f54e00",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            padding: "10px 14px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            fontWeight: 400,
            cursor: "pointer",
          }}
        >
          <Zap size={14} strokeWidth={1.5} />
          Open in Chat
        </button>
        <button
          onClick={() => onRegenerate(row.id)}
          className="w-full flex items-center justify-center gap-1.5 transition-all"
          style={{
            background: "#ebeae5",
            color: "#26251e",
            border: "none",
            borderRadius: "8px",
            padding: "10px 14px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            fontWeight: 400,
            cursor: "pointer",
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.color = "#cf2d56";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = "#26251e";
          }}
        >
          <RefreshCw size={14} strokeWidth={1.5} />
          Regenerate
        </button>
      </div>
    </div>
  );
}

// ── Filter pill ───────────────────────────────────────────────────────────────

function FilterPill({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        background: active ? "#26251e" : "#ebeae5",
        color: active ? "#ffffff" : "rgba(38,37,30,0.65)",
        border: "none",
        borderRadius: "9999px",
        padding: "4px 12px",
        fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
        fontSize: "12px",
        fontWeight: 400,
        cursor: "pointer",
        transition: "all 150ms ease",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </button>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function HistoryPage() {
  const router = useRouter();
  const [rows, setRows] = useState<HistoryRow[]>(MOCK_HISTORY);
  const [query, setQuery] = useState("");
  const [domainFilter, setDomainFilter] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<HistoryRow["status"] | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  const filtered = useMemo(() => {
    let list = rows;
    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter(
        (r) =>
          r.prompt.toLowerCase().includes(q) ||
          r.domain.toLowerCase().includes(q)
      );
    }
    if (domainFilter) list = list.filter((r) => r.domain === domainFilter);
    if (statusFilter) list = list.filter((r) => r.status === statusFilter);
    return [...list].sort((a, b) =>
      sortDir === "desc"
        ? b.date.localeCompare(a.date)
        : a.date.localeCompare(b.date)
    );
  }, [rows, query, domainFilter, statusFilter, sortDir]);

  const detailRow = rows.find((r) => r.id === selectedId) ?? null;

  function toggleSelect(id: string, e: React.MouseEvent) {
    e.stopPropagation();
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function bulkDelete() {
    setRows((prev) => prev.filter((r) => !selected.has(r.id)));
    if (selectedId && selected.has(selectedId)) setSelectedId(null);
    setSelected(new Set());
  }

  function handleRegenerate(id: string) {
    const row = rows.find((r) => r.id === id);
    if (row) {
      router.push(`/app/generate?q=${encodeURIComponent(row.prompt)}`);
    }
  }

  function handleOpenInChat(conversationId: string) {
    router.push(`/app/generate?conv=${conversationId}`);
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Main panel */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {/* Header */}
        <div
          className="px-6 py-5 flex-shrink-0"
          style={{ borderBottom: "1px solid rgba(38,37,30,0.08)" }}
        >
          <h1
            className="mb-4"
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "22px",
              fontWeight: 400,
              color: "#26251e",
              letterSpacing: "-0.22px",
            }}
          >
            History
          </h1>

          {/* Search + filters */}
          <div className="flex flex-wrap items-center gap-3">
            <div
              className="flex items-center gap-2 rounded-[8px] px-3 py-2"
              style={{
                background: "#ffffff",
                border: "1px solid rgba(38,37,30,0.1)",
                flex: "1 1 200px",
                maxWidth: "320px",
              }}
            >
              <Search size={14} strokeWidth={1.5} style={{ color: "rgba(38,37,30,0.35)", flexShrink: 0 }} />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search history…"
                style={{
                  flex: 1,
                  background: "transparent",
                  border: "none",
                  outline: "none",
                  fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                  fontSize: "13px",
                  color: "#26251e",
                }}
              />
            </div>

            {/* Domain filters */}
            <div className="flex items-center gap-1.5 flex-wrap">
              <FilterPill
                label="All"
                active={domainFilter === null}
                onClick={() => setDomainFilter(null)}
              />
              {ALL_DOMAINS.map((d) => (
                <FilterPill
                  key={d}
                  label={d}
                  active={domainFilter === d}
                  onClick={() => setDomainFilter(d === domainFilter ? null : d)}
                />
              ))}
            </div>

            {/* Sort toggle */}
            <button
              onClick={() => setSortDir((d) => (d === "desc" ? "asc" : "desc"))}
              className="flex items-center gap-1 transition-opacity hover:opacity-70"
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                fontFamily: "system-ui",
                fontSize: "12px",
                color: "rgba(38,37,30,0.5)",
              }}
            >
              Date <ChevronDown size={12} strokeWidth={2} style={{ transform: sortDir === "asc" ? "rotate(180deg)" : "none" }} />
            </button>

            {/* Bulk delete */}
            {selected.size > 0 && (
              <button
                onClick={bulkDelete}
                className="flex items-center gap-1.5 transition-opacity hover:opacity-80"
                style={{
                  background: "rgba(207,45,86,0.1)",
                  color: "#cf2d56",
                  border: "none",
                  borderRadius: "6px",
                  padding: "6px 12px",
                  fontFamily: "system-ui",
                  fontSize: "12px",
                  fontWeight: 500,
                  cursor: "pointer",
                }}
              >
                <Trash2 size={12} strokeWidth={1.5} />
                Delete {selected.size}
              </button>
            )}
          </div>
        </div>

        {/* Table */}
        <div className="flex-1 overflow-auto">
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead style={{ position: "sticky", top: 0, zIndex: 1 }}>
              <tr
                style={{
                  background: "#f7f7f4",
                  borderBottom: "1px solid rgba(38,37,30,0.1)",
                }}
              >
                <th style={{ width: "36px", padding: "10px 14px" }}>
                  <input
                    type="checkbox"
                    checked={selected.size === filtered.length && filtered.length > 0}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelected(new Set(filtered.map((r) => r.id)));
                      } else {
                        setSelected(new Set());
                      }
                    }}
                  />
                </th>
                {["Date", "Prompt", "Domain", "Rows", "Quality", "Status"].map((h) => (
                  <th
                    key={h}
                    style={{
                      fontFamily: "system-ui",
                      fontSize: "11px",
                      fontWeight: 600,
                      color: "rgba(38,37,30,0.55)",
                      textTransform: "uppercase",
                      letterSpacing: "0.048px",
                      padding: "10px 14px",
                      textAlign: "left",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filtered.map((row, i) => {
                const st = statusLabel(row.status);
                const isSelected = selectedId === row.id;
                return (
                  <tr
                    key={row.id}
                    onClick={() => setSelectedId(isSelected ? null : row.id)}
                    className="cursor-pointer transition-colors"
                    style={{
                      borderBottom:
                        i < filtered.length - 1
                          ? "1px solid rgba(38,37,30,0.06)"
                          : "none",
                      background: isSelected ? "#f2f1ed" : "transparent",
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected) (e.currentTarget as HTMLElement).style.background = "#fafaf8";
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected) (e.currentTarget as HTMLElement).style.background = "transparent";
                    }}
                  >
                    <td style={{ padding: "10px 14px" }}>
                      <input
                        type="checkbox"
                        checked={selected.has(row.id)}
                        onClick={(e) => toggleSelect(row.id, e)}
                        onChange={() => undefined}
                      />
                    </td>
                    <td
                      style={{
                        fontFamily: "var(--font-mono, monospace)",
                        fontSize: "12px",
                        color: "rgba(38,37,30,0.5)",
                        padding: "10px 14px",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {row.date}
                    </td>
                    <td
                      style={{
                        fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                        fontSize: "13px",
                        color: "#26251e",
                        padding: "10px 14px",
                        maxWidth: "320px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {row.prompt}
                    </td>
                    <td
                      style={{
                        fontFamily: "system-ui",
                        fontSize: "12px",
                        color: "rgba(38,37,30,0.6)",
                        padding: "10px 14px",
                        whiteSpace: "nowrap",
                      }}
                    >
                      {row.domain}
                    </td>
                    <td
                      style={{
                        fontFamily: "var(--font-mono, monospace)",
                        fontSize: "12px",
                        color: "rgba(38,37,30,0.6)",
                        padding: "10px 14px",
                        whiteSpace: "nowrap",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {formatRows(row.rows)}
                    </td>
                    <td
                      style={{
                        fontFamily: "var(--font-mono, monospace)",
                        fontSize: "13px",
                        color: row.status === "done" ? qualityColor(row.qualityScore) : "rgba(38,37,30,0.3)",
                        padding: "10px 14px",
                        whiteSpace: "nowrap",
                        fontVariantNumeric: "tabular-nums",
                      }}
                    >
                      {row.status === "done" ? row.qualityScore.toFixed(1) : "—"}
                    </td>
                    <td style={{ padding: "10px 14px" }}>
                      <span
                        className="inline-block px-2 py-0.5 rounded-[4px]"
                        style={{
                          background: st.bg,
                          color: st.color,
                          fontFamily: "system-ui",
                          fontSize: "11px",
                          fontWeight: 500,
                          whiteSpace: "nowrap",
                        }}
                      >
                        {st.label}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {filtered.length === 0 && (
            <div
              className="flex items-center justify-center py-16"
              style={{
                fontFamily: "system-ui",
                fontSize: "13px",
                color: "rgba(38,37,30,0.35)",
              }}
            >
              No results found
            </div>
          )}
        </div>
      </div>

      {/* Detail panel (slide-out) */}
      {detailRow && (
        <DetailPanel
          row={detailRow}
          onClose={() => setSelectedId(null)}
          onRegenerate={handleRegenerate}
          onOpenInChat={handleOpenInChat}
        />
      )}
    </div>
  );
}
