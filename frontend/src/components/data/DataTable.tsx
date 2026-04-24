"use client";

import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight, Search } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

type SortDir = "asc" | "desc";

interface SortState {
  col: string;
  dir: SortDir;
}

interface DataTableProps {
  rows: Record<string, unknown>[];
  /** If omitted, all keys from first row are used */
  columns?: string[];
  pageSize?: number;
  searchable?: boolean;
  /** Max rows shown before pagination (0 = show all) */
  maxRows?: number;
  onRowClick?: (row: Record<string, unknown>, index: number) => void;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function isNumeric(val: unknown): boolean {
  return typeof val === "number";
}

function cellStr(val: unknown): string {
  if (val === null || val === undefined) return "";
  return String(val);
}

function compare(a: unknown, b: unknown, dir: SortDir): number {
  const mult = dir === "asc" ? 1 : -1;
  if (typeof a === "number" && typeof b === "number") return (a - b) * mult;
  return String(a ?? "").localeCompare(String(b ?? "")) * mult;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function DataTable({
  rows,
  columns,
  pageSize = 25,
  searchable = true,
  maxRows = 0,
  onRowClick,
}: DataTableProps) {
  const [sort, setSort] = useState<SortState | null>(null);
  const [page, setPage] = useState(0);
  const [query, setQuery] = useState("");

  const cols = useMemo(
    () => columns ?? (rows.length > 0 ? Object.keys(rows[0]) : []),
    [columns, rows]
  );

  const filtered = useMemo(() => {
    if (!query.trim()) return rows;
    const q = query.toLowerCase();
    return rows.filter((row) =>
      cols.some((col) => cellStr(row[col]).toLowerCase().includes(q))
    );
  }, [rows, cols, query]);

  const sorted = useMemo(() => {
    if (!sort) return filtered;
    return [...filtered].sort((a, b) => compare(a[sort.col], b[sort.col], sort.dir));
  }, [filtered, sort]);

  const limited = maxRows > 0 ? sorted.slice(0, maxRows) : sorted;

  const totalPages = Math.ceil(limited.length / pageSize);
  const pageRows = limited.slice(page * pageSize, (page + 1) * pageSize);

  function toggleSort(col: string) {
    setSort((prev) => {
      if (prev?.col === col) {
        return prev.dir === "asc" ? { col, dir: "desc" } : null;
      }
      return { col, dir: "asc" };
    });
    setPage(0);
  }

  if (rows.length === 0) {
    return (
      <div
        className="rounded-[8px] flex items-center justify-center py-12"
        style={{
          background: "#ffffff",
          border: "1px solid rgba(38,37,30,0.1)",
          color: "rgba(38,37,30,0.35)",
          fontFamily: "system-ui",
          fontSize: "13px",
        }}
      >
        No data
      </div>
    );
  }

  return (
    <div
      className="rounded-[8px] overflow-hidden"
      style={{
        background: "#ffffff",
        border: "1px solid rgba(38,37,30,0.1)",
      }}
    >
      {/* Search bar */}
      {searchable && (
        <div
          className="px-4 py-3 flex items-center gap-2"
          style={{ borderBottom: "1px solid rgba(38,37,30,0.08)" }}
        >
          <Search size={14} strokeWidth={1.5} style={{ color: "rgba(38,37,30,0.35)", flexShrink: 0 }} />
          <input
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setPage(0);
            }}
            placeholder="Search…"
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
          <span
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "11px",
              color: "rgba(38,37,30,0.35)",
            }}
          >
            {filtered.length.toLocaleString()} rows
          </span>
        </div>
      )}

      {/* Table */}
      <div className="overflow-x-auto">
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr
              style={{
                background: "#f7f7f4",
                borderBottom: "1px solid rgba(38,37,30,0.1)",
              }}
            >
              {cols.map((col) => (
                <th
                  key={col}
                  onClick={() => toggleSort(col)}
                  className="cursor-pointer select-none"
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
                  <span className="flex items-center gap-1">
                    {col}
                    {sort?.col === col ? (
                      sort.dir === "asc" ? (
                        <ChevronDown size={10} strokeWidth={2} />
                      ) : (
                        <ChevronRight
                          size={10}
                          strokeWidth={2}
                          style={{ transform: "rotate(90deg)" }}
                        />
                      )
                    ) : null}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, ri) => (
              <tr
                key={ri}
                onClick={() => onRowClick?.(row, page * pageSize + ri)}
                style={{
                  borderBottom:
                    ri < pageRows.length - 1
                      ? "1px solid rgba(38,37,30,0.06)"
                      : "none",
                  cursor: onRowClick ? "pointer" : "default",
                }}
                onMouseEnter={(e) => {
                  if (onRowClick) {
                    (e.currentTarget as HTMLElement).style.background = "#fafaf8";
                  }
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.background = "transparent";
                }}
              >
                {cols.map((col) => {
                  const val = row[col];
                  const num = isNumeric(val);
                  return (
                    <td
                      key={col}
                      style={{
                        fontFamily: num
                          ? "var(--font-mono, monospace)"
                          : "var(--font-satoshi, system-ui, sans-serif)",
                        fontSize: num ? "13px" : "14px",
                        fontVariantNumeric: num ? "tabular-nums" : undefined,
                        color: "#26251e",
                        padding: "10px 14px",
                        whiteSpace: "nowrap",
                        maxWidth: "220px",
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                      }}
                    >
                      {cellStr(val)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div
          className="px-4 py-3 flex items-center justify-between"
          style={{ borderTop: "1px solid rgba(38,37,30,0.08)" }}
        >
          <span
            style={{
              fontFamily: "system-ui",
              fontSize: "12px",
              color: "rgba(38,37,30,0.45)",
            }}
          >
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <PageBtn
              label="Previous"
              disabled={page === 0}
              onClick={() => setPage((p) => p - 1)}
            />
            <PageBtn
              label="Next"
              disabled={page >= totalPages - 1}
              onClick={() => setPage((p) => p + 1)}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function PageBtn({
  label,
  disabled,
  onClick,
}: {
  label: string;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: "#ebeae5",
        border: "none",
        borderRadius: "6px",
        padding: "5px 10px",
        fontFamily: "system-ui",
        fontSize: "12px",
        color: disabled ? "rgba(38,37,30,0.25)" : "#26251e",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "color 150ms ease",
      }}
      onMouseEnter={(e) => {
        if (!disabled) (e.currentTarget as HTMLElement).style.color = "#cf2d56";
      }}
      onMouseLeave={(e) => {
        if (!disabled) (e.currentTarget as HTMLElement).style.color = "#26251e";
      }}
    >
      {label}
    </button>
  );
}
