"use client";

import { useState } from "react";
import { Download, ChevronDown, ChevronRight } from "lucide-react";
import type { GenerationResult } from "@/lib/stores/generationStore";

// ── Helpers ───────────────────────────────────────────────────────────────────

function qualityColor(score: number): string {
  if (score > 80) return "#1f8a65";
  if (score >= 60) return "#c08532";
  return "#cf2d56";
}

function formatRows(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return String(n);
}

// ── Section toggle ────────────────────────────────────────────────────────────

function Section({
  label,
  children,
  defaultOpen = false,
}: {
  label: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ borderTop: "1px solid rgba(38,37,30,0.08)" }}>
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 transition-colors"
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          textAlign: "left",
        }}
        onMouseEnter={(e) => {
          (e.currentTarget as HTMLElement).style.background = "#f7f7f4";
        }}
        onMouseLeave={(e) => {
          (e.currentTarget as HTMLElement).style.background = "transparent";
        }}
      >
        <span
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            fontWeight: 600,
            color: "rgba(38,37,30,0.55)",
            textTransform: "uppercase",
            letterSpacing: "0.048px",
          }}
        >
          {label}
        </span>
        {open ? (
          <ChevronDown size={14} strokeWidth={1.5} style={{ color: "rgba(38,37,30,0.4)" }} />
        ) : (
          <ChevronRight size={14} strokeWidth={1.5} style={{ color: "rgba(38,37,30,0.4)" }} />
        )}
      </button>
      {open && <div className="px-4 pb-4">{children}</div>}
    </div>
  );
}

// ── Mini data table ───────────────────────────────────────────────────────────

function PreviewTable({ rows }: { rows: Record<string, unknown>[] }) {
  if (!rows.length) return null;
  const cols = Object.keys(rows[0]);
  const displayCols = cols.slice(0, 6);
  const displayRows = rows.slice(0, 10);

  return (
    <div className="overflow-x-auto rounded-[6px]" style={{ border: "1px solid rgba(38,37,30,0.08)" }}>
      <table className="w-full" style={{ borderCollapse: "collapse", minWidth: "100%" }}>
        <thead>
          <tr style={{ background: "#f7f7f4", borderBottom: "1px solid rgba(38,37,30,0.1)" }}>
            {displayCols.map((col) => (
              <th
                key={col}
                style={{
                  fontFamily: "system-ui",
                  fontSize: "11px",
                  fontWeight: 600,
                  color: "rgba(38,37,30,0.55)",
                  textTransform: "uppercase",
                  letterSpacing: "0.048px",
                  padding: "8px 10px",
                  textAlign: "left",
                  whiteSpace: "nowrap",
                }}
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {displayRows.map((row, i) => (
            <tr
              key={i}
              style={{
                borderBottom:
                  i < displayRows.length - 1
                    ? "1px solid rgba(38,37,30,0.06)"
                    : "none",
              }}
            >
              {displayCols.map((col) => {
                const val = row[col];
                const isNum = typeof val === "number";
                return (
                  <td
                    key={col}
                    style={{
                      fontFamily: isNum ? "var(--font-mono, monospace)" : "var(--font-satoshi, system-ui, sans-serif)",
                      fontSize: isNum ? "12px" : "13px",
                      fontVariantNumeric: isNum ? "tabular-nums" : undefined,
                      color: "#26251e",
                      padding: "7px 10px",
                      whiteSpace: "nowrap",
                      maxWidth: "160px",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {String(val ?? "")}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Download button ───────────────────────────────────────────────────────────

function DownloadBtn({
  label,
  href,
}: {
  label: string;
  href: string;
}) {
  return (
    <a
      href={href}
      download
      className="flex items-center gap-1.5 transition-opacity hover:opacity-80"
      style={{
        background: "#ebeae5",
        color: "#26251e",
        border: "none",
        borderRadius: "6px",
        padding: "7px 12px",
        fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
        fontSize: "13px",
        fontWeight: 400,
        textDecoration: "none",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.color = "#cf2d56";
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.color = "#26251e";
      }}
    >
      <Download size={13} strokeWidth={1.5} />
      {label}
    </a>
  );
}

// ── Quality breakdown ─────────────────────────────────────────────────────────

function QualityBreakdown({ report }: { report: GenerationResult["qualityReport"] }) {
  if (!report) return null;
  const items: { label: string; score: number }[] = [
    { label: "Statistical", score: report.statistical },
    { label: "Causal", score: report.causal },
    { label: "Privacy", score: report.privacy },
    { label: "Diversity", score: report.diversity },
  ];
  return (
    <div className="grid grid-cols-2 gap-2">
      {items.map(({ label, score }) => (
        <div
          key={label}
          className="rounded-[6px] p-3"
          style={{ background: "#f7f7f4", border: "1px solid rgba(38,37,30,0.06)" }}
        >
          <span
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "18px",
              color: qualityColor(score),
            }}
          >
            {score.toFixed(0)}
          </span>
          <p
            style={{
              fontFamily: "system-ui",
              fontSize: "10px",
              fontWeight: 500,
              textTransform: "uppercase",
              letterSpacing: "0.048px",
              color: "rgba(38,37,30,0.45)",
              marginTop: "2px",
            }}
          >
            {label}
          </p>
        </div>
      ))}
    </div>
  );
}

// ── Main card ─────────────────────────────────────────────────────────────────

interface GenerationCardProps {
  result: GenerationResult;
}

export default function GenerationCard({ result }: GenerationCardProps) {
  const {
    qualityScore,
    rowCount,
    colCount,
    domain,
    previewRows,
    downloadUrls,
    generatedCode,
    qualityReport,
    schema,
    createdAt,
  } = result;

  return (
    <div
      className="rounded-[8px] overflow-hidden"
      style={{
        background: "#ffffff",
        border: "1px solid rgba(38,37,30,0.1)",
        boxShadow: "rgba(0,0,0,0.04) 0px 4px 12px",
      }}
    >
      {/* Header */}
      <div className="px-4 pt-4 pb-3 flex items-start gap-4">
        {/* Quality score */}
        <div className="flex flex-col items-center">
          <span
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "32px",
              fontWeight: 400,
              color: qualityColor(qualityScore),
              lineHeight: 1.1,
              letterSpacing: "-0.64px",
            }}
          >
            {qualityScore.toFixed(1)}
          </span>
          <span
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "9px",
              fontWeight: 500,
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              color: "rgba(38,37,30,0.45)",
              marginTop: "2px",
            }}
          >
            Quality
          </span>
        </div>

        {/* Stats */}
        <div className="flex-1">
          <p
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "15px",
              fontWeight: 400,
              color: "#26251e",
              marginBottom: "6px",
            }}
          >
            {domain} dataset generated
          </p>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {[
              { label: "Rows", val: formatRows(rowCount) },
              { label: "Cols", val: String(colCount) },
              {
                label: "Created",
                val: new Date(createdAt).toLocaleDateString("en-IN", {
                  day: "numeric",
                  month: "short",
                  year: "numeric",
                }),
              },
            ].map(({ label, val }) => (
              <span
                key={label}
                style={{
                  fontFamily: "system-ui",
                  fontSize: "12px",
                  color: "rgba(38,37,30,0.5)",
                }}
              >
                <span style={{ fontWeight: 500 }}>{label}: </span>
                <span
                  style={{ fontFamily: "var(--font-mono, monospace)", fontSize: "12px" }}
                >
                  {val}
                </span>
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Download row */}
      <div
        className="px-4 py-3 flex flex-wrap gap-2"
        style={{ borderTop: "1px solid rgba(38,37,30,0.08)", background: "#fafaf8" }}
      >
        <DownloadBtn label="CSV" href={downloadUrls.csv} />
        <DownloadBtn label="Excel" href={downloadUrls.excel} />
        <DownloadBtn label="JSON" href={downloadUrls.json} />
        <DownloadBtn label="Parquet" href={downloadUrls.parquet} />
      </div>

      {/* Expandable sections */}
      <Section label="Data Preview (10 rows)" defaultOpen>
        <PreviewTable rows={previewRows} />
      </Section>

      {qualityReport && (
        <Section label="Quality Report">
          <QualityBreakdown report={qualityReport} />
        </Section>
      )}

      {schema && Object.keys(schema).length > 0 && (
        <Section label="Schema">
          <pre
            className="overflow-x-auto rounded-[6px] p-3"
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "11px",
              color: "#26251e",
              background: "#f7f7f4",
              lineHeight: 1.67,
              maxHeight: "240px",
              overflowY: "auto",
            }}
          >
            {JSON.stringify(schema, null, 2)}
          </pre>
        </Section>
      )}

      {generatedCode && (
        <Section label="Generated Code">
          <pre
            className="overflow-x-auto rounded-[6px] p-3"
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "11px",
              color: "#26251e",
              background: "#f7f7f4",
              lineHeight: 1.67,
              maxHeight: "300px",
              overflowY: "auto",
            }}
          >
            {generatedCode}
          </pre>
        </Section>
      )}
    </div>
  );
}
