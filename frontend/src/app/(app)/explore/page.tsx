"use client";

import { useState } from "react";
import { Database, Search, BarChart2, List, FileText, X } from "lucide-react";
import UploadZone, { UploadButton, formatBytes } from "@/components/data/UploadZone";
import DataTable from "@/components/data/DataTable";
import ChartRenderer from "@/components/data/ChartRenderer";
import ChatInput, { type ChatInputOptions } from "@/components/chat/ChatInput";

// ── Mock datasets ─────────────────────────────────────────────────────────────

interface MockDataset {
  id: string;
  name: string;
  rowCount: number;
  colCount: number;
  sizeBytes: number;
  format: string;
  createdAt: string;
}

const MOCK_DATASETS: MockDataset[] = [
  {
    id: "d1",
    name: "healthcare_patients_mumbai.csv",
    rowCount: 10000,
    colCount: 18,
    sizeBytes: 2_400_000,
    format: "csv",
    createdAt: "2026-04-24",
  },
  {
    id: "d2",
    name: "banking_transactions_hdfc.parquet",
    rowCount: 50000,
    colCount: 22,
    sizeBytes: 8_700_000,
    format: "parquet",
    createdAt: "2026-04-23",
  },
  {
    id: "d3",
    name: "retail_ecommerce_india.csv",
    rowCount: 25000,
    colCount: 15,
    sizeBytes: 4_100_000,
    format: "csv",
    createdAt: "2026-04-22",
  },
];

const TOTAL_STORAGE = 25 * 1024 * 1024 * 1024; // 25 GB
const USED_STORAGE = MOCK_DATASETS.reduce((a, d) => a + d.sizeBytes, 0);

// ── Mock preview data ─────────────────────────────────────────────────────────

function mockPreviewRows(): Record<string, unknown>[] {
  return Array.from({ length: 20 }, (_, i) => ({
    id: i + 1,
    patient_id: `MUM${String(i + 1001).padStart(6, "0")}`,
    age: 25 + Math.floor(Math.random() * 60),
    gender: i % 2 === 0 ? "M" : "F",
    diagnosis: ["Hypertension", "Diabetes", "Cardiac", "Respiratory"][i % 4],
    systolic_bp: 110 + Math.floor(Math.random() * 50),
    diastolic_bp: 70 + Math.floor(Math.random() * 30),
    cholesterol: (4.0 + Math.random() * 4).toFixed(1),
    admission_date: `2026-0${(i % 4) + 1}-${String((i % 28) + 1).padStart(2, "0")}`,
  }));
}

function mockSchemaRows(): Record<string, unknown>[] {
  return [
    { column: "patient_id", type: "string", nullable: "No", unique: "Yes", example: "MUM001001" },
    { column: "age", type: "integer", nullable: "No", unique: "No", example: "45" },
    { column: "gender", type: "string", nullable: "No", unique: "No", example: "M" },
    { column: "diagnosis", type: "string", nullable: "Yes", unique: "No", example: "Hypertension" },
    { column: "systolic_bp", type: "float", nullable: "No", unique: "No", example: "132.5" },
    { column: "diastolic_bp", type: "float", nullable: "No", unique: "No", example: "84.0" },
    { column: "cholesterol", type: "float", nullable: "Yes", unique: "No", example: "5.2" },
    { column: "admission_date", type: "datetime", nullable: "No", unique: "No", example: "2026-04-15" },
  ];
}

function mockStatsData(): Record<string, unknown>[] {
  return [
    { age_bin: "20-30", count: 1240 },
    { age_bin: "31-40", count: 2180 },
    { age_bin: "41-50", count: 2840 },
    { age_bin: "51-60", count: 2100 },
    { age_bin: "61-70", count: 1120 },
    { age_bin: "71+", count: 520 },
  ];
}

// ── Tabs ──────────────────────────────────────────────────────────────────────

type TabId = "chat" | "preview" | "schema" | "stats";

const TABS: { id: TabId; label: string; icon: React.ComponentType<{ size?: number; strokeWidth?: number }> }[] = [
  { id: "chat", label: "Chat", icon: Search },
  { id: "preview", label: "Preview", icon: List },
  { id: "schema", label: "Schema", icon: FileText },
  { id: "stats", label: "Stats", icon: BarChart2 },
];

// ── Dataset sidebar item ──────────────────────────────────────────────────────

function DatasetItem({
  ds,
  active,
  onClick,
}: {
  ds: MockDataset;
  active: boolean;
  onClick: () => void;
}) {
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
          fontSize: "12px",
          color: active ? "#26251e" : "rgba(38,37,30,0.65)",
          marginBottom: "2px",
        }}
      >
        {ds.name}
      </p>
      <div className="flex items-center gap-2">
        <span
          style={{
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "11px",
            color: "rgba(38,37,30,0.4)",
          }}
        >
          {ds.rowCount.toLocaleString()} rows
        </span>
        <span
          style={{
            fontFamily: "system-ui",
            fontSize: "11px",
            color: "rgba(38,37,30,0.35)",
          }}
        >
          {formatBytes(ds.sizeBytes)}
        </span>
      </div>
    </button>
  );
}

// ── Dataset sidebar ───────────────────────────────────────────────────────────

function DatasetSidebar({
  datasets,
  selectedId,
  onSelect,
  onUpload,
}: {
  datasets: MockDataset[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onUpload: (file: File) => Promise<void>;
}) {
  const usedPct = (USED_STORAGE / TOTAL_STORAGE) * 100;

  return (
    <aside
      className="hidden md:flex flex-col flex-shrink-0"
      style={{
        width: "240px",
        background: "#f7f7f4",
        borderRight: "1px solid rgba(38,37,30,0.08)",
      }}
    >
      <div className="p-3 flex items-center justify-between">
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
          Datasets
        </span>
        <UploadButton onUpload={onUpload} />
      </div>

      <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
        {datasets.map((ds) => (
          <DatasetItem
            key={ds.id}
            ds={ds}
            active={ds.id === selectedId}
            onClick={() => onSelect(ds.id)}
          />
        ))}
      </div>

      {/* Storage indicator */}
      <div
        className="p-3"
        style={{ borderTop: "1px solid rgba(38,37,30,0.08)" }}
      >
        <div className="flex justify-between mb-1">
          <span
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              color: "rgba(38,37,30,0.45)",
            }}
          >
            Storage
          </span>
          <span
            style={{
              fontFamily: "var(--font-mono, monospace)",
              fontSize: "11px",
              color: "rgba(38,37,30,0.45)",
            }}
          >
            {formatBytes(USED_STORAGE)} / 25 GB
          </span>
        </div>
        <div
          className="rounded-full overflow-hidden"
          style={{ height: "3px", background: "rgba(38,37,30,0.08)" }}
        >
          <div
            style={{
              width: `${usedPct}%`,
              height: "100%",
              background: "#f54e00",
              borderRadius: "9999px",
            }}
          />
        </div>
      </div>
    </aside>
  );
}

// ── Tab bar ───────────────────────────────────────────────────────────────────

function TabBar({
  active,
  onChange,
}: {
  active: TabId;
  onChange: (t: TabId) => void;
}) {
  return (
    <div
      className="flex items-center gap-1 px-4 pt-4 pb-0"
      style={{ borderBottom: "1px solid rgba(38,37,30,0.1)" }}
    >
      {TABS.map(({ id, label, icon: Icon }) => {
        const isActive = id === active;
        return (
          <button
            key={id}
            onClick={() => onChange(id)}
            className="flex items-center gap-1.5 px-3 py-2 transition-colors"
            style={{
              background: "none",
              border: "none",
              borderBottom: isActive ? "2px solid #f54e00" : "2px solid transparent",
              cursor: "pointer",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "13px",
              fontWeight: 400,
              color: isActive ? "#26251e" : "rgba(38,37,30,0.5)",
              paddingBottom: "10px",
              borderRadius: 0,
              marginBottom: "-1px",
            }}
          >
            <Icon size={14} strokeWidth={1.5} />
            {label}
          </button>
        );
      })}
    </div>
  );
}

// ── Chat tab (simplified) ─────────────────────────────────────────────────────

interface ChatMsg {
  id: string;
  role: "user" | "assistant";
  content: string;
}

function ExploreChat({ datasetName }: { datasetName: string }) {
  const [msgs, setMsgs] = useState<ChatMsg[]>([]);

  function handleSend(content: string, _opts: ChatInputOptions) {
    const userMsg: ChatMsg = { id: crypto.randomUUID(), role: "user", content };
    const asstMsg: ChatMsg = {
      id: crypto.randomUUID(),
      role: "assistant",
      content: `Analysing **${datasetName}**: ${content} — (Connect to backend for real analysis)`,
    };
    setMsgs((prev) => [...prev, userMsg, asstMsg]);
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {msgs.length === 0 && (
          <p
            className="text-center pt-8"
            style={{
              fontFamily: "var(--font-serif, Georgia, serif)",
              fontSize: "16px",
              color: "rgba(38,37,30,0.4)",
            }}
          >
            Ask anything about this dataset
          </p>
        )}
        {msgs.map((m) => (
          <div
            key={m.id}
            className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className="max-w-[75%] px-4 py-2.5 rounded-[8px]"
              style={{
                background: m.role === "user" ? "#26251e" : "#ffffff",
                color: m.role === "user" ? "#ffffff" : "#26251e",
                border: m.role === "assistant" ? "1px solid rgba(38,37,30,0.1)" : "none",
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "14px",
              }}
            >
              {m.content}
            </div>
          </div>
        ))}
      </div>
      <div
        className="flex-shrink-0 p-4"
        style={{ borderTop: "1px solid rgba(38,37,30,0.08)" }}
      >
        <ChatInput onSend={handleSend} placeholder={`Ask about ${datasetName}…`} />
      </div>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function ExplorePage() {
  const [datasets, setDatasets] = useState<MockDataset[]>(MOCK_DATASETS);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("preview");

  const selected = datasets.find((d) => d.id === selectedId) ?? null;

  async function handleUpload(file: File): Promise<void> {
    // Simulate upload
    await new Promise((r) => setTimeout(r, 1200));
    const newDs: MockDataset = {
      id: crypto.randomUUID(),
      name: file.name,
      rowCount: Math.floor(Math.random() * 50000) + 1000,
      colCount: Math.floor(Math.random() * 20) + 5,
      sizeBytes: file.size,
      format: file.name.split(".").pop() ?? "csv",
      createdAt: new Date().toISOString().slice(0, 10),
    };
    setDatasets((prev) => [newDs, ...prev]);
    setSelectedId(newDs.id);
  }

  return (
    <div className="flex h-full">
      <DatasetSidebar
        datasets={datasets}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onUpload={handleUpload}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        {!selected ? (
          /* No selection — show upload zone */
          <div className="flex-1 flex items-center justify-center p-8">
            <div className="w-full max-w-[480px]">
              <div className="flex items-center gap-2 mb-6">
                <Database size={20} strokeWidth={1.5} style={{ color: "rgba(38,37,30,0.35)" }} />
                <h2
                  style={{
                    fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                    fontSize: "20px",
                    fontWeight: 400,
                    color: "#26251e",
                    letterSpacing: "-0.2px",
                  }}
                >
                  Explore a dataset
                </h2>
              </div>
              <p
                className="mb-6"
                style={{
                  fontFamily: "var(--font-serif, Georgia, serif)",
                  fontSize: "16px",
                  color: "rgba(38,37,30,0.5)",
                  lineHeight: 1.5,
                }}
              >
                Select a dataset from the sidebar, or upload a new one to start exploring.
              </p>
              <UploadZone onUpload={handleUpload} />
            </div>
          </div>
        ) : (
          /* Dataset selected */
          <div className="flex flex-col h-full">
            {/* Top bar */}
            <div
              className="flex items-center justify-between px-5 py-3 flex-shrink-0"
              style={{ borderBottom: "1px solid rgba(38,37,30,0.08)", background: "#ffffff" }}
            >
              <div>
                <p
                  style={{
                    fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                    fontSize: "14px",
                    fontWeight: 400,
                    color: "#26251e",
                  }}
                >
                  {selected.name}
                </p>
                <div className="flex items-center gap-3 mt-0.5">
                  {[
                    `${selected.rowCount.toLocaleString()} rows`,
                    `${selected.colCount} cols`,
                    formatBytes(selected.sizeBytes),
                    selected.format.toUpperCase(),
                  ].map((item, i) => (
                    <span
                      key={i}
                      style={{
                        fontFamily: "var(--font-mono, monospace)",
                        fontSize: "11px",
                        color: "rgba(38,37,30,0.4)",
                      }}
                    >
                      {item}
                    </span>
                  ))}
                </div>
              </div>
              <button
                onClick={() => setSelectedId(null)}
                style={{
                  background: "none",
                  border: "none",
                  cursor: "pointer",
                  color: "rgba(38,37,30,0.35)",
                }}
                className="transition-colors hover:text-[#26251e]"
              >
                <X size={16} strokeWidth={1.5} />
              </button>
            </div>

            {/* Tab bar */}
            <TabBar active={activeTab} onChange={setActiveTab} />

            {/* Tab content */}
            <div className="flex-1 overflow-auto">
              {activeTab === "chat" && (
                <ExploreChat datasetName={selected.name} />
              )}

              {activeTab === "preview" && (
                <div className="p-4">
                  <DataTable rows={mockPreviewRows()} pageSize={20} />
                </div>
              )}

              {activeTab === "schema" && (
                <div className="p-4">
                  <DataTable
                    rows={mockSchemaRows()}
                    columns={["column", "type", "nullable", "unique", "example"]}
                    searchable={false}
                    pageSize={50}
                  />
                </div>
              )}

              {activeTab === "stats" && (
                <div className="p-4 space-y-6">
                  <div
                    className="rounded-[8px] p-5"
                    style={{
                      background: "#ffffff",
                      border: "1px solid rgba(38,37,30,0.1)",
                    }}
                  >
                    <ChartRenderer
                      data={mockStatsData()}
                      xKey="age_bin"
                      yKeys={["count"]}
                      title="Age distribution"
                      height={220}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
