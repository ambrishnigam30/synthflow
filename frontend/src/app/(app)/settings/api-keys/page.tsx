"use client";

import { useState } from "react";
import { Plus, Copy, X, Check } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

type Scope = "generate" | "read" | "admin";

interface ApiKey {
  id: string;
  name: string;
  prefix: string;
  scopes: Scope[];
  createdAt: string;
  lastUsed: string | null;
  revoked: boolean;
}

// ── Mock data ─────────────────────────────────────────────────────────────────

const MOCK_KEYS: ApiKey[] = [
  {
    id: "k1",
    name: "Production API",
    prefix: "sf_live_a4bK…",
    scopes: ["generate", "read"],
    createdAt: "2026-03-15",
    lastUsed: "2026-04-24",
    revoked: false,
  },
  {
    id: "k2",
    name: "CI/CD Pipeline",
    prefix: "sf_live_c8xP…",
    scopes: ["read"],
    createdAt: "2026-02-20",
    lastUsed: "2026-04-22",
    revoked: false,
  },
  {
    id: "k3",
    name: "Old dev key",
    prefix: "sf_live_e1mN…",
    scopes: ["generate", "read", "admin"],
    createdAt: "2026-01-10",
    lastUsed: "2026-02-14",
    revoked: true,
  },
];

const ALL_SCOPES: { id: Scope; label: string; description: string }[] = [
  { id: "generate", label: "Generate", description: "Create new datasets" },
  { id: "read", label: "Read", description: "Read datasets and history" },
  { id: "admin", label: "Admin", description: "Manage settings and keys" },
];

// ── Copy button ───────────────────────────────────────────────────────────────

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  function handleCopy() {
    void navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <button
      onClick={handleCopy}
      className="flex items-center gap-1 transition-all"
      style={{
        background: "#ebeae5",
        border: "none",
        borderRadius: "6px",
        padding: "5px 10px",
        fontFamily: "system-ui",
        fontSize: "12px",
        color: copied ? "#1f8a65" : "#26251e",
        cursor: "pointer",
      }}
    >
      {copied ? <Check size={12} strokeWidth={2} /> : <Copy size={12} strokeWidth={1.5} />}
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

// ── Create key modal ──────────────────────────────────────────────────────────

function CreateKeyModal({
  onClose,
  onCreate,
}: {
  onClose: () => void;
  onCreate: (name: string, scopes: Scope[], fullKey: string) => void;
}) {
  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<Scope[]>(["generate", "read"]);
  const [creating, setCreating] = useState(false);

  function toggleScope(s: Scope) {
    setScopes((prev) =>
      prev.includes(s) ? prev.filter((x) => x !== s) : [...prev, s]
    );
  }

  async function handleCreate(e: React.FormEvent) {
    e.preventDefault();
    if (!name.trim() || !scopes.length) return;
    setCreating(true);
    await new Promise((r) => setTimeout(r, 600));
    const fullKey = `sf_live_${crypto.randomUUID().replace(/-/g, "").slice(0, 32)}`;
    onCreate(name.trim(), scopes, fullKey);
    setCreating(false);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(38,37,30,0.4)" }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-[460px] rounded-[8px] p-6"
        style={{
          background: "#ffffff",
          boxShadow: "rgba(0,0,0,0.14) 0px 28px 70px, rgba(0,0,0,0.1) 0px 14px 32px",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-5">
          <h2
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "16px",
              fontWeight: 400,
              color: "#26251e",
            }}
          >
            Create API key
          </h2>
          <button
            onClick={onClose}
            style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(38,37,30,0.4)" }}
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>

        <form onSubmit={handleCreate} className="space-y-5">
          <div>
            <label
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.048px",
                color: "rgba(38,37,30,0.45)",
                display: "block",
                marginBottom: "6px",
              }}
            >
              Name
            </label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
              placeholder="Production API, CI/CD pipeline…"
              style={{
                width: "100%",
                background: "transparent",
                border: "1px solid rgba(38,37,30,0.12)",
                borderRadius: "8px",
                padding: "9px 12px",
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "14px",
                color: "#26251e",
                outline: "none",
                boxSizing: "border-box",
              }}
              onFocus={(e) => { e.target.style.borderColor = "rgba(38,37,30,0.25)"; }}
              onBlur={(e) => { e.target.style.borderColor = "rgba(38,37,30,0.12)"; }}
            />
          </div>

          <div>
            <label
              style={{
                fontFamily: "system-ui",
                fontSize: "11px",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.048px",
                color: "rgba(38,37,30,0.45)",
                display: "block",
                marginBottom: "8px",
              }}
            >
              Scopes
            </label>
            <div className="space-y-2">
              {ALL_SCOPES.map((s) => (
                <label
                  key={s.id}
                  className="flex items-start gap-3 cursor-pointer"
                  style={{
                    padding: "10px 12px",
                    borderRadius: "8px",
                    border: `1px solid ${scopes.includes(s.id) ? "rgba(245,78,0,0.3)" : "rgba(38,37,30,0.1)"}`,
                    background: scopes.includes(s.id) ? "rgba(245,78,0,0.04)" : "transparent",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={scopes.includes(s.id)}
                    onChange={() => toggleScope(s.id)}
                    style={{ marginTop: "2px", accentColor: "#f54e00" }}
                  />
                  <div>
                    <p
                      style={{
                        fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                        fontSize: "13px",
                        color: "#26251e",
                        fontWeight: 400,
                      }}
                    >
                      {s.label}
                    </p>
                    <p
                      style={{
                        fontFamily: "system-ui",
                        fontSize: "11px",
                        color: "rgba(38,37,30,0.45)",
                        marginTop: "1px",
                      }}
                    >
                      {s.description}
                    </p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={creating || !name.trim() || !scopes.length}
            className="w-full transition-opacity hover:opacity-90"
            style={{
              background: "#f54e00",
              color: "#ffffff",
              border: "none",
              borderRadius: "8px",
              padding: "10px",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "14px",
              cursor: creating ? "not-allowed" : "pointer",
              opacity: (!name.trim() || !scopes.length) ? 0.5 : 1,
            }}
          >
            {creating ? "Creating…" : "Create key"}
          </button>
        </form>
      </div>
    </div>
  );
}

// ── New key reveal ────────────────────────────────────────────────────────────

function NewKeyReveal({ fullKey, onDismiss }: { fullKey: string; onDismiss: () => void }) {
  return (
    <div
      className="rounded-[8px] p-5 mb-4"
      style={{
        background: "rgba(31,138,101,0.06)",
        border: "1px solid rgba(31,138,101,0.25)",
      }}
    >
      <div className="flex items-start justify-between gap-3 mb-2">
        <div>
          <p
            style={{
              fontFamily: "system-ui",
              fontSize: "12px",
              fontWeight: 600,
              color: "#1f8a65",
              marginBottom: "4px",
            }}
          >
            ✓ Key created — save it now
          </p>
          <p
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              color: "rgba(38,37,30,0.55)",
              lineHeight: 1.5,
            }}
          >
            This is the only time you&apos;ll see the full key. Copy it somewhere safe.
          </p>
        </div>
        <button
          onClick={onDismiss}
          style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(38,37,30,0.35)", flexShrink: 0 }}
        >
          <X size={14} strokeWidth={1.5} />
        </button>
      </div>
      <div
        className="flex items-center gap-2 rounded-[6px] px-3 py-2"
        style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
      >
        <code
          style={{
            flex: 1,
            fontFamily: "var(--font-mono, monospace)",
            fontSize: "13px",
            color: "#26251e",
            wordBreak: "break-all",
          }}
        >
          {fullKey}
        </code>
        <CopyButton text={fullKey} />
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>(MOCK_KEYS);
  const [showCreate, setShowCreate] = useState(false);
  const [newFullKey, setNewFullKey] = useState<string | null>(null);

  function handleCreate(name: string, scopes: Scope[], fullKey: string) {
    const newKey: ApiKey = {
      id: crypto.randomUUID(),
      name,
      prefix: `sf_live_${fullKey.slice(8, 12)}…`,
      scopes,
      createdAt: new Date().toISOString().slice(0, 10),
      lastUsed: null,
      revoked: false,
    };
    setKeys((prev) => [newKey, ...prev]);
    setNewFullKey(fullKey);
    setShowCreate(false);
  }

  function handleRevoke(id: string) {
    setKeys((prev) => prev.map((k) => k.id === id ? { ...k, revoked: true } : k));
  }

  const activeKeys = keys.filter((k) => !k.revoked);
  const revokedKeys = keys.filter((k) => k.revoked);

  return (
    <div className="max-w-[680px]">
      {newFullKey && (
        <NewKeyReveal fullKey={newFullKey} onDismiss={() => setNewFullKey(null)} />
      )}

      {/* Header */}
      <div className="flex items-center justify-between mb-5">
        <p
          style={{
            fontFamily: "var(--font-serif, Georgia, serif)",
            fontSize: "15px",
            color: "rgba(38,37,30,0.55)",
            lineHeight: 1.5,
          }}
        >
          API keys authenticate your requests to the SynthFlow API.
        </p>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-1.5 flex-shrink-0 ml-4 transition-opacity hover:opacity-90"
          style={{
            background: "#f54e00",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            padding: "9px 14px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            cursor: "pointer",
            whiteSpace: "nowrap",
          }}
        >
          <Plus size={14} strokeWidth={2} />
          Create API key
        </button>
      </div>

      {/* Keys table */}
      <KeyTable keys={activeKeys} onRevoke={handleRevoke} />

      {revokedKeys.length > 0 && (
        <div className="mt-6">
          <p
            className="mb-3"
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.048px",
              color: "rgba(38,37,30,0.35)",
            }}
          >
            Revoked keys
          </p>
          <KeyTable keys={revokedKeys} revoked />
        </div>
      )}

      {showCreate && (
        <CreateKeyModal onClose={() => setShowCreate(false)} onCreate={handleCreate} />
      )}
    </div>
  );
}

function ScopeBadge({ scope }: { scope: Scope }) {
  const colors: Record<Scope, { bg: string; color: string }> = {
    generate: { bg: "rgba(245,78,0,0.1)", color: "#f54e00" },
    read: { bg: "rgba(159,187,224,0.25)", color: "#4a7ab5" },
    admin: { bg: "rgba(192,133,50,0.1)", color: "#c08532" },
  };
  const s = colors[scope];
  return (
    <span
      className="px-1.5 py-0.5 rounded-[3px]"
      style={{
        background: s.bg,
        color: s.color,
        fontFamily: "system-ui",
        fontSize: "10px",
        fontWeight: 500,
        textTransform: "uppercase",
        letterSpacing: "0.04px",
      }}
    >
      {scope}
    </span>
  );
}

function KeyTable({
  keys,
  onRevoke,
  revoked = false,
}: {
  keys: ApiKey[];
  onRevoke?: (id: string) => void;
  revoked?: boolean;
}) {
  if (!keys.length) return null;

  return (
    <div
      className="rounded-[8px] overflow-hidden"
      style={{
        background: "#ffffff",
        border: "1px solid rgba(38,37,30,0.1)",
        opacity: revoked ? 0.65 : 1,
      }}
    >
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr style={{ background: "#f7f7f4", borderBottom: "1px solid rgba(38,37,30,0.1)" }}>
            {["Name", "Key", "Scopes", "Created", "Last used", ""].map((h) => (
              <th
                key={h}
                style={{
                  fontFamily: "system-ui",
                  fontSize: "11px",
                  fontWeight: 600,
                  color: "rgba(38,37,30,0.45)",
                  textTransform: "uppercase",
                  letterSpacing: "0.048px",
                  padding: "10px 12px",
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
          {keys.map((key, i) => (
            <tr
              key={key.id}
              style={{
                borderBottom: i < keys.length - 1 ? "1px solid rgba(38,37,30,0.06)" : "none",
              }}
            >
              <td
                style={{
                  padding: "12px 12px",
                  fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                  fontSize: "13px",
                  color: "#26251e",
                  textDecoration: key.revoked ? "line-through" : "none",
                }}
              >
                {key.name}
              </td>
              <td
                style={{
                  padding: "12px 12px",
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "12px",
                  color: "rgba(38,37,30,0.5)",
                }}
              >
                {key.prefix}
              </td>
              <td style={{ padding: "12px 12px" }}>
                <div className="flex flex-wrap gap-1">
                  {key.scopes.map((s) => <ScopeBadge key={s} scope={s} />)}
                </div>
              </td>
              <td
                style={{
                  padding: "12px 12px",
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "12px",
                  color: "rgba(38,37,30,0.45)",
                  whiteSpace: "nowrap",
                }}
              >
                {key.createdAt}
              </td>
              <td
                style={{
                  padding: "12px 12px",
                  fontFamily: "var(--font-mono, monospace)",
                  fontSize: "12px",
                  color: "rgba(38,37,30,0.45)",
                  whiteSpace: "nowrap",
                }}
              >
                {key.lastUsed ?? "Never"}
              </td>
              <td style={{ padding: "12px 12px", textAlign: "right" }}>
                {!key.revoked && onRevoke && (
                  <button
                    onClick={() => onRevoke(key.id)}
                    style={{
                      background: "none",
                      border: "none",
                      cursor: "pointer",
                      fontFamily: "system-ui",
                      fontSize: "12px",
                      color: "rgba(38,37,30,0.4)",
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLElement).style.color = "#cf2d56";
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLElement).style.color = "rgba(38,37,30,0.4)";
                    }}
                  >
                    Revoke
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
