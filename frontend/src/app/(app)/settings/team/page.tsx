"use client";

import { useState } from "react";
import { X, UserPlus } from "lucide-react";
import { useAuthStore } from "@/lib/stores/authStore";

// ── Types ─────────────────────────────────────────────────────────────────────

type Role = "owner" | "admin" | "member";

interface TeamMember {
  id: string;
  name: string;
  email: string;
  role: Role;
  joinedAt: string;
}

// ── Mock data ─────────────────────────────────────────────────────────────────

const MOCK_MEMBERS: TeamMember[] = [
  {
    id: "m1",
    name: "Ambrish Nigam",
    email: "ambrish.nigam@gmail.com",
    role: "owner",
    joinedAt: "2026-01-15",
  },
];

// ── Role badge ────────────────────────────────────────────────────────────────

const ROLE_STYLE: Record<Role, { bg: string; color: string }> = {
  owner: { bg: "rgba(192,133,50,0.12)", color: "#c08532" },
  admin: { bg: "rgba(83,58,253,0.1)", color: "#533afd" },
  member: { bg: "rgba(38,37,30,0.08)", color: "rgba(38,37,30,0.6)" },
};

function RoleBadge({ role }: { role: Role }) {
  const s = ROLE_STYLE[role];
  return (
    <span
      className="px-2 py-0.5 rounded-[4px] capitalize"
      style={{
        background: s.bg,
        color: s.color,
        fontFamily: "system-ui",
        fontSize: "11px",
        fontWeight: 500,
      }}
    >
      {role}
    </span>
  );
}

// ── Invite modal ──────────────────────────────────────────────────────────────

function InviteModal({
  onClose,
  onInvite,
}: {
  onClose: () => void;
  onInvite: (email: string, role: Role) => void;
}) {
  const [email, setEmail] = useState("");
  const [role, setRole] = useState<Role>("member");
  const [sending, setSending] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!email.trim()) return;
    setSending(true);
    await new Promise((r) => setTimeout(r, 700));
    onInvite(email.trim(), role);
    setSending(false);
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(38,37,30,0.4)" }}
      onClick={onClose}
    >
      <div
        className="w-full max-w-[420px] rounded-[8px] p-6"
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
            Invite member
          </h2>
          <button
            onClick={onClose}
            style={{ background: "none", border: "none", cursor: "pointer", color: "rgba(38,37,30,0.4)" }}
          >
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
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
              Email address
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder="colleague@company.com"
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
                marginBottom: "6px",
              }}
            >
              Role
            </label>
            <select
              value={role}
              onChange={(e) => setRole(e.target.value as Role)}
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
                cursor: "pointer",
              }}
            >
              <option value="member">Member</option>
              <option value="admin">Admin</option>
            </select>
          </div>

          <button
            type="submit"
            disabled={sending || !email.trim()}
            className="w-full transition-opacity hover:opacity-90"
            style={{
              background: "#f54e00",
              color: "#ffffff",
              border: "none",
              borderRadius: "8px",
              padding: "10px",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "14px",
              cursor: sending ? "not-allowed" : "pointer",
              opacity: !email.trim() ? 0.5 : 1,
            }}
          >
            {sending ? "Sending invite…" : "Send invitation"}
          </button>
        </form>
      </div>
    </div>
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function TeamPage() {
  const user = useAuthStore((s) => s.user);
  const isPlanSufficient = user?.plan === "enterprise";

  const [hasTeam, setHasTeam] = useState(true);
  const [teamName, setTeamName] = useState("SynthFlow Team");
  const [nameInput, setNameInput] = useState("");
  const [members, setMembers] = useState<TeamMember[]>(MOCK_MEMBERS);
  const [showInvite, setShowInvite] = useState(false);

  function handleCreateTeam(e: React.FormEvent) {
    e.preventDefault();
    if (!nameInput.trim()) return;
    setTeamName(nameInput.trim());
    setHasTeam(true);
  }

  function handleInvite(email: string, role: Role) {
    const newMember: TeamMember = {
      id: crypto.randomUUID(),
      name: email.split("@")[0],
      email,
      role,
      joinedAt: new Date().toISOString().slice(0, 10),
    };
    setMembers((prev) => [...prev, newMember]);
    setShowInvite(false);
  }

  function handleRemove(id: string) {
    setMembers((prev) => prev.filter((m) => m.id !== id));
  }

  if (!isPlanSufficient && user?.plan !== "pro") {
    return (
      <div className="max-w-[560px]">
        <div
          className="rounded-[8px] p-6"
          style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
        >
          <p
            className="mb-2"
            style={{
              fontFamily: "system-ui",
              fontSize: "11px",
              fontWeight: 600,
              textTransform: "uppercase",
              letterSpacing: "0.048px",
              color: "#c08532",
            }}
          >
            Business Plan feature
          </p>
          <p
            style={{
              fontFamily: "var(--font-serif, Georgia, serif)",
              fontSize: "16px",
              color: "rgba(38,37,30,0.6)",
              lineHeight: 1.5,
              marginBottom: "16px",
            }}
          >
            Team collaboration is available on the Business plan. Upgrade to invite colleagues
            and manage roles.
          </p>
          <a
            href="/app/settings/billing"
            style={{
              display: "inline-block",
              background: "#f54e00",
              color: "#ffffff",
              borderRadius: "8px",
              padding: "10px 20px",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "13px",
              textDecoration: "none",
            }}
          >
            View plans
          </a>
        </div>
      </div>
    );
  }

  if (!hasTeam) {
    return (
      <div className="max-w-[400px]">
        <div
          className="rounded-[8px] p-6"
          style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
        >
          <h2
            className="mb-2"
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "16px",
              fontWeight: 400,
              color: "#26251e",
            }}
          >
            Create a team
          </h2>
          <p
            className="mb-5"
            style={{
              fontFamily: "system-ui",
              fontSize: "13px",
              color: "rgba(38,37,30,0.5)",
              lineHeight: 1.5,
            }}
          >
            Teams let you collaborate on datasets and share generation history.
          </p>
          <form onSubmit={handleCreateTeam} className="flex gap-2">
            <input
              value={nameInput}
              onChange={(e) => setNameInput(e.target.value)}
              placeholder="Acme Corp"
              required
              style={{
                flex: 1,
                background: "transparent",
                border: "1px solid rgba(38,37,30,0.12)",
                borderRadius: "8px",
                padding: "9px 12px",
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "14px",
                color: "#26251e",
                outline: "none",
              }}
            />
            <button
              type="submit"
              style={{
                background: "#f54e00",
                color: "#ffffff",
                border: "none",
                borderRadius: "8px",
                padding: "9px 16px",
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "13px",
                cursor: "pointer",
                whiteSpace: "nowrap",
              }}
            >
              Create
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-[680px]">
      {/* Team header */}
      <div
        className="rounded-[8px] p-5 mb-4 flex items-center justify-between"
        style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
      >
        <div>
          <p
            style={{
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "16px",
              fontWeight: 400,
              color: "#26251e",
            }}
          >
            {teamName}
          </p>
          <p
            style={{
              fontFamily: "system-ui",
              fontSize: "12px",
              color: "rgba(38,37,30,0.45)",
              marginTop: "2px",
            }}
          >
            {members.length} {members.length === 1 ? "member" : "members"}
          </p>
        </div>
        <button
          onClick={() => setShowInvite(true)}
          className="flex items-center gap-1.5 transition-opacity hover:opacity-90"
          style={{
            background: "#f54e00",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            padding: "9px 14px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            cursor: "pointer",
          }}
        >
          <UserPlus size={14} strokeWidth={1.5} />
          Invite member
        </button>
      </div>

      {/* Members table */}
      <div
        className="rounded-[8px] overflow-hidden"
        style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr
              style={{
                background: "#f7f7f4",
                borderBottom: "1px solid rgba(38,37,30,0.1)",
              }}
            >
              {["Member", "Role", "Joined", ""].map((h) => (
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
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {members.map((m, i) => (
              <tr
                key={m.id}
                style={{
                  borderBottom:
                    i < members.length - 1
                      ? "1px solid rgba(38,37,30,0.06)"
                      : "none",
                }}
              >
                <td style={{ padding: "12px 14px" }}>
                  <p
                    style={{
                      fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                      fontSize: "14px",
                      color: "#26251e",
                    }}
                  >
                    {m.name}
                  </p>
                  <p
                    style={{
                      fontFamily: "system-ui",
                      fontSize: "11px",
                      color: "rgba(38,37,30,0.45)",
                      marginTop: "2px",
                    }}
                  >
                    {m.email}
                  </p>
                </td>
                <td style={{ padding: "12px 14px" }}>
                  <RoleBadge role={m.role} />
                </td>
                <td
                  style={{
                    padding: "12px 14px",
                    fontFamily: "var(--font-mono, monospace)",
                    fontSize: "12px",
                    color: "rgba(38,37,30,0.45)",
                  }}
                >
                  {m.joinedAt}
                </td>
                <td style={{ padding: "12px 14px", textAlign: "right" }}>
                  {m.role !== "owner" && (
                    <button
                      onClick={() => handleRemove(m.id)}
                      style={{
                        background: "none",
                        border: "none",
                        cursor: "pointer",
                        fontFamily: "system-ui",
                        fontSize: "12px",
                        color: "rgba(38,37,30,0.4)",
                        transition: "color 150ms ease",
                      }}
                      onMouseEnter={(e) => {
                        (e.currentTarget as HTMLElement).style.color = "#cf2d56";
                      }}
                      onMouseLeave={(e) => {
                        (e.currentTarget as HTMLElement).style.color = "rgba(38,37,30,0.4)";
                      }}
                    >
                      Remove
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {showInvite && (
        <InviteModal onClose={() => setShowInvite(false)} onInvite={handleInvite} />
      )}
    </div>
  );
}
