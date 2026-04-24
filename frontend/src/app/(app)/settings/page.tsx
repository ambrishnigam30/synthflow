"use client";

import { useState, useRef, useEffect } from "react";
import { useAuthStore } from "@/lib/stores/authStore";
import { authApi } from "@/lib/api";

// ── Helpers ───────────────────────────────────────────────────────────────────

function SectionCard({ children }: { children: React.ReactNode }) {
  return (
    <div
      className="rounded-[8px] p-6 mb-4"
      style={{ background: "#ffffff", border: "1px solid rgba(38,37,30,0.1)" }}
    >
      {children}
    </div>
  );
}

function FieldLabel({ children }: { children: React.ReactNode }) {
  return (
    <label
      style={{
        fontFamily: "system-ui",
        fontSize: "11px",
        fontWeight: 600,
        textTransform: "uppercase",
        letterSpacing: "0.048px",
        color: "rgba(38,37,30,0.5)",
        display: "block",
        marginBottom: "6px",
      }}
    >
      {children}
    </label>
  );
}

function AppInput({
  value,
  onChange,
  readOnly,
  type = "text",
  placeholder,
}: {
  value: string;
  onChange?: (v: string) => void;
  readOnly?: boolean;
  type?: string;
  placeholder?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      onChange={(e) => onChange?.(e.target.value)}
      readOnly={readOnly}
      placeholder={placeholder}
      style={{
        width: "100%",
        background: readOnly ? "#f7f7f4" : "transparent",
        border: "1px solid rgba(38,37,30,0.12)",
        borderRadius: "8px",
        padding: "9px 12px",
        fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
        fontSize: "14px",
        color: readOnly ? "rgba(38,37,30,0.5)" : "#26251e",
        outline: "none",
        cursor: readOnly ? "not-allowed" : "text",
        boxSizing: "border-box",
      }}
      onFocus={(e) => {
        if (!readOnly) e.target.style.borderColor = "rgba(38,37,30,0.25)";
      }}
      onBlur={(e) => {
        e.target.style.borderColor = "rgba(38,37,30,0.12)";
      }}
    />
  );
}

// ── Page ─────────────────────────────────────────────────────────────────────

export default function ProfileSettingsPage() {
  const { user, setUser } = useAuthStore();
  // Populate from authStore name on mount; re-sync when user changes
  const [name, setName] = useState(user?.name ?? "");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Keep input in sync if authStore user changes (e.g. after initial hydration)
  useEffect(() => {
    setName(user?.name ?? "");
  }, [user?.name]);

  const fileRef = useRef<HTMLInputElement>(null);
  const [avatarPreview, setAvatarPreview] = useState<string | null>(user?.avatarUrl ?? null);
  const [avatarFile, setAvatarFile] = useState<File | null>(null);

  // Keep avatar preview in sync with authStore
  useEffect(() => {
    if (user?.avatarUrl) setAvatarPreview(user.avatarUrl);
  }, [user?.avatarUrl]);

  const displayName = name.trim();
  const initials = displayName
    ? displayName.split(" ").map((w) => w[0]).slice(0, 2).join("").toUpperCase()
    : (user?.email?.[0]?.toUpperCase() ?? "U");

  function handleAvatarChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setAvatarFile(file);
    const url = URL.createObjectURL(file);
    setAvatarPreview(url);
    e.target.value = "";
  }

  async function handleSave() {
    setSaving(true);
    setSaveError(null);
    try {
      // Build update payload
      const payload: { full_name?: string; avatar_url?: string } = {};
      if (name.trim() !== (user?.name ?? "")) payload.full_name = name.trim();

      // TODO: upload avatarFile to Supabase Storage and get URL, then set avatar_url
      // For now, skip avatar upload if no Supabase is configured
      void avatarFile;

      const updated = await authApi.updateMe(payload);
      // Sync authStore — backend returns full_name, map to store's name field
      setUser({
        id: updated.id,
        email: updated.email,
        name: updated.full_name ?? "",
        plan: updated.plan as "free" | "starter" | "pro" | "enterprise",
        avatarUrl: updated.avatar_url ?? undefined,
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  // ── Password change state ────────────────────────────────────────────────────

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [pwSaving, setPwSaving] = useState(false);
  const [pwSaved, setPwSaved] = useState(false);
  const [pwError, setPwError] = useState<string | null>(null);

  async function handleChangePassword() {
    setPwError(null);
    if (newPassword.length < 8) {
      setPwError("New password must be at least 8 characters.");
      return;
    }
    if (newPassword !== confirmPassword) {
      setPwError("Passwords do not match.");
      return;
    }
    setPwSaving(true);
    try {
      await authApi.changePassword(currentPassword, newPassword);
      setPwSaved(true);
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setTimeout(() => setPwSaved(false), 3000);
    } catch (err) {
      setPwError(err instanceof Error ? err.message : "Password change failed.");
    } finally {
      setPwSaving(false);
    }
  }

  // ── Delete account state ─────────────────────────────────────────────────────

  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteInput, setDeleteInput] = useState("");

  return (
    <div className="max-w-[560px]">
      {/* ── Profile section ────────────────────────────────────────────────── */}
      <SectionCard>
        <h2
          className="mb-5"
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "16px",
            fontWeight: 400,
            color: "#26251e",
          }}
        >
          Profile
        </h2>

        {/* Avatar */}
        <div className="flex items-center gap-5 mb-6">
          <div className="relative">
            <div
              className="w-20 h-20 rounded-full flex items-center justify-center overflow-hidden cursor-pointer"
              style={{ background: "#ebeae5", border: "2px solid rgba(38,37,30,0.1)" }}
              onClick={() => fileRef.current?.click()}
            >
              {avatarPreview ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={avatarPreview} alt="Avatar" className="w-full h-full object-cover" />
              ) : (
                <span
                  style={{
                    fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                    fontSize: "24px",
                    fontWeight: 400,
                    color: "#26251e",
                  }}
                >
                  {initials}
                </span>
              )}
            </div>
            <div
              className="absolute bottom-0 right-0 w-6 h-6 rounded-full flex items-center justify-center cursor-pointer"
              style={{ background: "#f54e00", border: "2px solid #f2f1ed" }}
              onClick={() => fileRef.current?.click()}
            >
              <span style={{ color: "#ffffff", fontSize: "12px", lineHeight: 1 }}>+</span>
            </div>
          </div>
          <div>
            <p
              style={{
                fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
                fontSize: "13px",
                color: "#26251e",
                marginBottom: "4px",
              }}
            >
              Profile photo
            </p>
            <p
              style={{
                fontFamily: "system-ui",
                fontSize: "12px",
                color: "rgba(38,37,30,0.45)",
              }}
            >
              JPG, PNG or GIF · max 2 MB
            </p>
            <button
              onClick={() => fileRef.current?.click()}
              className="mt-2 transition-all"
              style={{
                background: "#ebeae5",
                border: "none",
                borderRadius: "6px",
                padding: "5px 10px",
                fontFamily: "system-ui",
                fontSize: "12px",
                color: "#26251e",
                cursor: "pointer",
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.color = "#cf2d56"; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.color = "#26251e"; }}
            >
              Change photo
            </button>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            onChange={handleAvatarChange}
            style={{ display: "none" }}
          />
        </div>

        {/* Name */}
        <div className="mb-4">
          <FieldLabel>Full name</FieldLabel>
          <AppInput value={name} onChange={setName} placeholder="Your name" />
        </div>

        {/* Email (read-only) */}
        <div className="mb-6">
          <FieldLabel>Email address</FieldLabel>
          <AppInput value={user?.email ?? ""} readOnly />
          <p
            className="mt-1.5"
            style={{ fontFamily: "system-ui", fontSize: "11px", color: "rgba(38,37,30,0.4)" }}
          >
            To change your email, contact support.
          </p>
        </div>

        {saveError && (
          <p
            className="mb-3"
            style={{ fontFamily: "system-ui", fontSize: "12px", color: "#cf2d56" }}
          >
            {saveError}
          </p>
        )}

        {/* Save */}
        <button
          onClick={handleSave}
          disabled={saving}
          className="transition-opacity hover:opacity-90"
          style={{
            background: saving ? "rgba(38,37,30,0.4)" : "#f54e00",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            padding: "10px 20px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            fontWeight: 400,
            cursor: saving ? "not-allowed" : "pointer",
          }}
        >
          {saving ? "Saving…" : saved ? "Saved ✓" : "Save changes"}
        </button>
      </SectionCard>

      {/* ── Change password section ─────────────────────────────────────────── */}
      <SectionCard>
        <h2
          className="mb-1"
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "16px",
            fontWeight: 400,
            color: "#26251e",
          }}
        >
          Change password
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
          Not available for accounts signed in with Google.
        </p>

        <div className="mb-4">
          <FieldLabel>Current password</FieldLabel>
          <AppInput
            type="password"
            value={currentPassword}
            onChange={setCurrentPassword}
            placeholder="Enter current password"
          />
        </div>
        <div className="mb-4">
          <FieldLabel>New password</FieldLabel>
          <AppInput
            type="password"
            value={newPassword}
            onChange={setNewPassword}
            placeholder="At least 8 characters"
          />
        </div>
        <div className="mb-5">
          <FieldLabel>Confirm new password</FieldLabel>
          <AppInput
            type="password"
            value={confirmPassword}
            onChange={setConfirmPassword}
            placeholder="Repeat new password"
          />
        </div>

        {pwError && (
          <p
            className="mb-3"
            style={{ fontFamily: "system-ui", fontSize: "12px", color: "#cf2d56" }}
          >
            {pwError}
          </p>
        )}

        <button
          onClick={handleChangePassword}
          disabled={pwSaving || !currentPassword || !newPassword || !confirmPassword}
          className="transition-opacity hover:opacity-90"
          style={{
            background:
              pwSaving || !currentPassword || !newPassword || !confirmPassword
                ? "rgba(38,37,30,0.2)"
                : "#26251e",
            color: "#ffffff",
            border: "none",
            borderRadius: "8px",
            padding: "10px 20px",
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "13px",
            fontWeight: 400,
            cursor:
              pwSaving || !currentPassword || !newPassword || !confirmPassword
                ? "not-allowed"
                : "pointer",
          }}
        >
          {pwSaving ? "Updating…" : pwSaved ? "Password updated ✓" : "Update password"}
        </button>
      </SectionCard>

      {/* ── Danger zone ─────────────────────────────────────────────────────── */}
      <SectionCard>
        <h2
          className="mb-1"
          style={{
            fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
            fontSize: "16px",
            fontWeight: 400,
            color: "#cf2d56",
          }}
        >
          Danger zone
        </h2>
        <p
          className="mb-4"
          style={{
            fontFamily: "system-ui",
            fontSize: "13px",
            color: "rgba(38,37,30,0.5)",
            lineHeight: 1.5,
          }}
        >
          Permanently delete your account and all associated data. This action cannot be undone.
        </p>

        {showDeleteConfirm ? (
          <div
            className="rounded-[8px] p-4 mb-3"
            style={{ background: "rgba(207,45,86,0.05)", border: "1px solid rgba(207,45,86,0.15)" }}
          >
            <p
              className="mb-3"
              style={{ fontFamily: "system-ui", fontSize: "13px", color: "#cf2d56", fontWeight: 500 }}
            >
              Are you sure? Type DELETE to confirm.
            </p>
            <input
              value={deleteInput}
              onChange={(e) => setDeleteInput(e.target.value)}
              placeholder="DELETE"
              style={{
                background: "transparent",
                border: "1px solid rgba(207,45,86,0.3)",
                borderRadius: "6px",
                padding: "7px 10px",
                fontFamily: "var(--font-mono, monospace)",
                fontSize: "13px",
                color: "#26251e",
                outline: "none",
                width: "100%",
                marginBottom: "10px",
                boxSizing: "border-box",
              }}
            />
            <div className="flex gap-2">
              <button
                onClick={() => { setShowDeleteConfirm(false); setDeleteInput(""); }}
                style={{
                  background: "#ebeae5",
                  border: "none",
                  borderRadius: "6px",
                  padding: "7px 14px",
                  fontFamily: "system-ui",
                  fontSize: "12px",
                  color: "#26251e",
                  cursor: "pointer",
                }}
              >
                Cancel
              </button>
              <button
                disabled={deleteInput !== "DELETE"}
                style={{
                  background: deleteInput === "DELETE" ? "#cf2d56" : "rgba(207,45,86,0.3)",
                  border: "none",
                  borderRadius: "6px",
                  padding: "7px 14px",
                  fontFamily: "system-ui",
                  fontSize: "12px",
                  color: "#ffffff",
                  cursor: deleteInput === "DELETE" ? "pointer" : "not-allowed",
                }}
              >
                Delete my account
              </button>
            </div>
          </div>
        ) : (
          <button
            onClick={() => setShowDeleteConfirm(true)}
            className="transition-all"
            style={{
              background: "transparent",
              border: "1px solid rgba(207,45,86,0.3)",
              borderRadius: "8px",
              padding: "9px 16px",
              fontFamily: "var(--font-satoshi, system-ui, sans-serif)",
              fontSize: "13px",
              color: "#cf2d56",
              cursor: "pointer",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.background = "rgba(207,45,86,0.05)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.background = "transparent";
            }}
          >
            Delete account
          </button>
        )}
      </SectionCard>
    </div>
  );
}
