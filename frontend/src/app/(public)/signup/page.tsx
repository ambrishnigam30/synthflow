"use client";

import Link from "next/link";
import { useState } from "react";

export default function SignupPage() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (password.length < 8) {
      setError("Password must be at least 8 characters.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}/api/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_name: fullName, email, password }),
      });
      const body = await res.json();
      if (!res.ok) {
        if (res.status === 409) {
          setError("This email is already registered. Try logging in.");
        } else {
          setError(body?.detail?.message ?? "Signup failed. Please try again.");
        }
        return;
      }
      const { access_token, refresh_token } = body.data;
      localStorage.setItem("sf_access", access_token);
      localStorage.setItem("sf_refresh", refresh_token);
      // Set cookie so middleware can verify auth on protected routes (15 min)
      document.cookie = `sf_access=${access_token}; path=/; max-age=900; SameSite=Strict`;
      window.location.href = "/dashboard";
    } catch {
      setError("Could not reach the server. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-[calc(100vh-56px)] flex items-center justify-center px-4 py-16"
      style={{ background: "#ffffff" }}>
      <div className="w-full max-w-[420px]">

        {/* Logo */}
        <div className="flex items-center gap-2.5 mb-10 justify-center">
          <div className="w-8 h-8 rounded-[5px] flex items-center justify-center" style={{ background: "#3d4043" }}>
            <span className="text-white text-[16px]" style={{ fontWeight: 300 }}>S</span>
          </div>
          <span className="text-[16px] font-[400]" style={{ color: "#061b31" }}>SynthFlow</span>
        </div>

        <div className="rounded-[6px] border border-[#e5edf5] p-8"
          style={{ boxShadow: "rgba(50,50,93,0.25) 0px 30px 45px -30px, rgba(0,0,0,0.1) 0px 18px 36px -18px" }}>

          <h1 className="mb-2 text-center" style={{ fontSize: "22px", fontWeight: 300, letterSpacing: "-0.22px", color: "#061b31" }}>
            Create your account
          </h1>
          <p className="mb-8 text-center text-[14px] font-[300]" style={{ color: "#64748d" }}>
            Free forever. No credit card required.
          </p>

          {/* Google OAuth */}
          <button
            type="button"
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-[4px] border border-[#e5edf5] text-[14px] font-[400] mb-5 transition-colors hover:bg-[#f8fafc]"
            style={{ color: "#061b31", background: "#fff" }}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" fill="#4285F4"/>
              <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
              <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l3.66-2.84z" fill="#FBBC05"/>
              <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
            </svg>
            Continue with Google
          </button>

          <div className="flex items-center gap-3 mb-5">
            <div className="flex-1 h-px" style={{ background: "#e5edf5" }} />
            <span className="text-[12px] font-[300]" style={{ color: "#64748d" }}>or</span>
            <div className="flex-1 h-px" style={{ background: "#e5edf5" }} />
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-[14px] font-[400] mb-1.5" style={{ color: "#273951" }}>Full name</label>
              <input
                type="text"
                value={fullName}
                onChange={e => setFullName(e.target.value)}
                required
                autoComplete="name"
                placeholder="Ambrish Nigam"
                className="w-full border border-[#e5edf5] rounded-[4px] px-3 py-2 text-[14px] font-[300] bg-white placeholder:text-[#64748d] focus:outline-none focus:border-[#533afd] transition-colors"
                style={{ color: "#061b31" }}
              />
            </div>

            <div>
              <label className="block text-[14px] font-[400] mb-1.5" style={{ color: "#273951" }}>Work email</label>
              <input
                type="email"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
                placeholder="you@company.com"
                className="w-full border border-[#e5edf5] rounded-[4px] px-3 py-2 text-[14px] font-[300] bg-white placeholder:text-[#64748d] focus:outline-none focus:border-[#533afd] transition-colors"
                style={{ color: "#061b31" }}
              />
            </div>

            <div>
              <label className="block text-[14px] font-[400] mb-1.5" style={{ color: "#273951" }}>Password</label>
              <input
                type="password"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                minLength={8}
                autoComplete="new-password"
                placeholder="At least 8 characters"
                className="w-full border border-[#e5edf5] rounded-[4px] px-3 py-2 text-[14px] font-[300] bg-white placeholder:text-[#64748d] focus:outline-none focus:border-[#533afd] transition-colors"
                style={{ color: "#061b31" }}
              />
            </div>

            {error && (
              <p className="text-[12px] font-[300] text-[#ea2261]">{error}</p>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-[4px] text-[15px] font-[400] transition-colors disabled:opacity-60 mt-2"
              style={{ background: loading ? "#4434d4" : "#533afd", color: "#fff" }}
            >
              {loading ? "Creating account…" : "Create free account"}
            </button>
          </form>

          <p className="mt-5 text-center text-[12px] font-[300]" style={{ color: "#64748d" }}>
            By signing up you agree to our{" "}
            <a href="#" className="underline">Terms of Service</a>
            {" "}and{" "}
            <a href="#" className="underline">Privacy Policy</a>.
          </p>
        </div>

        <p className="mt-6 text-center text-[14px] font-[300]" style={{ color: "#64748d" }}>
          Already have an account?{" "}
          <Link href="/login" className="font-[400] hover:underline" style={{ color: "#533afd" }}>
            Sign in
          </Link>
        </p>
      </div>
    </div>
  );
}
