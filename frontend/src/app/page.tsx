import Link from "next/link";

/* ── Marketing Nav ──────────────────────────────────────────────────────── */
function Nav() {
  return (
    <header
      className="sticky top-0 z-50 w-full"
      style={{ background: "rgba(255,255,255,0.92)", backdropFilter: "blur(12px)" }}
    >
      <div className="mx-auto max-w-[1080px] px-6 h-14 flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-[4px] flex items-center justify-center" style={{ backgroundColor: "#3d4043" }}>
            <span className="text-white text-sm" style={{ fontWeight: 300 }}>S</span>
          </div>
          <span className="text-sm font-[400]" style={{ color: "#061b31" }}>SynthFlow</span>
        </Link>

        <nav className="hidden md:flex items-center gap-6">
          {[{ href: "#how-it-works", label: "Product" }, { href: "/pricing", label: "Pricing" }, { href: "#", label: "Docs" }].map(({ href, label }) => (
            <Link key={label} href={href} className="text-sm font-[400] hover:text-[#533afd] transition-colors" style={{ color: "#061b31" }}>
              {label}
            </Link>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <Link href="/login" className="text-sm font-[400] hidden sm:block" style={{ color: "#061b31" }}>Log in</Link>
          <Link href="/signup" className="text-sm font-[400] px-4 py-1.5 rounded-[4px]" style={{ background: "#533afd", color: "#fff" }}>
            Get Started Free
          </Link>
        </div>
      </div>
    </header>
  );
}

/* ── Hero ───────────────────────────────────────────────────────────────── */
function Hero() {
  return (
    <section className="mx-auto max-w-[1080px] px-6 pt-24 pb-20 text-center">
      <div className="inline-flex items-center gap-2 px-3 py-1 rounded-[4px] mb-8 border border-[#d6d9fc] bg-[rgba(83,58,253,0.04)]">
        <span className="w-1.5 h-1.5 rounded-full bg-[#533afd]" />
        <span className="text-[12px] font-[400]" style={{ color: "#533afd" }}>Now in public beta — free forever</span>
      </div>

      <h1
        className="mb-6 mx-auto max-w-[780px]"
        style={{ fontSize: "56px", fontWeight: 300, lineHeight: 1.03, letterSpacing: "-1.4px", color: "#061b31" }}
      >
        Data that understands the real world
      </h1>

      <p
        className="mb-10 mx-auto max-w-[520px]"
        style={{ fontSize: "18px", fontWeight: 300, lineHeight: 1.4, color: "#64748d" }}
      >
        Describe your dataset in plain English. SynthFlow generates causally realistic, statistically accurate, privacy-safe synthetic data — in seconds.
      </p>

      <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
        <Link
          href="/signup"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-[4px] text-[16px] font-[400]"
          style={{ background: "#533afd", color: "#fff" }}
        >
          Get Started Free
        </Link>
        <Link
          href="#how-it-works"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-[4px] text-[16px] font-[400] border border-[#b9b9f9]"
          style={{ color: "#533afd" }}
        >
          See how it works →
        </Link>
      </div>

      {/* Demo card */}
      <div className="mt-16 mx-auto max-w-[680px] rounded-[6px] border border-[#e5edf5] overflow-hidden text-left"
        style={{ boxShadow: "rgba(50,50,93,0.25) 0px 30px 45px -30px, rgba(0,0,0,0.1) 0px 18px 36px -18px" }}>
        <div className="px-4 py-3 border-b border-[#e5edf5] flex items-center gap-2" style={{ background: "#f8fafc" }}>
          <div className="flex gap-1.5">
            <span className="w-3 h-3 rounded-full bg-[#ea2261] opacity-70" />
            <span className="w-3 h-3 rounded-full bg-[#f59e0b] opacity-70" />
            <span className="w-3 h-3 rounded-full bg-[#15be53] opacity-70" />
          </div>
          <span className="text-[12px] font-[400]" style={{ color: "#64748d" }}>
            Generate 5000 Indian healthcare records with causal age-BMI correlation
          </span>
        </div>
        <div className="p-4" style={{ background: "#ffffff" }}>
          <div className="space-y-1.5">
            {[
              { phase: "Intent Parsing", status: "done", color: "#108c3d" },
              { phase: "Knowledge Graph", status: "done", color: "#108c3d" },
              { phase: "Schema Design (15 columns)", status: "done", color: "#108c3d" },
              { phase: "Constraint Physics", status: "active", color: "#533afd" },
              { phase: "Statistical Modeling", status: "pending", color: "#64748d" },
              { phase: "Glass Box Code", status: "pending", color: "#64748d" },
            ].map(({ phase, status, color }) => (
              <div key={phase} className="flex items-center gap-3 py-0.5">
                <span className="text-[12px]" style={{ color, minWidth: "12px" }}>
                  {status === "done" ? "✓" : status === "active" ? "●" : "○"}
                </span>
                <span className="text-[13px] font-[400]" style={{ color: status === "pending" ? "#64748d" : "#061b31" }}>
                  {phase}
                </span>
                {status === "active" && (
                  <span className="ml-auto text-[11px] font-[400] px-2 py-0.5 rounded-[4px] border border-[#d6d9fc]" style={{ color: "#533afd", background: "rgba(83,58,253,0.05)" }}>
                    generating…
                  </span>
                )}
              </div>
            ))}
          </div>
          <div className="mt-4 pt-3 border-t border-[#e5edf5] flex items-center justify-between">
            <span className="text-[12px] font-[300]" style={{ color: "#64748d" }}>Quality score</span>
            <span className="text-[18px] font-[300]" style={{ color: "#108c3d" }}>94.7</span>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── How It Works ───────────────────────────────────────────────────────── */
function HowItWorks() {
  const steps = [
    {
      num: "01",
      title: "Understand",
      body: "SynthFlow's intent engine parses your natural-language description, infers domain, region, schema — no SQL, no configuration files.",
    },
    {
      num: "02",
      title: "Architect",
      body: "A causal knowledge graph designs realistic column relationships. Age correlates with salary. Admission before discharge. Always.",
    },
    {
      num: "03",
      title: "Generate",
      body: "The Glass Box Code Synthesizer writes a deterministic, auditable Python function. You can read, modify, and reproduce every output.",
    },
  ];

  return (
    <section id="how-it-works" className="py-24" style={{ background: "#ffffff" }}>
      <div className="mx-auto max-w-[1080px] px-6">
        <div className="text-center mb-14">
          <p className="text-[12px] font-[400] mb-3 tracking-widest uppercase" style={{ color: "#533afd" }}>How it works</p>
          <h2 style={{ fontSize: "32px", fontWeight: 300, lineHeight: 1.1, letterSpacing: "-0.64px", color: "#061b31" }}>
            From prompt to production-ready data
          </h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {steps.map(({ num, title, body }) => (
            <div key={num} className="rounded-[6px] border border-[#e5edf5] p-6"
              style={{ boxShadow: "rgba(50,50,93,0.15) 0px 15px 35px -20px" }}>
              <div className="text-[12px] font-[400] mb-4 tabular-nums" style={{ color: "#533afd" }}>{num}</div>
              <h3 style={{ fontSize: "22px", fontWeight: 300, lineHeight: 1.1, letterSpacing: "-0.22px", color: "#061b31", marginBottom: "12px" }}>
                {title}
              </h3>
              <p style={{ fontSize: "15px", fontWeight: 300, lineHeight: 1.5, color: "#64748d" }}>{body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ── Five Pillars ───────────────────────────────────────────────────────── */
function FivePillars() {
  const pillars = [
    { icon: "◎", title: "Zero Hardcoding", body: "No city lists. No salary tables. The LLM generates all domain knowledge — every domain, every region, from first principles." },
    { icon: "⟁", title: "Causal Realism", body: "A directed acyclic graph enforces real-world constraints. Discharge after admission. Income correlates with postal code." },
    { icon: "≈", title: "Distributional Accuracy", body: "Salary is log-normal. Age is truncated normal. Statistical distributions are validated semantically, not by guesswork." },
    { icon: "⏱", title: "Temporal Realism", body: "Diwali revenue spikes. Monday morning hospital admissions surge. Weekly and seasonal patterns are baked in." },
    { icon: "∿", title: "Dirty Data", body: "Real data has typos, mixed date formats, nulls, and near-duplicates. SynthFlow injects them at configurable rates." },
  ];

  return (
    <section className="py-24" style={{ background: "#1c1e54" }}>
      <div className="mx-auto max-w-[1080px] px-6">
        <div className="text-center mb-14">
          <p className="text-[12px] font-[400] mb-3 tracking-widest uppercase" style={{ color: "#b9b9f9" }}>Five Pillars</p>
          <h2 style={{ fontSize: "32px", fontWeight: 300, lineHeight: 1.1, letterSpacing: "-0.64px", color: "#ffffff" }}>
            What makes SynthFlow different
          </h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {pillars.map(({ icon, title, body }) => (
            <div key={title} className="rounded-[6px] border p-5" style={{ borderColor: "rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.04)" }}>
              <div className="text-[22px] mb-3" style={{ color: "#b9b9f9" }}>{icon}</div>
              <h3 className="mb-2" style={{ fontSize: "16px", fontWeight: 400, color: "#ffffff" }}>{title}</h3>
              <p style={{ fontSize: "13px", fontWeight: 300, lineHeight: 1.5, color: "rgba(255,255,255,0.6)" }}>{body}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ── Domain Showcase ────────────────────────────────────────────────────── */
function DomainShowcase() {
  const domains = [
    "Healthcare", "Finance", "E-Commerce", "Real Estate", "Logistics",
    "Education", "HR / Payroll", "Manufacturing", "Retail", "Insurance",
  ];

  return (
    <section className="py-20 overflow-hidden" style={{ background: "#ffffff" }}>
      <div className="mx-auto max-w-[1080px] px-6 mb-8 text-center">
        <h2 style={{ fontSize: "26px", fontWeight: 300, letterSpacing: "-0.26px", color: "#061b31" }}>
          Works for any domain, any region
        </h2>
        <p className="mt-3 text-[15px] font-[300]" style={{ color: "#64748d" }}>
          Zero configuration. Just describe your data.
        </p>
      </div>
      <div className="flex gap-3 overflow-x-auto pb-2 px-6 no-scrollbar">
        {domains.map(d => (
          <span
            key={d}
            className="inline-flex items-center px-4 py-2 rounded-[4px] border border-[#e5edf5] whitespace-nowrap text-[14px] font-[300] flex-shrink-0"
            style={{ color: "#273951" }}
          >
            {d}
          </span>
        ))}
      </div>
    </section>
  );
}

/* ── Glass Box Section ──────────────────────────────────────────────────── */
function GlassBox() {
  const code = `def generate(row_count: int, seed: int) -> pd.DataFrame:
    """
    Deterministic, auditable, zero-API. Read it. Modify it.
    Reproduce it exactly with the same seed.
    """
    rng = np.random.default_rng(seed)
    ages = rng.normal(42, 12, row_count).clip(18, 90).astype(int)
    # BMI correlates with age via causal DAG rule
    bmis = 22 + (ages - 40) * 0.06 + rng.normal(0, 2.5, row_count)
    ...`;

  return (
    <section className="py-24" style={{ background: "#1c1e54" }}>
      <div className="mx-auto max-w-[1080px] px-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          <div>
            <p className="text-[12px] font-[400] mb-4 tracking-widest uppercase" style={{ color: "#b9b9f9" }}>Glass Box</p>
            <h2 className="mb-5" style={{ fontSize: "32px", fontWeight: 300, lineHeight: 1.1, letterSpacing: "-0.64px", color: "#ffffff" }}>
              Every generation is a readable Python function
            </h2>
            <p className="mb-6" style={{ fontSize: "16px", fontWeight: 300, lineHeight: 1.5, color: "rgba(255,255,255,0.7)" }}>
              No black box. SynthFlow synthesizes a pure Python function you can read, audit, modify, and run yourself — with no API calls, no dependencies except numpy and pandas.
            </p>
            <Link
              href="/signup"
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-[4px] text-[15px] font-[400]"
              style={{ background: "#533afd", color: "#fff" }}
            >
              Try it free →
            </Link>
          </div>

          <div className="rounded-[6px] overflow-hidden border" style={{ borderColor: "rgba(255,255,255,0.1)" }}>
            <div className="px-4 py-2.5 border-b flex items-center gap-2" style={{ background: "rgba(255,255,255,0.04)", borderColor: "rgba(255,255,255,0.08)" }}>
              <span className="text-[12px] font-[500]" style={{ fontFamily: "monospace", color: "#b9b9f9" }}>generate.py</span>
            </div>
            <pre className="p-5 overflow-x-auto text-[12px] leading-[1.7]" style={{ fontFamily: "monospace", color: "rgba(255,255,255,0.8)", background: "rgba(0,0,0,0.3)" }}>
              <code>{code}</code>
            </pre>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ── Pricing Teaser ─────────────────────────────────────────────────────── */
function PricingTeaser() {
  const plans = [
    { id: "free", name: "Free", price: "$0", priceNote: "forever", cta: "Get started", ctaHref: "/signup", highlight: false,
      features: ["10 generations/month", "1,000 rows max", "1 dataset upload", "Community support"] },
    { id: "pro", name: "Pro", price: "$19", priceNote: "/month", cta: "Start free trial", ctaHref: "/signup", highlight: true,
      features: ["200 generations/month", "100,000 rows max", "10 dataset uploads", "API access", "Priority support"] },
    { id: "business", name: "Business", price: "$49", priceNote: "/month", cta: "Start free trial", ctaHref: "/signup", highlight: false,
      features: ["Unlimited generations", "1M rows max", "50 dataset uploads", "Team (10 members)", "Webhooks", "Dedicated support"] },
    { id: "enterprise", name: "Enterprise", price: "Custom", priceNote: "", cta: "Contact sales", ctaHref: "mailto:sales@synthflow.ai", highlight: false,
      features: ["Unlimited everything", "Custom integrations", "SSO/SAML", "SLA", "Onboarding"] },
  ];

  return (
    <section className="py-24" style={{ background: "#ffffff" }}>
      <div className="mx-auto max-w-[1080px] px-6">
        <div className="text-center mb-14">
          <h2 className="mb-4" style={{ fontSize: "32px", fontWeight: 300, lineHeight: 1.1, letterSpacing: "-0.64px", color: "#061b31" }}>
            Simple, transparent pricing
          </h2>
          <p style={{ fontSize: "16px", fontWeight: 300, color: "#64748d" }}>Start free. Upgrade when you need more.</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {plans.map(({ id, name, price, priceNote, cta, ctaHref, highlight, features }) => (
            <div
              key={id}
              className={`rounded-[6px] border p-6 flex flex-col ${highlight ? "border-[#533afd]" : "border-[#e5edf5]"}`}
              style={{
                boxShadow: highlight
                  ? "rgba(83,58,253,0.2) 0px 20px 40px -20px, rgba(50,50,93,0.15) 0px 20px 40px -20px"
                  : "rgba(50,50,93,0.1) 0px 20px 40px -25px",
              }}
            >
              {highlight && (
                <div className="mb-3">
                  <span className="text-[10px] font-[400] px-2 py-0.5 rounded-[4px] border border-[rgba(83,58,253,0.3)] bg-[rgba(83,58,253,0.06)]" style={{ color: "#533afd" }}>
                    Most popular
                  </span>
                </div>
              )}
              <h3 className="mb-1" style={{ fontSize: "16px", fontWeight: 400, color: "#061b31" }}>{name}</h3>
              <div className="mb-5 flex items-baseline gap-1">
                <span style={{ fontSize: "28px", fontWeight: 300, color: "#061b31" }}>{price}</span>
                {priceNote && <span style={{ fontSize: "14px", fontWeight: 300, color: "#64748d" }}>{priceNote}</span>}
              </div>
              <ul className="space-y-2 mb-6 flex-1">
                {features.map(f => (
                  <li key={f} className="flex items-start gap-2">
                    <span className="mt-0.5 text-[#15be53] text-[12px]">✓</span>
                    <span style={{ fontSize: "13px", fontWeight: 300, color: "#64748d" }}>{f}</span>
                  </li>
                ))}
              </ul>
              <Link
                href={ctaHref}
                className="text-center text-[14px] font-[400] py-2 rounded-[4px] transition-colors"
                style={
                  highlight
                    ? { background: "#533afd", color: "#fff" }
                    : { background: "transparent", color: "#533afd", border: "1px solid #b9b9f9" }
                }
              >
                {cta}
              </Link>
            </div>
          ))}
        </div>

        <p className="text-center mt-8 text-[14px] font-[300]" style={{ color: "#64748d" }}>
          All plans include Indian (₹) and international ($) pricing.{" "}
          <Link href="/pricing" className="text-[#533afd]">See full comparison →</Link>
        </p>
      </div>
    </section>
  );
}

/* ── Final CTA ──────────────────────────────────────────────────────────── */
function FinalCTA() {
  return (
    <section className="py-24" style={{ background: "#1c1e54" }}>
      <div className="mx-auto max-w-[1080px] px-6 text-center">
        <h2 className="mb-5" style={{ fontSize: "48px", fontWeight: 300, lineHeight: 1.1, letterSpacing: "-0.96px", color: "#ffffff" }}>
          Ready to generate your first dataset?
        </h2>
        <p className="mb-10" style={{ fontSize: "18px", fontWeight: 300, color: "rgba(255,255,255,0.7)" }}>
          Free forever. No credit card. No configuration. Just describe what you need.
        </p>
        <Link
          href="/signup"
          className="inline-flex items-center gap-2 px-6 py-3 rounded-[4px] text-[16px] font-[400]"
          style={{ background: "#533afd", color: "#fff" }}
        >
          Get Started Free →
        </Link>
      </div>
    </section>
  );
}

/* ── Footer ─────────────────────────────────────────────────────────────── */
function Footer() {
  return (
    <footer style={{ background: "#0d253d", color: "rgba(255,255,255,0.6)" }}>
      <div className="mx-auto max-w-[1080px] px-6 py-12">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded-[4px] flex items-center justify-center bg-white/10">
              <span className="text-white text-[12px]" style={{ fontWeight: 300 }}>S</span>
            </div>
            <span className="text-white text-[13px] font-[400]">SynthFlow</span>
          </div>
          <p className="text-[12px] font-[300]" style={{ color: "rgba(255,255,255,0.35)" }}>
            © 2026 SynthFlow · INTEGRATED SYNERGY
          </p>
          <div className="flex gap-4">
            {["Privacy", "Terms", "Contact"].map(l => (
              <a key={l} href="#" className="text-[13px] font-[300] hover:text-white transition-colors">{l}</a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}

/* ── Page ────────────────────────────────────────────────────────────────── */
export default function LandingPage() {
  return (
    <div className="min-h-screen" style={{ background: "#ffffff", fontFamily: '"Geist", system-ui, -apple-system, sans-serif', fontFeatureSettings: '"ss01"' }}>
      <Nav />
      <Hero />
      <HowItWorks />
      <FivePillars />
      <DomainShowcase />
      <GlassBox />
      <PricingTeaser />
      <FinalCTA />
      <Footer />
    </div>
  );
}
