import Link from "next/link";

const PLANS = [
  {
    id: "free",
    name: "Free",
    price_usd: "$0",
    price_inr: "₹0",
    period: "forever",
    cta: "Get started",
    ctaHref: "/signup",
    highlight: false,
    features: {
      "Generations / month": "10",
      "Max rows per generation": "1,000",
      "Dataset uploads": "1",
      "API access": false,
      "Team members": "—",
      "Webhooks": false,
      "Priority support": false,
      "Glass Box code": true,
      "CSV / JSON / Parquet export": true,
    },
  },
  {
    id: "pro",
    name: "Pro",
    price_usd: "$19",
    price_inr: "₹499",
    period: "/ month",
    cta: "Start free trial",
    ctaHref: "/signup",
    highlight: true,
    badge: "Most Popular",
    features: {
      "Generations / month": "200",
      "Max rows per generation": "100,000",
      "Dataset uploads": "10",
      "API access": true,
      "Team members": "—",
      "Webhooks": false,
      "Priority support": false,
      "Glass Box code": true,
      "CSV / JSON / Parquet export": true,
    },
  },
  {
    id: "business",
    name: "Business",
    price_usd: "$49",
    price_inr: "₹1,999",
    period: "/ month",
    cta: "Start free trial",
    ctaHref: "/signup",
    highlight: false,
    features: {
      "Generations / month": "Unlimited",
      "Max rows per generation": "1,000,000",
      "Dataset uploads": "50",
      "API access": true,
      "Team members": "10",
      "Webhooks": true,
      "Priority support": true,
      "Glass Box code": true,
      "CSV / JSON / Parquet export": true,
    },
  },
  {
    id: "enterprise",
    name: "Enterprise",
    price_usd: "Custom",
    price_inr: "Custom",
    period: "",
    cta: "Contact sales",
    ctaHref: "mailto:sales@synthflow.ai",
    highlight: false,
    features: {
      "Generations / month": "Unlimited",
      "Max rows per generation": "Unlimited",
      "Dataset uploads": "Unlimited",
      "API access": true,
      "Team members": "Unlimited",
      "Webhooks": true,
      "Priority support": true,
      "Glass Box code": true,
      "CSV / JSON / Parquet export": true,
    },
  },
];

const ALL_FEATURES = [
  "Generations / month",
  "Max rows per generation",
  "Dataset uploads",
  "API access",
  "Team members",
  "Webhooks",
  "Priority support",
  "Glass Box code",
  "CSV / JSON / Parquet export",
];

function FeatureValue({ value }: { value: boolean | string }) {
  if (value === true) return <span style={{ color: "#15be53" }}>✓</span>;
  if (value === false) return <span style={{ color: "#64748d" }}>—</span>;
  return <span>{value}</span>;
}

const FAQ = [
  {
    q: "Can I change plans later?",
    a: "Yes. You can upgrade or downgrade at any time. Upgrades take effect immediately; downgrades apply at the end of your billing period.",
  },
  {
    q: "What payment methods do you accept?",
    a: "We accept all major credit cards globally (via Stripe) and UPI / Indian cards via Razorpay for Indian customers.",
  },
  {
    q: "What happens when I reach my generation limit?",
    a: "You'll receive a warning at 80% usage. When the limit is reached, additional generations will be blocked until the next billing cycle or you upgrade.",
  },
  {
    q: "Is the generated data really private?",
    a: "Yes. SynthFlow generates synthetic data — it never trains on, stores, or transmits your real data. All processing happens in isolated containers.",
  },
  {
    q: "What is Glass Box code?",
    a: "Every generation produces a standalone, readable Python function (generate.py) you can download, audit, and run locally. No black boxes.",
  },
];

export default function PricingPage() {
  return (
    <div style={{ background: "#ffffff", fontFamily: '"Geist", system-ui, -apple-system, sans-serif', fontFeatureSettings: '"ss01"' }}>

      {/* Hero */}
      <section className="py-20 text-center mx-auto max-w-[1080px] px-6">
        <h1 className="mb-4" style={{ fontSize: "48px", fontWeight: 300, lineHeight: 1.1, letterSpacing: "-0.96px", color: "#061b31" }}>
          Simple, transparent pricing
        </h1>
        <p style={{ fontSize: "18px", fontWeight: 300, color: "#64748d" }}>
          Start free. Pay only when you need more. No hidden fees.
        </p>
        <div className="mt-4 flex items-center justify-center gap-2">
          <span className="text-[12px] font-[400]" style={{ color: "#64748d" }}>USD</span>
          <span className="text-[12px] font-[400]" style={{ color: "#064748d" }}>·</span>
          <span className="text-[12px] font-[400]" style={{ color: "#64748d" }}>INR prices shown below</span>
        </div>
      </section>

      {/* Plan cards */}
      <section className="pb-20 mx-auto max-w-[1080px] px-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {PLANS.map(plan => (
            <div
              key={plan.id}
              className={`rounded-[6px] border p-6 flex flex-col ${plan.highlight ? "border-[#533afd]" : "border-[#e5edf5]"}`}
              style={{
                boxShadow: plan.highlight
                  ? "rgba(83,58,253,0.2) 0px 20px 40px -20px, rgba(50,50,93,0.15) 0px 20px 40px -20px"
                  : "rgba(50,50,93,0.1) 0px 20px 40px -25px",
              }}
            >
              {(plan as typeof plan & { badge?: string }).badge && (
                <div className="mb-3">
                  <span className="text-[10px] font-[400] px-2 py-0.5 rounded-[4px] border border-[rgba(83,58,253,0.3)] bg-[rgba(83,58,253,0.06)]" style={{ color: "#533afd" }}>
                    {(plan as typeof plan & { badge?: string }).badge}
                  </span>
                </div>
              )}
              <h2 className="mb-2" style={{ fontSize: "18px", fontWeight: 400, color: "#061b31" }}>{plan.name}</h2>
              <div className="mb-1 flex items-baseline gap-1">
                <span style={{ fontSize: "32px", fontWeight: 300, color: "#061b31", letterSpacing: "-0.64px" }}>
                  {plan.price_usd}
                </span>
                {plan.period && <span style={{ fontSize: "14px", fontWeight: 300, color: "#64748d" }}>{plan.period}</span>}
              </div>
              {plan.price_inr !== "Custom" && (
                <p className="mb-5 text-[12px] font-[300]" style={{ color: "#64748d" }}>
                  {plan.price_inr}{plan.period}
                </p>
              )}
              <div className="flex-1" />
              <Link
                href={plan.ctaHref}
                className="mt-4 text-center text-[14px] font-[400] py-2 px-4 rounded-[4px] transition-colors block"
                style={
                  plan.highlight
                    ? { background: "#533afd", color: "#fff" }
                    : { background: "transparent", color: "#533afd", border: "1px solid #b9b9f9" }
                }
              >
                {plan.cta}
              </Link>
            </div>
          ))}
        </div>
      </section>

      {/* Feature comparison matrix */}
      <section className="py-16" style={{ background: "#f8fafc" }}>
        <div className="mx-auto max-w-[1080px] px-6">
          <h2 className="mb-10 text-center" style={{ fontSize: "26px", fontWeight: 300, letterSpacing: "-0.26px", color: "#061b31" }}>
            Full feature comparison
          </h2>
          <div className="overflow-x-auto rounded-[6px] border border-[#e5edf5] bg-white"
            style={{ boxShadow: "rgba(50,50,93,0.1) 0px 10px 30px -15px" }}>
            <table className="w-full">
              <thead>
                <tr style={{ background: "#f8fafc", borderBottom: "1px solid #e5edf5" }}>
                  <th className="text-left px-5 py-3 text-[12px] font-[400] w-[200px]" style={{ color: "#64748d" }}>Feature</th>
                  {PLANS.map(p => (
                    <th key={p.id} className="px-4 py-3 text-[12px] font-[400] text-center" style={{ color: p.highlight ? "#533afd" : "#061b31" }}>
                      {p.name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ALL_FEATURES.map((feature, i) => (
                  <tr key={feature} style={{ borderBottom: i < ALL_FEATURES.length - 1 ? "1px solid #e5edf5" : undefined }}>
                    <td className="px-5 py-3 text-[13px] font-[300]" style={{ color: "#273951" }}>{feature}</td>
                    {PLANS.map(p => (
                      <td key={p.id} className="px-4 py-3 text-[13px] font-[300] text-center" style={{ color: "#273951" }}>
                        <FeatureValue value={(p.features as Record<string, boolean | string>)[feature]} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section className="py-20 mx-auto max-w-[680px] px-6">
        <h2 className="mb-10 text-center" style={{ fontSize: "26px", fontWeight: 300, letterSpacing: "-0.26px", color: "#061b31" }}>
          Frequently asked questions
        </h2>
        <div className="space-y-6">
          {FAQ.map(({ q, a }) => (
            <div key={q} className="rounded-[5px] border border-[#e5edf5] p-5">
              <h3 className="mb-2" style={{ fontSize: "15px", fontWeight: 400, color: "#061b31" }}>{q}</h3>
              <p style={{ fontSize: "14px", fontWeight: 300, lineHeight: 1.6, color: "#64748d" }}>{a}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 text-center" style={{ background: "#1c1e54" }}>
        <h2 className="mb-5" style={{ fontSize: "32px", fontWeight: 300, letterSpacing: "-0.64px", color: "#ffffff" }}>
          Start generating for free
        </h2>
        <p className="mb-8" style={{ fontSize: "16px", fontWeight: 300, color: "rgba(255,255,255,0.7)" }}>
          No credit card required. Upgrade anytime.
        </p>
        <Link
          href="/signup"
          className="inline-flex items-center gap-2 px-5 py-2.5 rounded-[4px] text-[15px] font-[400]"
          style={{ background: "#533afd", color: "#fff" }}
        >
          Create free account →
        </Link>
      </section>
    </div>
  );
}
