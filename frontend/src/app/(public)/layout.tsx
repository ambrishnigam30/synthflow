import Link from "next/link";

function MarketingNav() {
  return (
    <header
      className="sticky top-0 z-50 w-full"
      style={{ background: "rgba(255,255,255,0.92)", backdropFilter: "blur(12px)" }}
    >
      <div className="mx-auto max-w-[1080px] px-6 h-14 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2.5">
          <div
            className="w-7 h-7 rounded-[4px] flex items-center justify-center"
            style={{ backgroundColor: "#3d4043" }}
          >
            <span className="text-white text-[14px]" style={{ fontWeight: 300 }}>S</span>
          </div>
          <span
            className="text-[14px]"
            style={{ color: "#061b31", fontWeight: 400, letterSpacing: "-0.14px" }}
          >
            SynthFlow
          </span>
        </Link>

        {/* Nav links */}
        <nav className="hidden md:flex items-center gap-6">
          {[
            { href: "/#how-it-works", label: "Product" },
            { href: "/pricing", label: "Pricing" },
            { href: "#", label: "Docs" },
          ].map(({ href, label }) => (
            <Link
              key={label}
              href={href}
              className="text-[14px] font-[400] transition-colors hover:text-[#533afd]"
              style={{ color: "#061b31" }}
            >
              {label}
            </Link>
          ))}
        </nav>

        {/* CTAs */}
        <div className="flex items-center gap-3">
          <Link
            href="/login"
            className="text-[14px] font-[400] hidden sm:block transition-colors"
            style={{ color: "#061b31" }}
          >
            Log in
          </Link>
          <Link
            href="/signup"
            className="text-[14px] font-[400] px-4 py-[6px] rounded-[4px] transition-colors"
            style={{ background: "#533afd", color: "#ffffff" }}
          >
            Get Started Free
          </Link>
        </div>
      </div>
    </header>
  );
}

function MarketingFooter() {
  return (
    <footer style={{ background: "#1c1e54", color: "rgba(255,255,255,0.7)" }}>
      <div className="mx-auto max-w-[1080px] px-6 py-16">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-12">
          {/* Brand */}
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-7 h-7 rounded-[4px] flex items-center justify-center bg-white/10">
                <span className="text-white text-[14px]" style={{ fontWeight: 300 }}>S</span>
              </div>
              <span className="text-white text-[14px] font-[400]">SynthFlow</span>
            </div>
            <p className="text-[14px] font-[300] leading-relaxed">
              Autonomous synthetic data generation. Data that understands the real world.
            </p>
          </div>

          {[
            {
              title: "Product",
              links: ["Features", "Pricing", "Changelog", "Roadmap"],
            },
            {
              title: "Resources",
              links: ["Documentation", "API Reference", "Blog", "Community"],
            },
            {
              title: "Company",
              links: ["About", "Privacy", "Terms", "Contact"],
            },
          ].map(({ title, links }) => (
            <div key={title}>
              <h4 className="text-white text-[13px] font-[400] mb-4">{title}</h4>
              <ul className="space-y-2">
                {links.map(link => (
                  <li key={link}>
                    <a
                      href="#"
                      className="text-[14px] font-[300] transition-colors hover:text-white"
                      style={{ color: "rgba(255,255,255,0.6)" }}
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>

        <div className="border-t border-white/10 pt-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-[13px] font-[300]" style={{ color: "rgba(255,255,255,0.5)" }}>
            © 2026 SynthFlow. All rights reserved.
          </p>
          <p className="text-[12px] font-[300]" style={{ color: "rgba(255,255,255,0.35)" }}>
            INTEGRATED SYNERGY
          </p>
        </div>
      </div>
    </footer>
  );
}

export default function PublicLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen flex flex-col" style={{ background: "#ffffff" }}>
      <MarketingNav />
      <main className="flex-1">{children}</main>
      <MarketingFooter />
    </div>
  );
}
