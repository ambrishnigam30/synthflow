export default function Home() {
  return (
    <main
      style={{
        backgroundColor: "#ffffff",
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: '"Geist", system-ui, -apple-system, sans-serif',
        fontFeatureSettings: '"ss01"',
      }}
    >
      <div style={{ textAlign: "center", maxWidth: "560px", padding: "0 24px" }}>
        {/* Logo mark placeholder */}
        <div
          style={{
            width: "48px",
            height: "48px",
            borderRadius: "8px",
            backgroundColor: "#3d4043",
            margin: "0 auto 32px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <span
            style={{
              color: "#ffffff",
              fontSize: "24px",
              fontWeight: 300,
              letterSpacing: "-0.5px",
            }}
          >
            S
          </span>
        </div>

        <h1
          style={{
            fontSize: "48px",
            fontWeight: 300,
            lineHeight: 1.15,
            letterSpacing: "-0.96px",
            color: "#061b31",
            margin: "0 0 16px",
          }}
        >
          SynthFlow
        </h1>

        <p
          style={{
            fontSize: "18px",
            fontWeight: 300,
            lineHeight: 1.4,
            color: "#64748d",
            margin: "0 0 48px",
          }}
        >
          Data that understands the real world
        </p>

        <span
          style={{
            display: "inline-block",
            padding: "1px 8px",
            borderRadius: "4px",
            backgroundColor: "rgba(21,190,83,0.15)",
            border: "1px solid rgba(21,190,83,0.4)",
            color: "#108c3d",
            fontSize: "12px",
            fontWeight: 300,
          }}
        >
          Layer 0 scaffold — coming soon
        </span>
      </div>
    </main>
  );
}
