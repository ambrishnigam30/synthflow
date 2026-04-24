import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Move the dev-mode indicator away from the content area
  devIndicators: {
    position: "bottom-left",
  },
};

export default nextConfig;
