// Static export so FastAPI can mount `out/` as a single-process deployment.
// In dev, rewrites proxy API calls to the local FastAPI server (default :8765).

const API = process.env.TEN_API || "http://127.0.0.1:8765";

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",
  trailingSlash: true,
  // `next/image` optimization is a runtime feature; static export needs unoptimized.
  images: { unoptimized: true },
  async rewrites() {
    return [
      { source: "/search/:path*", destination: `${API}/search/:path*` },
      { source: "/search",        destination: `${API}/search` },
      { source: "/stats",         destination: `${API}/stats` },
      { source: "/health",        destination: `${API}/health` },
      { source: "/clip/:path*",   destination: `${API}/clip/:path*` },
      { source: "/thumb/:path*",  destination: `${API}/thumb/:path*` },
    ];
  },
};

export default nextConfig;
