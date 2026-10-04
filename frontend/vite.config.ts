import path from "node:path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// Dev server proxies API calls to a running service (local or the live server):
//   SENTINEL_API=https://zylo7-sentinel.duckdns.org pnpm dev
const api = process.env.SENTINEL_API ?? "http://localhost:8000"

export default defineConfig({
  base: "/ui/", // the FastAPI app serves index.html at "/" and the build output under "/ui"
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  server: {
    proxy: Object.fromEntries(
      ["/predict", "/batch", "/health", "/model-card"].map((p) => [p, { target: api, changeOrigin: true, secure: true }]),
    ),
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
})
