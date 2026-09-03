import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev-only convenience: proxy /api/* and /ws/* to the FastAPI server (see
// oil_refinery/app/run_server.*) so the frontend can call relative paths with no CORS setup. Not
// present in a production deployment -- see the plan's "Remote hosting readiness" notes for how
// that would be served. /ws needs `ws: true` -- Vite's HTTP proxy doesn't upgrade connections to a
// websocket on its own.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8010',
      '/ws': { target: 'ws://localhost:8010', ws: true },
    },
  },
  // maplibre-gl ships a *separate* worker bundle (dist/maplibre-gl-worker.mjs) that it loads
  // dynamically at runtime for GeoJSON source processing. Vite's esbuild-based dep pre-bundler
  // doesn't follow that reference, so under `optimizeDeps` it 404s every single time -- confirmed
  // live: web.err.log repeated "The file does not exist at .../maplibre-gl-worker.mjs ... Try
  // adding it to optimizeDeps.exclude" on every GeoJSON source load, and every GeoJSON source
  // (site-boundaries, site-labels) silently never finished loading as a result (source.loaded()
  // stuck at false forever, no error surfaced to the page) -- while raster sources, which need no
  // worker at all, rendered fine the whole time. Excluding it here is exactly what Vite's own
  // error message recommends.
  optimizeDeps: {
    exclude: ['maplibre-gl'],
  },
})
