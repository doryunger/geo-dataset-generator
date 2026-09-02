import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev-only convenience: proxy /api/* to the FastAPI server (see oil_refinery/app/run_server.*)
// so the frontend can call relative paths with no CORS setup. Not present in a production
// deployment -- see the plan's "Remote hosting readiness" notes for how that would be served.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8010',
    },
  },
})
