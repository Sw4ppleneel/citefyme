import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 5173 is taken by the other projects in this workspace; the backend's CORS
// allowlist is pinned to this port.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: { '/api': { target: 'http://127.0.0.1:8002', changeOrigin: true } },
  },
})
