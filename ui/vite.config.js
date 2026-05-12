import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const API = process.env.TEN_API || 'http://127.0.0.1:8765'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/search':       { target: API, changeOrigin: true },
      '/stats':        { target: API, changeOrigin: true },
      '/health':       { target: API, changeOrigin: true },
      '/clip':         { target: API, changeOrigin: true },
      '/thumb':        { target: API, changeOrigin: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})
