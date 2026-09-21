import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // El dev server proxea /api al backend, asi que en desarrollo no hay CORS
    // y en produccion el mismo FastAPI sirve el bundle. Una sola URL en ambos casos.
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
