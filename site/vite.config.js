import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
// The site reads the level results JSON files at build time, so every recorded number on the page is imported, never typed.
export default defineConfig({
  base: '/finetune-lab/',
  plugins: [react()],
  server: { fs: { allow: ['..'] } },
});
