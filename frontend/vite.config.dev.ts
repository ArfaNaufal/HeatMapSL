import { defineConfig, loadEnv } from 'vite'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

export default defineConfig(( { mode } ) => {
  return {
    plugins: [
      // The React and Tailwind plugins are both required for Make, even if
      // Tailwind is not being actively used – do not remove them
      react(),
      tailwindcss(),
      basicSsl(),
    ],
    
    server: {
      port: 3000,
      https: true,
      host: true,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000', // Use IP instead of localhost
          changeOrigin: true,
          secure: false,
        }
      },
    },
    resolve: {
      alias: {
        // Alias @ to the src directory
        '@': path.resolve(__dirname, './src'),
      },
    },
  }
})
