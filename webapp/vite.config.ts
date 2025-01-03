import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    hmr: false
    /*{
      host: 'app.codecollective.us', // Use your domain for HMR in production
      protocol: 'wss', // Secure WebSocket (works with port 443)
    },*/
  },
});