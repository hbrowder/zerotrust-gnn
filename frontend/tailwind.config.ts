import type { Config } from 'tailwindcss'

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gray: {
          900: '#111827',
          800: '#1f2937',
          700: '#374151',
          400: '#9ca3af',
          200: '#e5e7eb',
        },
        blue: {
          900: '#1e3a8a',
          400: '#60a5fa',
        },
        green: {
          400: '#4ade80',
          600: '#16a34a',
        },
        red: {
          400: '#f87171',
          600: '#dc2626',
          700: '#b91c1c',
          900: '#7f1d1d',
        },
        orange: {
          600: '#ea580c',
          700: '#c2410c',
          900: '#7c2d12',
        },
        yellow: {
          400: '#facc15',
          600: '#ca8a04',
          700: '#a16207',
          900: '#713f12',
        },
      },
      backdropBlur: {
        sm: '4px',
      },
    },
  },
  plugins: [],
} satisfies Config
