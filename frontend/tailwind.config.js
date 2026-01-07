/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Claude aesthetic palette (Anthropic)
        claude: {
          charcoal: '#1F2937', // Charcoal grey
          'charcoal-light': '#374151',
          'charcoal-dark': '#111827',
          coral: '#FF6B6B', // Coral orange
          'coral-light': '#FF8787',
          'coral-dark': '#E55555',
          white: '#FFFFFF',
          'grey-light': '#E5E7EB',
          'grey': '#9CA3AF',
          'grey-dark': '#6B7280',
        },
        primary: {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
      },
      animation: {
        'float': 'float 3s ease-in-out infinite',
        'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
        'data-flow': 'data-flow 2s linear infinite',
        'robot-idle': 'robot-idle 2s ease-in-out infinite',
        'robot-active': 'robot-active 0.5s ease-in-out infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        'glow-pulse': {
          '0%, 100%': { opacity: '0.5', boxShadow: '0 0 10px currentColor' },
          '50%': { opacity: '1', boxShadow: '0 0 20px currentColor, 0 0 30px currentColor' },
        },
        'data-flow': {
          '0%': { transform: 'translateX(-100%)', opacity: '0' },
          '50%': { opacity: '1' },
          '100%': { transform: 'translateX(100%)', opacity: '0' },
        },
        'robot-idle': {
          '0%, 100%': { transform: 'translateY(0px) rotate(0deg)' },
          '25%': { transform: 'translateY(-3px) rotate(1deg)' },
          '75%': { transform: 'translateY(-3px) rotate(-1deg)' },
        },
        'robot-active': {
          '0%, 100%': { transform: 'scale(1) rotate(0deg)' },
          '50%': { transform: 'scale(1.05) rotate(2deg)' },
        },
      },
    },
  },
  plugins: [],
}

