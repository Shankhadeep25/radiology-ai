/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        display: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
      },
      colors: {
        clinical: {
          50: '#ecfeff',
          100: '#cffafe',
          200: '#a5f3fc',
          300: '#67e8f9',
          400: '#22d3ee',
          500: '#06b6d4',
          600: '#0891b2',
          700: '#0e7490',
          900: '#164e63',
        },
        accent: {
          violet: '#8b5cf6',
          pink: '#ec4899',
          cyan: '#22d3ee',
        },
      },
      boxShadow: {
        glass: '0 4px 16px rgba(0, 0, 0, 0.1)',
        glow: '0 0 20px rgba(34, 211, 238, 0.2)',
        'glow-violet': '0 0 20px rgba(139, 92, 246, 0.2)',
      },
    },
  },
  plugins: [],
}
