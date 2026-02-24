/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        'nmri-bg': '#0a0c10',
        'nmri-bg-secondary': '#0c0e14',
        'nmri-surface': '#12151c',
        'nmri-text': '#66aa88',
        'nmri-text-secondary': '#556677',
        'nmri-text-data': '#aabbcc',
        'nmri-accent': '#00ffaa',
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "'Fira Code'", "'Courier New'", 'monospace'],
      },
    },
  },
  plugins: [],
};
