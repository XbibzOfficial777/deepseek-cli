import type { Theme, ThemeKey } from './types';

export const THEMES: Record<ThemeKey, Theme> = {
  blue: {
    name: 'Blue',
    primary: '#5b8def',
    primaryGlow: 'rgba(91,141,239,0.18)',
    primaryBorder: 'rgba(91,141,239,0.16)',
    primaryBg: 'rgba(91,141,239,0.06)',
    accent: '#a78bfa',
  },
  violet: {
    name: 'Violet',
    primary: '#a78bfa',
    primaryGlow: 'rgba(167,139,250,0.18)',
    primaryBorder: 'rgba(167,139,250,0.16)',
    primaryBg: 'rgba(167,139,250,0.06)',
    accent: '#f472b6',
  },
  emerald: {
    name: 'Emerald',
    primary: '#22c55e',
    primaryGlow: 'rgba(34,197,94,0.18)',
    primaryBorder: 'rgba(34,197,94,0.16)',
    primaryBg: 'rgba(34,197,94,0.06)',
    accent: '#a78bfa',
  },
  rose: {
    name: 'Rose',
    primary: '#ef4565',
    primaryGlow: 'rgba(239,69,101,0.18)',
    primaryBorder: 'rgba(239,69,101,0.16)',
    primaryBg: 'rgba(239,69,101,0.06)',
    accent: '#a78bfa',
  },
};

export const DEFAULT_THEME: ThemeKey = 'blue';
export const THEME_STORAGE_KEY = 'deepseek_admin_theme';

export function getStoredTheme(): ThemeKey {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  if (stored && stored in THEMES) return stored as ThemeKey;
  return DEFAULT_THEME;
}

export function setStoredTheme(key: ThemeKey): void {
  localStorage.setItem(THEME_STORAGE_KEY, key);
}
