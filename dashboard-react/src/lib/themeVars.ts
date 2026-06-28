import { THEMES } from './theme';
import type { ThemeKey } from './types';

export function applyThemeVars(themeKey: ThemeKey): void {
  const t = THEMES[themeKey];
  if (!t) return;
  const root = document.documentElement;
  root.style.setProperty('--primary', t.primary);
  root.style.setProperty('--primary-glow', t.primaryGlow);
  root.style.setProperty('--primary-border', t.primaryBorder);
  root.style.setProperty('--primary-bg', t.primaryBg);
  root.style.setProperty('--accent', t.accent);
}
