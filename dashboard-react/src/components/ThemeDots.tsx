import { THEMES, setStoredTheme } from '../lib/theme';
import type { ThemeKey } from '../lib/types';
import { useToast } from './Toast';

interface ThemeDotsProps {
  active: ThemeKey;
  onChange: (key: ThemeKey) => void;
}

export function ThemeDots({ active, onChange }: ThemeDotsProps) {
  const { showToast } = useToast();
  const keys = Object.keys(THEMES) as ThemeKey[];

  const handleClick = (key: ThemeKey) => {
    onChange(key);
    setStoredTheme(key);
    showToast(`Theme: ${THEMES[key].name}`, 'info');
  };

  return (
    <div className="theme-dots" role="radiogroup" aria-label="Theme">
      {keys.map((key) => {
        const t = THEMES[key];
        return (
          <button
            key={key}
            type="button"
            className={`theme-dot ${key === active ? 'active' : ''}`}
            data-theme={key}
            onClick={() => handleClick(key)}
            aria-label={`Theme: ${t.name}`}
            title={t.name}
            style={{ background: `linear-gradient(135deg, ${t.primary}, ${t.accent})` }}
          />
        );
      })}
    </div>
  );
}
