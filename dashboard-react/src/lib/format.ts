// ── Formatting utilities ───────────────────────────────────────

export function formatNumber(num: number): string {
  return Math.round(num).toLocaleString();
}

export function formatNumberShort(num: number): string {
  if (num >= 1_000_000) return (num / 1_000_000).toFixed(1) + 'M';
  if (num >= 1_000) return (num / 1_000).toFixed(1) + 'k';
  return Math.round(num).toString();
}

export function getRelativeTime(isoString: string | undefined): string {
  if (!isoString) return 'Never';
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 10) return 'Just now';
  if (diffSec < 60) return `${diffSec}s ago`;
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHour = Math.floor(diffMin / 60);
  if (diffHour < 24) return `${diffHour}h ago`;
  return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

export function getCycleCountdown(cycleStartStr: string | undefined): string {
  if (!cycleStartStr) return 'Not Syncing';
  const start = new Date(cycleStartStr);
  const end = new Date(start.getTime() + 24 * 60 * 60 * 1000);
  const now = new Date();
  const diff = end.getTime() - now.getTime();
  if (diff <= 0) return 'Resetting now...';
  const hours = Math.floor(diff / (1000 * 60 * 60));
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
  const seconds = Math.floor((diff % (1000 * 60)) / 1000);
  return `${hours}h ${minutes}m ${seconds}s`;
}

export function computeAdminIdentifier(passcode: string): string | null {
  if (!passcode) return null;
  let h = 0;
  for (let i = 0; i < passcode.length; i++) {
    h = ((h << 5) - h) + passcode.charCodeAt(i);
    h |= 0;
  }
  const hex = (h >>> 0).toString(16).padStart(8, '0').toUpperCase();
  return `admin_${hex}`;
}

export function getPasscodeStrength(pass: string): { score: number; text: string; color: string } {
  if (!pass) return { score: 0, text: '', color: '#7b7e8a' };
  if (pass.length < 6) return { score: 1, text: 'Short (min 6 characters)', color: '#ef4565' };
  let score = 2;
  const hasLetters = /[a-zA-Z]/.test(pass);
  const hasNumbers = /[0-9]/.test(pass);
  const hasSpecial = /[^a-zA-Z0-9]/.test(pass);
  if (hasLetters && hasNumbers) score = 3;
  if (hasLetters && hasNumbers && hasSpecial && pass.length >= 8) score = 4;
  if (score === 2) return { score: 2, text: 'Weak — mix letters and numbers', color: '#f59e0b' };
  if (score === 3) return { score: 3, text: 'Medium strength', color: '#eab308' };
  return { score: 4, text: 'Strong passcode', color: '#22c55e' };
}
