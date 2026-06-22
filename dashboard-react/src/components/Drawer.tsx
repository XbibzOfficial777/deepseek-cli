import { useState, useEffect } from 'react';
import { Monitor, Command, Apple, Terminal, Gavel, ShieldCheck, Edit3, X } from 'lucide-react';
import type { UserRecord } from '../lib/types';
import { formatNumberShort, getCycleCountdown } from '../lib/format';

interface DrawerProps {
  user: UserRecord | null;
  onClose: () => void;
  onBan: (user: UserRecord) => void;
  onLimit: (user: UserRecord) => void;
}

function getPlatformIcon(platform: string | undefined) {
  if (!platform) return Monitor;
  if (platform.includes('win')) return Command;
  if (platform.includes('darwin')) return Apple;
  if (platform.includes('linux')) return Terminal;
  return Monitor;
}

export function Drawer({ user, onClose, onBan, onLimit }: DrawerProps) {
  const [countdown, setCountdown] = useState('');

  useEffect(() => {
    if (!user) return;
    const update = () => setCountdown(getCycleCountdown(user.cycle_start));
    update();
    const timer = setInterval(update, 1000);
    return () => clearInterval(timer);
  }, [user]);

  if (!user) return null;

  const tokens = user.tokens || { input: 0, output: 0, total: 0, limit: 0 };
  const total = tokens.total || 1;
  const inPct = Math.round(((tokens.input || 0) / total) * 100);
  const outPct = Math.round(((tokens.output || 0) / total) * 100);
  const PlatformIcon = getPlatformIcon(user.platform);
  const usernameParts = (user.username || '').split('@');
  const sysUser = usernameParts[0] || user.username;
  const hostnameVal = user.hostname || usernameParts[1] || 'Unknown';

  return (
    <div id="drawer-overlay" className="drawer-overlay active">
      <div id="drawer-backdrop" className="drawer-backdrop" onClick={onClose} />
      <div className="drawer">
        <div className="drawer-header">
          <h3>Client Profile</h3>
          <button id="drawer-close" type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div id="drawer-body" className="drawer-body">
          <div className="drawer-profile">
            <div className="drawer-avatar">
              <img
                src={user.avatar_url || `https://api.dicebear.com/7.x/bottts/svg?seed=${encodeURIComponent(user.username)}`}
                alt={user.username}
                onError={(e) => {
                  const img = e.currentTarget;
                  img.style.display = 'none';
                  img.parentElement!.textContent = (user.username || '').substring(0, 2).toUpperCase();
                }}
              />
            </div>
            <div className="drawer-name">{sysUser}</div>
            <div className="drawer-ver">{hostnameVal} · v{user.version || 'Unknown'}</div>
          </div>

          <div className="drawer-section">
            <h4>Device Info</h4>
            <div className="drawer-row">
              <span>Hostname</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>{hostnameVal}</span>
            </div>
            <div className="drawer-row">
              <span>Platform</span>
              <span>
                <PlatformIcon size={12} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 4 }} />
                {user.platform || 'Unknown'}
              </span>
            </div>
            <div className="drawer-row">
              <span>Architecture</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{user.arch || 'Unknown'}</span>
            </div>
            <div className="drawer-row">
              <span>OS Release</span>
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}>{user.os_release || 'Unknown'}</span>
            </div>
            <div className="drawer-row">
              <span>System User</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{sysUser}</span>
            </div>
          </div>

          <div className="drawer-section">
            <h4>Connection</h4>
            <div className="drawer-row">
              <span>IP Address</span>
              <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--primary)' }}>{user.ip}</span>
            </div>
            <div className="drawer-row">
              <span>Network Status</span>
              <span>
                <span className={`status-dot ${user.banned ? 'banned' : user.status}`} style={{ marginRight: 6, display: 'inline-block', verticalAlign: 'middle' }} />
                {user.banned ? 'Banned' : user.status}
              </span>
            </div>
            <div className="drawer-row">
              <span>Last Online</span>
              <span>{new Date(user.last_online).toLocaleString()}</span>
            </div>
            <div className="drawer-row">
              <span>Total Calls</span>
              <span style={{ color: 'var(--primary)', fontFamily: 'var(--font-mono)' }}>{user.total_calls || 0}</span>
            </div>
          </div>

          <div className="drawer-section">
            <h4>Token Analysis</h4>
            <div className="drawer-row">
              <span>Input Tokens</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{formatNumberShort(tokens.input || 0)}</span>
            </div>
            <div className="drawer-row">
              <span>Output Tokens</span>
              <span style={{ fontFamily: 'var(--font-mono)' }}>{formatNumberShort(tokens.output || 0)}</span>
            </div>
            <div className="drawer-row">
              <span>Total Tokens</span>
              <span style={{ color: 'var(--primary)', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                {formatNumberShort(tokens.total || 0)}
              </span>
            </div>
            <div className="drawer-row">
              <span>Token Limit</span>
              <span>{tokens.limit && tokens.limit > 0 ? formatNumberShort(tokens.limit) : 'Unlimited'}</span>
            </div>
            <div style={{ marginTop: 10 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                <span>Input {inPct}%</span>
                <span>Output {outPct}%</span>
              </div>
              <div className="split-bar">
                <div style={{ flex: tokens.input || 1, background: 'var(--accent)', minWidth: 2 }} />
                <div style={{ flex: tokens.output || 1, background: 'var(--primary)', minWidth: 2 }} />
              </div>
            </div>
          </div>

          <div className="drawer-section">
            <h4>Rolling 24-Hour Cycle</h4>
            <div className="drawer-row">
              <span>Cycle Start</span>
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)' }}>
                {user.cycle_start ? new Date(user.cycle_start).toLocaleString() : 'N/A'}
              </span>
            </div>
            <div className="drawer-row">
              <span>Cycle Reset In</span>
              <span id="drawer-countdown" style={{ color: 'var(--warning)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
                {countdown}
              </span>
            </div>
          </div>

          <div className="drawer-actions">
            <button
              id="drawer-ban-btn"
              type="button"
              className={`btn ${user.banned ? 'btn-primary' : 'btn-danger'}`}
              onClick={() => onBan(user)}
            >
              {user.banned ? <ShieldCheck size={13} /> : <Gavel size={13} />}
              {user.banned ? ' Unban IP' : ' Ban IP'}
            </button>
            <button id="drawer-limit-btn" type="button" className="btn" onClick={() => onLimit(user)}>
              <Edit3 size={13} /> Configure Token Limit
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
