import { useState, useEffect } from 'react';
import { X, RefreshCw, Send, Bell, CheckCircle2, AlertTriangle, Info } from 'lucide-react';
import type { CliUser } from '../lib/types';

// Telegram contact link — used for banned/limited users
const TELEGRAM_URL = 'https://t.me/XbibzOfficial';
const TELEGRAM_LABEL = '@XbibzOfficial on Telegram';

interface CliUsersModalProps {
  isOpen: boolean;
  users: CliUser[] | undefined;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onRefresh: () => void;
  onSearch: (query: string) => void;
  onBanToggle: (uid: string, banned: boolean) => void;
  onDelete: (uid: string) => void;
  onNotify?: (uid: string, username: string) => void;
}

export function CliUsersModal({
  isOpen,
  users,
  loading,
  error,
  onClose,
  onRefresh,
  onSearch,
  onBanToggle,
  onDelete,
  onNotify,
}: CliUsersModalProps) {
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (isOpen) {
      setSearch('');
      onRefresh();
    }
  }, [isOpen, onRefresh]);

  if (!isOpen) return null;

  const filtered = users?.filter((u) => {
    if (!search) return true;
    const q = search.toLowerCase();
    return (u.username || '').toLowerCase().includes(q) || (u.email || '').toLowerCase().includes(q);
  }) || [];



  return (
    <div id="users-modal" className="modal-overlay active" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box" style={{ maxWidth: 780, width: '96%' }}>
        <div className="modal-header">
          <h3>CLI User Accounts (Firebase)</h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            Registered <code>dscli</code> accounts (Firebase Auth + Realtime Database). Ban
            blocks the user from launching the CLI; delete removes their profile record.
            Use the <strong>Notify</strong> button to send in-app notifications.
          </p>
          {/* Telegram support notice */}
          <div className="uda-help-box">
            <Send size={16} style={{ color: 'var(--primary)', flexShrink: 0 }} />
            <div style={{ flex: 1, fontSize: 12 }}>
              <strong style={{ color: 'var(--primary)' }}>Need help?</strong>{' '}
              Contact <a
                href={TELEGRAM_URL}
                target="_blank"
                rel="noopener noreferrer"
                style={{ color: 'var(--primary)', fontWeight: 600 }}
              >
                {TELEGRAM_LABEL}
              </a>
              <span style={{ color: 'var(--text-muted)' }}> — Telegram opens automatically.</span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 10 }}>
            <input
              type="text"
              id="users-search"
              placeholder="Search username / email…"
              style={{ flex: 1 }}
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                onSearch(e.target.value);
              }}
            />
            <button id="users-refresh" type="button" className="btn" onClick={onRefresh}>
              <RefreshCw size={13} /> Refresh
            </button>
          </div>
          <div id="users-list" style={{ maxHeight: '52vh', overflow: 'auto' }}>
            {loading && (
              <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-muted)' }}>
                <div className="uda-spinner" style={{ margin: '0 auto 10px' }} />
                Loading…
              </div>
            )}
            {error && (
              <div style={{ padding: 20, textAlign: 'center', color: 'var(--danger)' }}>
                <AlertTriangle size={16} style={{ verticalAlign: 'middle', marginRight: 6 }} />
                Error: {error}
              </div>
            )}
            {!loading && !error && filtered.length === 0 && (
              <div style={{ padding: 28, textAlign: 'center', color: 'var(--text-muted)' }}>
                <Info size={20} style={{ opacity: 0.6, marginBottom: 8 }} />
                <div>No CLI users found</div>
              </div>
            )}
            {filtered.map((u) => (
              <div
                key={u.uid}
                className="cli-user-row"
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    {u.username || '(no name)'}
                    {u.banned && (
                      <span className="badge-tag badge-tag-danger">
                        banned
                      </span>
                    )}
                    {u.disabled && (
                      <span className="badge-tag badge-tag-warning">
                        disabled
                      </span>
                    )}
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: 'var(--text-muted)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      fontFamily: 'var(--font-mono)',
                      marginTop: 2,
                    }}
                  >
                    {u.email || ''} ·{' '}
                    {u.email_verified || u.emailVerified ? (
                      <span style={{ color: 'var(--success)' }}>✓ verified</span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>unverified</span>
                    )}
                    {u.lastRefreshAt && (
                      <span> · {new Date(u.lastRefreshAt).toLocaleString()}</span>
                    )}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 6, flexShrink: 0 }}>
                  {onNotify && (
                    <button
                      type="button"
                      className="btn btn-icon"
                      style={{ padding: '6px 10px', fontSize: 11, color: 'var(--primary)' }}
                      onClick={() => onNotify(u.uid, u.username || u.email || u.uid)}
                      title={`Send notification to ${u.username || u.email || u.uid}`}
                    >
                      <Bell size={13} /> Notify
                    </button>
                  )}
                  {!u.banned ? (
                    <button
                      type="button"
                      className="btn"
                      style={{ padding: '6px 10px', fontSize: 11 }}
                      onClick={() => onBanToggle(u.uid, false)}
                      data-action="ban"
                    >
                      Ban
                    </button>
                  ) : (
                    <button
                      type="button"
                      className="btn"
                      style={{ padding: '6px 10px', fontSize: 11, color: 'var(--success)', borderColor: 'var(--success)' }}
                      onClick={() => onBanToggle(u.uid, true)}
                      data-action="unban"
                    >
                      <CheckCircle2 size={12} /> Unban
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-icon"
                    style={{ color: 'var(--danger)' }}
                    onClick={() => onDelete(u.uid)}
                    title="Delete user"
                  >
                    <X size={14} />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
        <div className="modal-footer">
          <button id="btn-close-users" type="button" className="btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
