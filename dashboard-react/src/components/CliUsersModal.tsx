import { useState, useEffect } from 'react';
import { X, RefreshCw } from 'lucide-react';
import type { CliUser } from '../lib/types';

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
      <div className="modal-box" style={{ maxWidth: 740, width: '96%' }}>
        <div className="modal-header">
          <h3>CLI User Accounts (Firebase)</h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p>
            Registered <code>dscli</code> accounts (Firebase Auth + Realtime Database). Ban
            blocks the user from launching the CLI; delete removes their profile record.
          </p>
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
          <div id="users-list" style={{ maxHeight: '50vh', overflow: 'auto' }}>
            {loading && (
              <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)' }}>
                Loading…
              </div>
            )}
            {error && (
              <div style={{ padding: 20, textAlign: 'center', color: 'var(--danger)' }}>
                Error: {error}
              </div>
            )}
            {!loading && !error && filtered.length === 0 && (
              <div style={{ padding: 20, textAlign: 'center', color: 'var(--text-muted)' }}>
                No CLI users found
              </div>
            )}
            {filtered.map((u) => (
              <div
                key={u.uid}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '10px 12px',
                  border: '1px solid var(--surface-border)',
                  borderRadius: 'var(--r-md)',
                  marginBottom: 8,
                  background: 'var(--surface)',
                }}
              >
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontWeight: 600, fontSize: 13, fontFamily: 'var(--font-mono)' }}>
                    {u.username || '(no name)'}
                    {u.banned && (
                      <span
                        style={{
                          color: 'var(--danger)',
                          fontSize: 10.5,
                          textTransform: 'uppercase',
                          letterSpacing: 0.4,
                          marginLeft: 6,
                        }}
                      >
                        · banned
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
                    }}
                  >
                    {u.email || ''} ·{' '}
                    {u.email_verified || u.emailVerified ? (
                      <span style={{ color: 'var(--success)' }}>✓ verified</span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)' }}>unverified</span>
                    )}
                  </div>
                </div>
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
                    style={{ padding: '6px 10px', fontSize: 11 }}
                    onClick={() => onBanToggle(u.uid, true)}
                    data-action="unban"
                  >
                    Unban
                  </button>
                )}
                <button
                  type="button"
                  className="btn btn-danger"
                  style={{ padding: '6px 10px', fontSize: 11 }}
                  onClick={() => onDelete(u.uid)}
                >
                  Delete
                </button>
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
