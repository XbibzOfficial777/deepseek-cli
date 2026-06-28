import { useState } from 'react';
import { X, Bell, Send, AlertTriangle, Info, CheckCircle2, AlertCircle } from 'lucide-react';

interface NotifyModalProps {
  isOpen: boolean;
  targetUid: string;
  targetName: string;
  onClose: () => void;
  onSend: (params: { target_uid: string; title: string; message: string; type: string; severity: 'info' | 'warning' | 'danger' | 'success' }) => void;
  loading: boolean;
}

const severityMeta = {
  info: { icon: Info, color: 'var(--primary)', border: 'rgba(91,141,239,0.35)', bg: 'rgba(91,141,239,0.08)' },
  success: { icon: CheckCircle2, color: 'var(--success)', border: 'rgba(34,197,94,0.35)', bg: 'rgba(34,197,94,0.08)' },
  warning: { icon: AlertTriangle, color: 'var(--warning)', border: 'rgba(245,158,11,0.35)', bg: 'rgba(245,158,11,0.08)' },
  danger: { icon: AlertCircle, color: 'var(--danger)', border: 'rgba(239,69,101,0.35)', bg: 'rgba(239,69,101,0.08)' },
};

export function NotifyModal({ isOpen, targetUid, targetName, onClose, onSend, loading }: NotifyModalProps) {
  const [title, setTitle] = useState('');
  const [message, setMessage] = useState('');
  const [severity, setSeverity] = useState<'info' | 'warning' | 'danger' | 'success'>('info');

  if (!isOpen) return null;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!title.trim() || loading) return;
    onSend({ target_uid: targetUid, title: title.trim(), message: message.trim(), type: 'admin', severity });
  }

  return (
    <div className="modal-overlay active" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box" style={{ maxWidth: 520, width: '96%' }}>
        <div className="modal-header">
          <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Bell size={16} style={{ color: 'var(--primary)' }} />
            Send Notification
          </h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 8 }}>
            Target: <span style={{ color: 'var(--text)', fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{targetName}</span>
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                Title
              </label>
              <input
                type="text"
                placeholder="e.g. System maintenance"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                maxLength={120}
                required
                autoFocus
                style={{ width: '100%' }}
              />
            </div>

            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                Message
              </label>
              <textarea
                placeholder="Enter your notification message…"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                rows={4}
                maxLength={800}
                style={{
                  width: '100%',
                  padding: 10,
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--surface-border)',
                  borderRadius: 'var(--r-md)',
                  color: 'var(--text)',
                  fontFamily: 'inherit',
                  fontSize: 13,
                  resize: 'vertical',
                  outline: 'none',
                }}
                onFocus={(e) => { (e.target as HTMLTextAreaElement).style.borderColor = 'var(--primary)'; }}
                onBlur={(e) => { (e.target as HTMLTextAreaElement).style.borderColor = 'var(--surface-border)'; }}
              />
              <div style={{ fontSize: 10, color: 'var(--text-muted)', textAlign: 'right', marginTop: 4 }}>
                {message.length} / 800
              </div>
            </div>

            <div>
              <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                Severity
              </label>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {(Object.keys(severityMeta) as Array<'info'|'success'|'warning'|'danger'>).map((s) => {
                  const m = severityMeta[s];
                  const SIcon = m.icon;
                  const active = severity === s;
                  return (
                    <button
                      key={s}
                      type="button"
                      onClick={() => setSeverity(s)}
                      className="btn"
                      style={{
                        flex: 1,
                        minWidth: 90,
                        justifyContent: 'center',
                        gap: 6,
                        borderColor: active ? m.border : 'var(--surface-border)',
                        background: active ? m.bg : 'var(--surface)',
                        color: active ? m.color : 'var(--text-muted)',
                        fontWeight: 600,
                        textTransform: 'capitalize',
                        fontSize: 12,
                      }}
                    >
                      <SIcon size={13} />
                      {s}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="modal-footer" style={{ margin: '0 -20px -20px', paddingTop: 14 }}>
              <button type="button" className="btn" onClick={onClose} disabled={loading}>
                Cancel
              </button>
              <button type="submit" className="btn btn-primary" disabled={loading || !title.trim()}>
                {loading ? (
                  <>
                    <span className="uda-spinner" />
                    Sending…
                  </>
                ) : (
                  <>
                    <Send size={13} /> Send Notification
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
