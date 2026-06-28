import { useState, useEffect } from 'react';
import { X } from 'lucide-react';

interface VersionModalProps {
  isOpen: boolean;
  currentVersion: string | null;
  apiUrl: string | null;
  loading: boolean;
  onClose: () => void;
  onPublish: (version: string) => Promise<void> | void;
}

export function VersionModal({
  isOpen,
  currentVersion,
  apiUrl,
  loading,
  onClose,
  onPublish,
}: VersionModalProps) {
  const [value, setValue] = useState('');

  useEffect(() => {
    if (isOpen && currentVersion) {
      setValue(currentVersion);
    }
  }, [isOpen, currentVersion]);

  if (!isOpen) return null;

  const handlePublish = () => {
    const cleaned = value.trim().replace(/^v/i, '');
    if (!cleaned) return;
    onPublish(cleaned);
  };

  return (
    <div className="modal-overlay active" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        <div className="modal-header">
          <h3>Release Version Control</h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p>
            Set the latest released version. This is stored in the public registry Gist
            (<code>endpoint.json</code>) and every <code>dscli</code> client fetches it on
            startup — change it here and all clients instantly show <b>Update Available</b>.
            The <code>api_url</code> is preserved (merge-only, no duplicate Gists).
          </p>
          <div style={{ marginBottom: 12 }}>
            <label>Current Released Version</label>
            <div
              id="version-current"
              style={{ fontSize: 17, fontWeight: 700, color: 'var(--primary)', fontFamily: 'var(--font-heading)' }}
            >
              {loading ? 'Loading…' : currentVersion ? `v${currentVersion}` : '(not set)'}
            </div>
            {apiUrl && (
              <div
                id="version-apiurl"
                style={{ fontSize: 11, color: 'var(--text-muted)', wordBreak: 'break-all', fontFamily: 'var(--font-mono)' }}
              >
                api_url: {apiUrl}
              </div>
            )}
          </div>
          <div>
            <label htmlFor="version-input">New Latest Version</label>
            <input
              type="text"
              id="version-input"
              placeholder="e.g. 7.8 or 7.8.1"
              inputMode="decimal"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handlePublish()}
            />
            <div className="strength-text">
              Use a semantic version like 7.8 — must be numeric (digits and dots).
            </div>
          </div>
        </div>
        <div className="modal-footer">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" onClick={handlePublish} disabled={loading}>
            Publish Version
          </button>
        </div>
      </div>
    </div>
  );
}
