import { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import type { UserRecord } from '../lib/types';

interface LimitModalProps {
  user: UserRecord | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (limitValue: number) => void;
}

export function LimitModal({ user, isOpen, onClose, onSave }: LimitModalProps) {
  const [value, setValue] = useState('');

  useEffect(() => {
    if (user && isOpen) {
      setValue(user.tokens?.limit ? String(user.tokens.limit) : '');
    }
  }, [user, isOpen]);

  if (!isOpen || !user) return null;

  const handleSave = () => {
    const num = parseInt(value) || 0;
    onSave(num);
  };

  return (
    <div id="limit-modal" className="modal-overlay active" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        <div className="modal-header">
          <h3>Set IP Token Limit</h3>
          <button id="limit-modal-close" type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p>
            Configure the maximum tokens allowed for <strong id="limit-user-label">{user.username}</strong> (<span id="limit-ip-label">{user.ip}</span>).
          </p>
          <div>
            <label htmlFor="limit-value-input">Token Limit</label>
            <input
              type="number"
              id="limit-value-input"
              placeholder="e.g. 500000"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSave()}
              autoFocus
            />
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 5 }}>
              Set to 0 or leave empty for unlimited usage.
            </div>
          </div>
        </div>
        <div className="modal-footer">
          <button id="btn-cancel-limit" type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button id="btn-save-limit" type="button" className="btn btn-primary" onClick={handleSave}>
            Save Limit
          </button>
        </div>
      </div>
    </div>
  );
}
