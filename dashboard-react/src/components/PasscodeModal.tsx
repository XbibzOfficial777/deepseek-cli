import { useState, useEffect } from 'react';
import { X, Eye, EyeOff } from 'lucide-react';
import { getPasscodeStrength } from '../lib/format';

interface PasscodeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (newPasscode: string) => Promise<void> | void;
}

export function PasscodeModal({ isOpen, onClose, onSave }: PasscodeModalProps) {
  const [value, setValue] = useState('');
  const [showPassword, setShowPassword] = useState(false);

  useEffect(() => {
    if (isOpen) setValue('');
  }, [isOpen]);

  if (!isOpen) return null;

  const strength = getPasscodeStrength(value);
  const fillWidth = value.length > 0 ? (strength.score / 4) * 100 : 0;

  const handleSave = () => {
    if (!value) return;
    onSave(value);
  };

  return (
    <div className="modal-overlay active" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        <div className="modal-header">
          <h3>Change Admin Passcode</h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p>
            Enter a new passcode to protect this admin panel. This passcode will be dynamically
            updated in your private Gist database.
          </p>
          <div>
            <label>New Passcode</label>
            <div className="password-wrap">
              <input
                type={showPassword ? 'text' : 'password'}
                id="new-passcode-input"
                placeholder="Enter new passcode"
                value={value}
                onChange={(e) => setValue(e.target.value)}
                autoFocus
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            <div className="strength-bar">
              <div
                className="strength-fill"
                style={{ width: `${fillWidth}%`, background: strength.color }}
              />
            </div>
            <div className="strength-text" style={{ color: strength.color }}>
              {strength.text}
            </div>
          </div>
        </div>
        <div className="modal-footer">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="button" className="btn btn-primary" onClick={handleSave}>
            Update Passcode
          </button>
        </div>
      </div>
    </div>
  );
}
