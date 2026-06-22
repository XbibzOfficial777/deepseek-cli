import { useState, useRef, useEffect } from 'react';
import { Shield, Eye, EyeOff, Loader2 } from 'lucide-react';
import { setPasscode } from '../api/client';
import { api } from '../api/client';
import { useToast } from './Toast';

interface LoginOverlayProps {
  onSuccess: () => void;
}

export function LoginOverlay({ onSuccess }: LoginOverlayProps) {
  const [passcode, setPasscodeState] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [shake, setShake] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { showToast } = useToast();

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleLogin = async () => {
    if (!passcode.trim()) {
      triggerShake();
      showToast('Please enter passcode', 'error');
      return;
    }

    setLoading(true);
    setPasscode(passcode.trim());

    try {
      await api.getData();
      onSuccess();
    } catch {
      setPasscode('');
      triggerShake();
      showToast('Invalid passcode', 'error');
    } finally {
      setLoading(false);
    }
  };

  const triggerShake = () => {
    setShake(true);
    setTimeout(() => setShake(false), 500);
  };

  return (
    <div className="login-overlay">
      <div className={`login-box ${shake ? 'shake' : ''}`}>
        <div className="login-shield">
          <Shield size={26} style={{ color: 'var(--primary)' }} />
        </div>
        <div className="login-title">DeepSeek Admin Console</div>
        <div className="login-desc">Secure access for authorized network administrators only.</div>
        <form
          className="login-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleLogin();
          }}
        >
          <div className="form-group">
            <label htmlFor="passcode-input">Admin Passcode</label>
            <div className="password-wrap">
              <input
                ref={inputRef}
                type={showPassword ? 'text' : 'password'}
                id="passcode-input"
                placeholder="••••••••"
                autoComplete="off"
                value={passcode}
                onChange={(e) => setPasscodeState(e.target.value)}
                disabled={loading}
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? 'Hide password' : 'Show password'}
              >
                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
          </div>
          <button type="submit" className="btn-unlock" disabled={loading}>
            {loading ? (
              <>
                <Loader2 size={14} className="spinning" /> Verifying...
              </>
            ) : (
              'Unlock Console'
            )}
          </button>
        </form>
        <div className="login-foot">v7.7 · Cloudflare Worker · Gist-Synced</div>
      </div>
    </div>
  );
}
