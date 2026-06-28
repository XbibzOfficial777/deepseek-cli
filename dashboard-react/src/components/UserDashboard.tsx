import { useState, useEffect, useRef, type FormEvent } from 'react';
import {
  LogIn,
  LogOut,
  User as UserIcon,
  Edit3,
  Check,
  X,
  Trash2,
  Send,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  KeyRound,
  UserPlus,
  Clock,
  Activity,
  Eye,
  EyeOff,
  Mail,
  Shield,
  Sparkles,
  ArrowRight,
  Bell,
  Inbox,
  Zap,
  Info,
  XCircle,
} from 'lucide-react';
import {
  signIn,
  signUp,
  signOut,
  resetPassword,
  onAuthChange,
  api,
} from '../lib/firebase';
import type { User } from 'firebase/auth';
import { useDeleteNotification } from '../hooks/useDashboardData';

type ViewMode = 'login' | 'signup' | 'forgot';

const TELEGRAM_URL = 'https://t.me/XbibzOfficial';

export function UserDashboard() {
  const userDeleteNotif = useDeleteNotification();
  const [user, setUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>('login');
  const [profile, setProfile] = useState<any>(null);
  const [stats, setStats] = useState<any>(null);
  const [notifs, setNotifs] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [editingUsername, setEditingUsername] = useState(false);
  const [newUsername, setNewUsername] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [dismissedAlerts, setDismissedAlerts] = useState<Set<string>>(new Set());

  // Subscribe to auth state
  useEffect(() => {
    const unsub = onAuthChange((u) => {
      setUser(u);
      setAuthLoading(false);
      if (u) {
        loadAllData(u);
      } else {
        setProfile(null);
        setStats(null);
        setNotifs(null);
        setDismissedAlerts(new Set());
      }
    });
    return unsub;
  }, []);

  async function loadAllData(u: User) {
    setLoading(true);
    setError(null);
    try {
      const [p, s, n] = await Promise.all([
        api.getProfile(u),
        api.getStats(u),
        api.getNotifications(u),
      ]);
      setProfile(p);
      setStats(s);
      setNotifs(n);
      if (!editingUsername) setNewUsername(p.username || '');
    } catch (e: any) {
      setError(e.message || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }

  async function handleLogin(email: string, password: string) {
    setLoading(true);
    setError(null);
    try {
      await signIn(email, password);
      setSuccess('Logged in successfully');
    } catch (e: any) {
      setError(e.message || 'Login failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleSignUp(email: string, password: string, username: string) {
    setLoading(true);
    setError(null);
    try {
      const u = await signUp(email, password);
      await signIn(email, password);
      try {
        await api.updateUsername(u, username);
      } catch (e) {
        // Username edit can be done later
      }
      setSuccess('Account created! You can now use this account in dscli.');
    } catch (e: any) {
      setError(e.message || 'Sign up failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleForgotPassword(email: string) {
    setLoading(true);
    setError(null);
    try {
      await resetPassword(email);
      setSuccess(`Password reset email sent to ${email}`);
      setViewMode('login');
    } catch (e: any) {
      setError(e.message || 'Failed to send reset email');
    } finally {
      setLoading(false);
    }
  }

  async function handleUpdateUsername() {
    if (!user) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.updateUsername(user, newUsername);
      setSuccess(result.message || 'Username updated');
      setEditingUsername(false);
      await loadAllData(user);
    } catch (e: any) {
      setError(e.message || 'Update failed');
    } finally {
      setLoading(false);
    }
  }

  async function handleDeleteAccount() {
    if (!user) return;
    setLoading(true);
    setError(null);
    try {
      await api.deleteAccount(user);
      setSuccess('Account deleted');
      await signOut();
    } catch (e: any) {
      setError(e.message || 'Delete failed');
    } finally {
      setLoading(false);
      setConfirmDelete(false);
    }
  }

  async function handleDeleteNotification(id: string) {
    if (!user) return;
    try {
      await userDeleteNotif.mutateAsync({ id });
      // Optimistically remove from local state
      setNotifs((prev: any) => {
        if (!prev?.notifications) return prev;
        return {
          ...prev,
          notifications: prev.notifications.filter((n: any) => n.id !== id),
        };
      });
    } catch (e: any) {
      setError(e.message || 'Failed to delete notification');
    }
  }

  function handleDismissAlert(key: string) {
    setDismissedAlerts((prev) => new Set(prev).add(key));
  }

  async function handleLogout() {
    await signOut();
  }

  if (authLoading) {
    return (
      <div className="uda-overlay">
        <div className="uda-loader">
          <div className="uda-loader-ring" />
          <div className="uda-loader-text">Loading your account…</div>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <AuthView
        mode={viewMode}
        setMode={setViewMode}
        onLogin={handleLogin}
        onSignUp={handleSignUp}
        onForgot={handleForgotPassword}
        loading={loading}
        error={error}
        success={success}
      />
    );
  }

  const displayName = profile?.username || (user.email || '').split('@')[0] || 'User';
  const hasBanner = (notifs?.current_status?.banned || notifs?.current_status?.limited) && !dismissedAlerts.has('system');

  return (
    <div className="user-dashboard">
      {/* Dismissible system banner for banned / limited */}
      {hasBanner && (
        <div className={`user-alert ${notifs.current_status.banned ? 'alert-danger' : 'alert-warning'}`}>
          <div className="user-alert-icon">
            {notifs.current_status.banned ? <AlertTriangle size={18} /> : <Zap size={18} />}
          </div>
          <div className="user-alert-body">
            <strong>{notifs.current_status.banned ? 'Account banned' : 'Token limit reached'}</strong>
            <div>
              {notifs.current_status.banned
                ? 'Your account has been suspended. Contact admin to restore access.'
                : 'You have exceeded your token limit. Contact admin to increase it.'}
            </div>
            <a href={TELEGRAM_URL} target="_blank" rel="noopener noreferrer" className="user-alert-link">
              <Send size={12} /> Contact @XbibzOfficial on Telegram
            </a>
          </div>
          <button className="user-alert-dismiss" onClick={() => handleDismissAlert('system')} title="Dismiss">
            <X size={14} />
          </button>
        </div>
      )}

      {/* Header */}
      <header className="user-header user-header-glass">
        <div className="user-header-left">
          <div className="user-avatar-big">
            <UserIcon size={22} color="white" />
          </div>
          <div>
            <div className="user-header-title">
              <span className="account-badge-inline">MY ACCOUNT</span>
              <span className="user-greeting">Welcome back,</span>
              <span className="user-name-highlight">{displayName}</span>
            </div>
            <div className="user-header-sub">{user.email} · {profile?.platform ? profile.platform : 'Web'}</div>
          </div>
        </div>
        <div className="user-header-right">
          <a href="/admin" className="btn btn-ghost" title="Go to Admin Console">
            <Shield size={13} /> Admin
          </a>
          <a href="/" className="btn btn-ghost" title="Back to landing">
            ← Home
          </a>
          <button type="button" className="btn btn-ghost" onClick={() => user && loadAllData(user)} disabled={loading}>
            <RefreshCw size={13} className={loading ? 'spinning' : ''} /> Refresh
          </button>
          <button type="button" className="btn btn-ghost" onClick={handleLogout}>
            <LogOut size={13} /> Logout
          </button>
        </div>
      </header>

      {/* Error/Success toasts */}
      {error && (
        <div className="user-toast user-toast-error">
          <XCircle size={14} /> {error}
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}
      {success && (
        <div className="user-toast user-toast-success">
          <CheckCircle2 size={14} /> {success}
          <button onClick={() => setSuccess(null)}>×</button>
        </div>
      )}

      <div className="user-grid user-grid-modern">
        {/* Profile card */}
        <div className="user-card user-card-glass">
          <div className="user-card-header">
            <UserIcon size={18} />
            <h3>Profile</h3>
          </div>
          <div className="user-card-body">
            <div className="user-field">
              <label>Email</label>
              <div className="user-field-value">{user.email}</div>
            </div>
            <div className="user-field">
              <label>Username</label>
              {editingUsername ? (
                <div className="user-edit-row">
                  <input
                    type="text"
                    value={newUsername}
                    onChange={(e) => setNewUsername(e.target.value)}
                    maxLength={32}
                    placeholder="e.g. myusername"
                    autoFocus
                  />
                  <button className="btn btn-primary" onClick={handleUpdateUsername} disabled={loading}>
                    <Check size={13} /> Save
                  </button>
                  <button className="btn btn-ghost" onClick={() => { setEditingUsername(false); setNewUsername(profile?.username || ''); }}>
                    <X size={13} /> Cancel
                  </button>
                </div>
              ) : (
                <div className="user-field-row">
                  <div className="user-field-value user-field-mono">{profile?.username || '—'}</div>
                  <button className="btn btn-icon btn-ghost" onClick={() => { setEditingUsername(true); setNewUsername(profile?.username || ''); }}>
                    <Edit3 size={13} />
                  </button>
                </div>
              )}
              <div className="user-field-hint">
                This username syncs to dscli on next launch.
              </div>
            </div>
            <div className="user-field">
              <label>Verified</label>
              <div className="user-field-value">
                {profile?.email_verified ? (
                  <span className="user-badge user-badge-success">
                    <CheckCircle2 size={12} /> verified
                  </span>
                ) : (
                  <span className="user-badge user-badge-warn">
                    <AlertTriangle size={12} /> unverified
                  </span>
                )}
              </div>
            </div>
            {profile?.created_at && (
              <div className="user-field">
                <label>Joined</label>
                <div className="user-field-value">{new Date(profile.created_at).toLocaleDateString()}</div>
              </div>
            )}
          </div>
        </div>

        {/* Stats card */}
        <div className="user-card user-card-glass">
          <div className="user-card-header">
            <Activity size={18} />
            <h3>Usage</h3>
          </div>
          <div className="user-card-body">
            <div className="user-stat-row">
              <div className="user-stat user-stat-glass">
                <div className="user-stat-label">Total tokens</div>
                <div className="user-stat-value">{(stats?.total_tokens || 0).toLocaleString()}</div>
              </div>
              <div className="user-stat user-stat-glass">
                <div className="user-stat-label">Total calls</div>
                <div className="user-stat-value">{(stats?.total_calls || 0).toLocaleString()}</div>
              </div>
            </div>
            <div className="user-field">
              <label>IPs used</label>
              <div className="user-field-value mono">
                {stats?.ips_used?.length > 0
                  ? stats.ips_used.map((ip: string) => (
                      <span key={ip} className="ip-badge">{ip}</span>
                    ))
                  : '—'}
              </div>
            </div>
            {stats?.recent_activity && stats.recent_activity.length > 0 && (
              <div className="user-field">
                <label>Recent activity</label>
                <div className="user-activity-list">
                  {stats.recent_activity.map((a: any, i: number) => (
                    <div key={i} className="user-activity-row">
                      <span className="ip-badge" style={{ fontSize: 10 }}>{a.ip}</span>
                      <span className="activity-time">
                        <Clock size={10} />{' '}
                        {a.last_online ? new Date(a.last_online).toLocaleString() : '—'}
                      </span>
                      <span className="activity-tokens">{(a.tokens || 0).toLocaleString()} tok</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Notifications card */}
        <div className="user-card user-card-glass user-card-wide">
          <div className="user-card-header">
            <Bell size={18} />
            <h3>Notifications</h3>
            <span className="user-count-badge">{notifs?.notifications?.length || 0}</span>
          </div>
          <div className="user-card-body">
            {!notifs?.notifications || notifs.notifications.length === 0 ? (
              <div className="user-empty-state">
                <Inbox size={28} className="user-empty-icon" />
                <div className="user-empty-title">No notifications</div>
                <div className="user-empty-sub">You’re all caught up.</div>
              </div>
            ) : (
              <div className="user-notif-list">
                {notifs.notifications.map((n: any) => (
                  <div key={n.id} className={`user-notif-item user-notif-${n.severity || 'info'}`}>
                    <div className="user-notif-meta">
                      <div className="user-notif-dot" />
                      <div className="user-notif-title">
                        {n.title}
                        {!n.read && <span className="user-notif-unread">New</span>}
                      </div>
                      <button
                        className="user-notif-delete"
                        onClick={() => handleDeleteNotification(n.id)}
                        title="Delete notification"
                        disabled={userDeleteNotif.isPending}
                      >
                        <Trash2 size={12} />
                      </button>
                    </div>
                    <div className="user-notif-msg">{n.message}</div>
                    <div className="user-notif-time">
                      <Clock size={10} />{' '}
                      {n.created_at ? new Date(n.created_at).toLocaleString() : '—'}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Help card */}
        <div className="user-card user-card-glass">
          <div className="user-card-header">
            <Info size={18} />
            <h3>Help</h3>
          </div>
          <div className="user-card-body">
            <div className="user-help-body">
              <div className="user-help-text">
                Need assistance? Contact admin directly via Telegram.
              </div>
              <a href={TELEGRAM_URL} target="_blank" rel="noopener noreferrer" className="user-help-btn">
                <Send size={13} /> Contact @XbibzOfficial
              </a>
            </div>
          </div>
        </div>

        {/* Danger zone */}
        <div className="user-card user-card-danger">
          <div className="user-card-header">
            <Trash2 size={18} />
            <h3>Delete Account</h3>
          </div>
          <div className="user-card-body">
            <div className="user-danger-text">
              Permanently delete your account, including Firebase Auth, RTDB profile, and all CLI usage records.
              This cannot be undone.
            </div>
            {!confirmDelete ? (
              <button className="btn btn-danger" onClick={() => setConfirmDelete(true)}>
                <Trash2 size={13} /> Delete my account
              </button>
            ) : (
              <div className="user-confirm-row">
                <span className="user-confirm-label">Are you sure? This is irreversible.</span>
                <button className="btn btn-ghost" onClick={() => setConfirmDelete(false)}>
                  <X size={13} /> Cancel
                </button>
                <button className="btn btn-danger" onClick={handleDeleteAccount} disabled={loading}>
                  <Trash2 size={13} /> Yes, delete permanently
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Auth view (polished, modern design) ───────────────────────────

function AuthView({
  mode,
  setMode,
  onLogin,
  onSignUp,
  onForgot,
  loading,
  error,
  success,
}: {
  mode: ViewMode;
  setMode: (m: ViewMode) => void;
  onLogin: (email: string, password: string) => void;
  onSignUp: (email: string, password: string, username: string) => void;
  onForgot: (email: string) => void;
  loading: boolean;
  error: string | null;
  success: string | null;
}) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [focused, setFocused] = useState<string | null>(null);
  const emailRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (mode === 'login') emailRef.current?.focus();
  }, [mode]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (loading) return;
    if (mode === 'login') onLogin(email, password);
    else if (mode === 'signup') onSignUp(email, password, username);
    else if (mode === 'forgot') onForgot(email);
  }

  const tabs = [
    { id: 'login' as const, label: 'Sign In', icon: LogIn },
    { id: 'signup' as const, label: 'Sign Up', icon: UserPlus },
    { id: 'forgot' as const, label: 'Reset', icon: KeyRound },
  ];

  const modeMeta = {
    login: {
      title: 'Welcome back',
      subtitle: 'Sign in with the same email + password you used to register in dscli.',
      buttonLabel: 'Sign In',
      buttonIcon: LogIn,
    },
    signup: {
      title: 'Create your account',
      subtitle: 'Use the same credentials later in dscli to sync your username.',
      buttonLabel: 'Create Account',
      buttonIcon: UserPlus,
    },
    forgot: {
      title: 'Reset password',
      subtitle: "We'll send a reset link to your email.",
      buttonLabel: 'Send Reset Link',
      buttonIcon: Send,
    },
  }[mode];

  return (
    <div className="uda-overlay">
      <div className="uda-bg-glow uda-bg-glow-1" />
      <div className="uda-bg-glow uda-bg-glow-2" />

      <div className="uda-card">
        {/* Animated gradient border */}
        <div className="uda-border-anim" />

        <div className="uda-card-body">
          {/* Logo + brand */}
          <div className="uda-brand">
            <div className="uda-logo">
              <Shield size={26} color="white" strokeWidth={2.4} />
            </div>
            <div className="uda-brand-text">
              <div className="uda-brand-title">
                <span className="uda-badge">{modeMeta.buttonLabel.toUpperCase()}</span>
                {' '}DeepSeek CLI · Account
              </div>
              <div className="uda-brand-sub">User Dashboard · Firebase Auth</div>
            </div>
          </div>

          {/* Mode tabs */}
          <div className="uda-tabs">
            {tabs.map((t) => {
              const Icon = t.icon;
              const active = mode === t.id;
              return (
                <button
                  type="button"
                  key={t.id}
                  className={`uda-tab ${active ? 'uda-tab-active' : ''}`}
                  onClick={() => setMode(t.id)}
                  disabled={loading}
                >
                  <Icon size={13} />
                  <span>{t.label}</span>
                </button>
              );
            })}
          </div>

          {/* Title + subtitle */}
          <h1 className="uda-title">{modeMeta.title}</h1>
          <p className="uda-subtitle">{modeMeta.subtitle}</p>

          {/* Alerts */}
          {error && (
            <div className="uda-alert uda-alert-error">
              <AlertTriangle size={15} />
              <span>{error}</span>
            </div>
          )}
          {success && (
            <div className="uda-alert uda-alert-success">
              <CheckCircle2 size={15} />
              <span>{success}</span>
            </div>
          )}

          {/* Form */}
          <form className="uda-form" onSubmit={handleSubmit} noValidate>
            {/* Email */}
            <div className={`uda-field ${focused === 'email' ? 'uda-field-focused' : ''}`}>
              <label htmlFor="uda-email">Email</label>
              <div className="uda-input-wrap">
                <Mail size={14} className="uda-input-icon" />
                <input
                  ref={emailRef}
                  id="uda-email"
                  type="email"
                  placeholder="you@example.com"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onFocus={() => setFocused('email')}
                  onBlur={() => setFocused(null)}
                  required
                  disabled={loading}
                />
              </div>
            </div>

            {/* Username (signup only) */}
            {mode === 'signup' && (
              <div className={`uda-field ${focused === 'username' ? 'uda-field-focused' : ''}`}>
                <label htmlFor="uda-username">Username (synced to dscli)</label>
                <div className="uda-input-wrap">
                  <UserIcon size={14} className="uda-input-icon" />
                  <input
                    id="uda-username"
                    type="text"
                    placeholder="e.g. myname or user@host"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    onFocus={() => setFocused('username')}
                    onBlur={() => setFocused(null)}
                    minLength={2}
                    maxLength={32}
                    pattern="[a-zA-Z0-9_@.\-]+"
                    required
                    disabled={loading}
                  />
                </div>
                <div className="uda-field-hint">
                  Letters, digits, _, @, ., – (2-32 chars)
                </div>
              </div>
            )}

            {/* Password (login + signup) */}
            {mode !== 'forgot' && (
              <div className={`uda-field ${focused === 'password' ? 'uda-field-focused' : ''}`}>
                <label htmlFor="uda-password">Password</label>
                <div className="uda-input-wrap">
                  <KeyRound size={14} className="uda-input-icon" />
                  <input
                    id="uda-password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="••••••••"
                    autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onFocus={() => setFocused('password')}
                    onBlur={() => setFocused(null)}
                    minLength={6}
                    required
                    disabled={loading}
                  />
                  <button
                    type="button"
                    className="uda-eye-toggle"
                    onClick={() => setShowPassword(!showPassword)}
                    tabIndex={-1}
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                </div>
                {mode === 'signup' && (
                  <div className="uda-field-hint">Minimum 6 characters</div>
                )}
              </div>
            )}

            {/* Submit */}
            <button type="submit" className="uda-submit" disabled={loading}>
              {loading ? (
                <>
                  <span className="uda-spinner" />
                  <span>Please wait…</span>
                </>
              ) : (
                <>
                  {(() => {
                    const Icon = modeMeta.buttonIcon;
                    return <Icon size={15} />;
                  })()}
                  <span>{modeMeta.buttonLabel}</span>
                  <ArrowRight size={14} />
                </>
              )}
            </button>
          </form>

          {/* Footer */}
          <div className="uda-foot">
            {mode === 'login' && (
              <span>
                Don't have an account?{' '}
                <button className="link-btn" onClick={() => setMode('signup')}>Create one</button>
              </span>
            )}
            {mode === 'signup' && (
              <span>
                Already have an account?{' '}
                <button className="link-btn" onClick={() => setMode('login')}>Sign in</button>
              </span>
            )}
            {mode === 'forgot' && (
              <span>
                Remember your password?{' '}
                <button className="link-btn" onClick={() => setMode('login')}>Back to sign in</button>
              </span>
            )}
          </div>

          {/* Help / Telegram */}
          <div className="uda-help">
            <span>Need help?</span>
            <a href={TELEGRAM_URL} target="_blank" rel="noopener noreferrer">
              <Send size={11} /> @XbibzOfficial on Telegram
            </a>
          </div>

          {/* Footer tagline */}
          <div className="uda-tagline">
            <Sparkles size={11} /> v7.7 · User Dashboard · Firebase Auth
          </div>
        </div>
      </div>
    </div>
  );
}
