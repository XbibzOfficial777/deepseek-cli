import { Shield, User as UserIcon, ArrowRight, Send, Users, Ban, UploadCloud, Lock, Edit3, LineChart, Bell, Trash2 } from 'lucide-react';

interface LandingProps {
  onAdmin: () => void;
  onUser: () => void;
}

const TELEGRAM_URL = 'https://t.me/XbibzOfficial';

export function Landing({ onAdmin, onUser }: LandingProps) {
  return (
    <>
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow-2" />
      <div className="landing-container">
        <div className="landing-header">
          <div className="landing-logo">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <div>
            <div className="landing-title">DeepSeek CLI · Admin Console</div>
            <div className="landing-sub">v7.7 · Choose your dashboard</div>
          </div>
          <a
            href="https://github.com/XbibzOfficial777/deepseek-cli"
            target="_blank"
            rel="noopener noreferrer"
            className="btn"
            style={{ background: 'transparent', border: '1px solid var(--surface-border)' }}
          >
            ★ GitHub
          </a>
        </div>

        <div className="landing-grid">
          <button
            type="button"
            className="landing-card landing-card-admin"
            onClick={onAdmin}
          >
            <div className="landing-card-icon">
              <Shield size={36} />
            </div>
            <div className="landing-card-badge">ADMIN · PASSCODE</div>
            <div className="landing-card-title">Network Admin Console</div>
            <div className="landing-card-desc">
              Manage <strong>all</strong> CLI users — ban, unban, set token limits,
              publish updates, view global stats.
            </div>
            <div className="landing-card-features">
              <div><Users size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} /> All CLI users + IPs</div>
              <div><Ban size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} /> Ban / Unban (real Firebase Auth)</div>
              <div><UploadCloud size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} /> Update registry / publish version</div>
              <div><Lock size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} /> Passcode-gated access</div>
            </div>
            <div className="landing-card-cta">
              Enter Admin Console <ArrowRight size={14} />
            </div>
            <div className="landing-card-url">/admin</div>
          </button>

          <button
            type="button"
            className="landing-card landing-card-user"
            onClick={onUser}
          >
            <div className="landing-card-icon">
              <UserIcon size={36} />
            </div>
            <div className="landing-card-badge" style={{ background: 'rgba(56,139,253,0.18)', color: '#5b8def' }}>USER · FIREBASE AUTH</div>
            <div className="landing-card-title">My Account</div>
            <div className="landing-card-desc">
              Manage <strong>your own</strong> dscli account — same email/password
              you used to register in dscli.
            </div>
            <div className="landing-card-features">
              <div><Edit3 size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} /> Edit username (syncs to dscli)</div>
              <div><LineChart size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} /> Your token usage + history</div>
              <div><Bell size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} /> Ban / limit notifications</div>
              <div><Trash2 size={14} style={{ display: 'inline-block', verticalAlign: 'middle', marginRight: 6 }} /> Delete account</div>
            </div>
            <div className="landing-card-cta">
              Sign in to My Account <ArrowRight size={14} />
            </div>
            <div className="landing-card-url">/account</div>
          </button>
        </div>

        <div className="landing-footer">
          <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
            Need help? Contact{' '}
            <a
              href={TELEGRAM_URL}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'var(--primary)', fontWeight: 600 }}
            >
              <Send size={11} style={{ verticalAlign: 'middle', marginRight: 4 }} /> @XbibzOfficial on Telegram
            </a>
          </span>
        </div>
      </div>
    </>
  );
}
