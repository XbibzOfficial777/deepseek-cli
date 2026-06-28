import { ShieldCheck } from 'lucide-react';
import { getAdminDisplayName, getAdminAvatarUrl, computeAdminIdentifier } from '../lib/format';

interface AdminChipProps {
  passcode: string;
}

export function AdminChip({ passcode }: AdminChipProps) {
  if (!passcode) return null;
  const identifier = computeAdminIdentifier(passcode);
  if (!identifier) return null;

  const displayName = getAdminDisplayName(passcode);
  const avatarUrl = getAdminAvatarUrl(passcode);

  return (
    <div className="admin-chip" title={`Session ID: ${identifier}`}>
      <div className="admin-chip-avatar">
        <img
          src={avatarUrl}
          alt="admin avatar"
          loading="lazy"
          onError={(e) => {
            const img = e.currentTarget;
            img.style.display = 'none';
            const parent = img.parentElement;
            if (parent) parent.textContent = displayName.slice(-2).toUpperCase();
          }}
        />
      </div>
      <div className="admin-chip-info">
        <div className="admin-chip-name">{displayName}</div>
        <div className="admin-chip-role">
          <ShieldCheck size={9} style={{ verticalAlign: 'middle', marginRight: 3 }} />
          Network Admin
        </div>
      </div>
    </div>
  );
}
