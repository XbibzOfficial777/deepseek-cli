import { computeAdminIdentifier } from '../lib/format';

interface AdminChipProps {
  passcode: string;
}

export function AdminChip({ passcode }: AdminChipProps) {
  if (!passcode) return null;
  const identifier = computeAdminIdentifier(passcode);
  if (!identifier) return null;

  return (
    <div className="admin-chip">
      <div className="admin-chip-avatar">{identifier.substring(0, 2).toUpperCase()}</div>
      <div className="admin-chip-info">
        <div className="admin-chip-name">{identifier}</div>
        <div className="admin-chip-role">Network Admin</div>
      </div>
    </div>
  );
}
