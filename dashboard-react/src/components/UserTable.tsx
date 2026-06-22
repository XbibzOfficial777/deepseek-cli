import { Gavel, ShieldCheck, Edit3, Trash2, ShieldAlert } from 'lucide-react';
import type { UserRecord } from '../lib/types';
import { formatNumber, formatNumberShort, getRelativeTime } from '../lib/format';

export type ActionKind = 'ban' | 'limit' | 'delete';

interface UserTableProps {
  users: UserRecord[];
  onAction: (kind: ActionKind, username: string) => void;
  onRowClick: (user: UserRecord) => void;
}

export function UserTable({ users, onAction, onRowClick }: UserTableProps) {
  if (users.length === 0) {
    return (
      <div className="table-wrap">
        <div className="empty-state">
          <ShieldAlert size={28} />
          <p>No records match your search criteria</p>
        </div>
      </div>
    );
  }

  return (
    <div className="table-wrap">
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>User</th>
              <th>IP Address</th>
              <th>Status</th>
              <th>Tokens / Limit</th>
              <th>Last Tool</th>
              <th>Calls</th>
              <th>Last Active</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="table-body">
            {users.map((user) => (
              <UserRow
                key={user.username}
                user={user}
                onAction={onAction}
                onRowClick={onRowClick}
              />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface UserRowProps {
  user: UserRecord;
  onAction: (kind: ActionKind, username: string) => void;
  onRowClick: (user: UserRecord) => void;
}

function UserRow({ user, onAction, onRowClick }: UserRowProps) {
  const tokens = user.tokens || { input: 0, output: 0, total: 0, limit: 0 };
  const limit = tokens.limit || 0;
  const total = tokens.total || 1;

  const usagePct = limit > 0 ? (total / limit) * 100 : 0;
  let limitLabelClass = '';
  if (limit > 0) {
    if (usagePct >= 100) limitLabelClass = 'exceeded';
    else if (usagePct >= 85) limitLabelClass = 'warning';
  }

  const inPct = limit > 0 ? ((tokens.input || 0) / limit) * 100 : ((tokens.input || 0) / total) * 100;
  const outPct = limit > 0 ? ((tokens.output || 0) / limit) * 100 : ((tokens.output || 0) / total) * 100;

  const limitBadge = limit > 0 ? (
    <span className={`token-limit ${limitLabelClass}`}>
      {formatNumberShort(total)} / {formatNumberShort(limit)}
    </span>
  ) : (
    <span className="token-limit">{formatNumberShort(total)} / ∞</span>
  );

  const statusBadge = user.banned ? (
    <span className="status-badge">
      <span className="status-dot banned" /> banned
    </span>
  ) : (
    <span className="status-badge">
      <span className={`status-dot ${user.status}`} /> {user.status}
    </span>
  );

  const toolBadge = user.banned ? (
    <span className="tool-badge" style={{ opacity: 0.5 }}>—</span>
  ) : (
    <span className="tool-badge" title={user.last_tool}>{user.last_tool || '—'}</span>
  );

  const unameParts = (user.username || '').split('@');
  const displayName = unameParts[0] || user.username;
  const displayHost = unameParts[1] || user.hostname || '';

  return (
    <tr
      className={user.banned ? 'banned' : ''}
      onClick={(e) => {
        const target = e.target as HTMLElement;
        if (!target.closest('.row-actions')) onRowClick(user);
      }}
    >
      <td>
        <div className="user-cell">
          <div className="user-avatar">
            <img
              src={user.avatar_url || `https://api.dicebear.com/7.x/bottts/svg?seed=${encodeURIComponent(user.username)}`}
              alt={user.username}
              onError={(e) => {
                const img = e.currentTarget;
                img.style.display = 'none';
                img.parentElement!.textContent = (user.username || '').substring(0, 2).toUpperCase();
              }}
            />
          </div>
          <div className="user-info">
            <div className="user-name" title={user.username}>{displayName}</div>
            <div className="user-meta">
              {displayHost}
              <span className="dot" />
              v{user.version || '?'}
            </div>
          </div>
        </div>
      </td>
      <td>
        <span className="ip-badge">{user.ip}</span>
      </td>
      <td>{statusBadge}</td>
      <td>
        <div className="token-cell">
          <div className="token-row">
            <span className="token-total">{formatNumber(tokens.total)}</span>
            {limitBadge}
          </div>
          <div className="token-bar">
            <div className="bar-in" style={{ width: `${Math.min(inPct, 100)}%` }} />
            <div
              className={limitLabelClass === 'exceeded' ? 'bar-danger' : 'bar-out'}
              style={{ width: `${Math.min(outPct, 100)}%` }}
            />
          </div>
        </div>
      </td>
      <td>{toolBadge}</td>
      <td>
        <span className="call-badge">{user.banned ? 0 : user.total_calls || 0}</span>
      </td>
      <td>
        <span title={user.last_online ? new Date(user.last_online).toLocaleString() : ''}>
          {user.banned ? 'Blocked' : getRelativeTime(user.last_online)}
        </span>
      </td>
      <td>
        <div className="row-actions">
          <button
            type="button"
            className={`action-btn ${user.banned ? 'ban-active' : ''}`}
            data-action="ban"
            data-username={user.username}
            title={user.banned ? 'Unban IP' : 'Ban IP'}
            onClick={(e) => {
              e.stopPropagation();
              onAction('ban', user.username);
            }}
          >
            {user.banned ? <ShieldCheck size={13} /> : <Gavel size={13} />}
          </button>
          <button
            type="button"
            className={`action-btn ${limit > 0 ? 'limit-active' : ''}`}
            data-action="limit"
            data-username={user.username}
            title="Set Token Limit"
            onClick={(e) => {
              e.stopPropagation();
              onAction('limit', user.username);
            }}
          >
            <Edit3 size={13} />
          </button>
          <button
            type="button"
            className="action-btn"
            data-action="delete"
            data-username={user.username}
            title="Delete Record"
            onClick={(e) => {
              e.stopPropagation();
              onAction('delete', user.username);
            }}
          >
            <Trash2 size={13} />
          </button>
        </div>
      </td>
    </tr>
  );
}
