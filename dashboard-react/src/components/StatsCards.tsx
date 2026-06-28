import { useMemo } from 'react';
import { Users, Cpu, Ban, Wrench } from 'lucide-react';
import type { UserRecord } from '../lib/types';
import { formatNumber, formatNumberShort } from '../lib/format';

interface StatsCardsProps {
  users: UserRecord[];
}

export function StatsCards({ users }: StatsCardsProps) {
  const stats = useMemo(() => {
    const totalUsers = users.length;
    const onlineUsers = users.filter((u) => u.status === 'online' && !u.banned).length;
    const bannedCount = users.filter((u) => u.banned).length;
    const bannedPct = totalUsers > 0 ? Math.round((bannedCount / totalUsers) * 100) : 0;
    const totalTokens = users.reduce((acc, u) => acc + (u.tokens?.total || 0), 0);
    const avgTokens = totalUsers > 0 ? totalTokens / totalUsers : 0;

    const toolCounts: Record<string, number> = {};
    users.forEach((user) => {
      if (!user.banned && user.last_tool && user.last_tool !== 'none' && user.last_tool !== 'initialization') {
        toolCounts[user.last_tool] = (toolCounts[user.last_tool] || 0) + (user.total_calls || 0);
      }
    });
    let topTool = '—';
    let topToolCount = 0;
    for (const t in toolCounts) {
      if (toolCounts[t] > topToolCount) {
        topToolCount = toolCounts[t];
        topTool = t;
      }
    }

    return { totalUsers, onlineUsers, bannedCount, bannedPct, totalTokens, avgTokens, topTool, topToolCount };
  }, [users]);

  return (
    <div className="stats-grid">
      <div className="stat-card">
        <div className="stat-icon" style={{ background: 'var(--primary-bg)', border: '1px solid var(--primary-border)' }}>
          <Users size={18} style={{ color: 'var(--primary)' }} />
        </div>
        <div className="stat-info">
          <div className="stat-label">Tracked IPs</div>
          <div className="stat-value">{stats.totalUsers}</div>
          <div className="stat-sub">{stats.onlineUsers} online</div>
        </div>
      </div>
      <div className="stat-card">
        <div className="stat-icon" style={{ background: 'rgba(167,139,250,0.1)', border: '1px solid rgba(167,139,250,0.2)' }}>
          <Cpu size={18} style={{ color: 'var(--accent)' }} />
        </div>
        <div className="stat-info">
          <div className="stat-label">Total Tokens</div>
          <div className="stat-value">{formatNumber(stats.totalTokens)}</div>
          <div className="stat-sub green">Avg: {formatNumberShort(stats.avgTokens)}/user</div>
        </div>
      </div>
      <div className="stat-card">
        <div className="stat-icon" style={{ background: 'var(--danger-glow)', border: '1px solid rgba(239,69,101,0.2)' }}>
          <Ban size={18} style={{ color: 'var(--danger)' }} />
        </div>
        <div className="stat-info">
          <div className="stat-label">Banned IPs</div>
          <div className="stat-value" style={{ color: 'var(--danger)' }}>{stats.bannedCount}</div>
          <div className="stat-sub">{stats.bannedPct}% of total</div>
        </div>
      </div>
      <div className="stat-card">
        <div className="stat-icon" style={{ background: 'var(--warning-glow)', border: '1px solid rgba(245,158,11,0.2)' }}>
          <Wrench size={18} style={{ color: 'var(--warning)' }} />
        </div>
        <div className="stat-info">
          <div className="stat-label">Top Tool</div>
          <div className="stat-value" style={{ fontSize: 17 }}>{stats.topTool}</div>
          <div className="stat-sub">{stats.topToolCount} calls</div>
        </div>
      </div>
    </div>
  );
}
