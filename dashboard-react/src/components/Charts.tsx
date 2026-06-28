import { useMemo } from 'react';
import type { UserRecord } from '../lib/types';
import { formatNumber } from '../lib/format';
import { THEMES } from '../lib/theme';
import type { ThemeKey } from '../lib/types';

interface ChartsProps {
  users: UserRecord[];
  theme: ThemeKey;
}

export function Charts({ users, theme }: ChartsProps) {
  return (
    <div className="analytics-grid">
      <ToolChart users={users} theme={theme} />
      <TokenChart users={users} />
      <StatusChart users={users} />
    </div>
  );
}

function ToolChart({ users, theme }: { users: UserRecord[]; theme: ThemeKey }) {
  const toolStats = useMemo(() => {
    const counts: Record<string, number> = {};
    users.forEach((u) => {
      if (u.last_tool && u.last_tool !== 'none' && u.last_tool !== 'initialization') {
        counts[u.last_tool] = (counts[u.last_tool] || 0) + (u.total_calls || 0);
      }
    });
    const stats = Object.entries(counts)
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 4);
    return stats;
  }, [users]);

  if (toolStats.length === 0) {
    return (
      <div className="analytics-card">
        <div className="analytics-title">Top Tool Allocations</div>
        <div className="chart-empty">No tool calls recorded yet</div>
      </div>
    );
  }

  const maxCalls = Math.max(...toolStats.map((t) => t.count));
  const t = THEMES[theme];

  return (
    <div className="analytics-card">
      <div className="analytics-title">Top Tool Allocations</div>
      {toolStats.map((ts) => (
        <div key={ts.name} className="chart-row">
          <div className="chart-label" title={ts.name}>{ts.name}</div>
          <div className="chart-track">
            <div
              className="chart-fill"
              style={{
                width: `${(ts.count / maxCalls * 100).toFixed(1)}%`,
                background: `linear-gradient(90deg, ${t.primary}, ${t.accent})`,
              }}
            />
          </div>
          <div className="chart-val">{ts.count}</div>
        </div>
      ))}
    </div>
  );
}

function TokenChart({ users }: { users: UserRecord[] }) {
  const { totalInput, totalOutput } = useMemo(() => {
    return users.reduce(
      (acc, u) => ({
        totalInput: acc.totalInput + (u.tokens?.input || 0),
        totalOutput: acc.totalOutput + (u.tokens?.output || 0),
      }),
      { totalInput: 0, totalOutput: 0 }
    );
  }, [users]);

  const grandTotal = totalInput + totalOutput || 1;
  const inPct = Math.round((totalInput / grandTotal) * 100);
  const outPct = Math.round((totalOutput / grandTotal) * 100);

  return (
    <div className="analytics-card">
      <div className="analytics-title">Network Token Allocation</div>
      <div className="status-counts">
        <span style={{ color: 'var(--accent)', fontWeight: 600 }}>Input {inPct}%</span>
        <span style={{ color: 'var(--primary)', fontWeight: 600 }}>Output {outPct}%</span>
      </div>
      <div className="split-bar">
        <div style={{ flex: totalInput || 1, background: 'var(--accent)', minWidth: 2 }} />
        <div style={{ flex: totalOutput || 1, background: 'var(--primary)', minWidth: 2 }} />
      </div>
      <div className="split-labels">
        <span>{formatNumber(totalInput)} tokens</span>
        <span>{formatNumber(totalOutput)} tokens</span>
      </div>
    </div>
  );
}

function StatusChart({ users }: { users: UserRecord[] }) {
  const { online, away, offline, banned } = useMemo(() => {
    return {
      online: users.filter((u) => u.status === 'online' && !u.banned).length,
      away: users.filter((u) => u.status === 'away' && !u.banned).length,
      offline: users.filter((u) => u.status === 'offline' && !u.banned).length,
      banned: users.filter((u) => u.banned).length,
    };
  }, [users]);

  return (
    <div className="analytics-card">
      <div className="analytics-title">Client Status Allocation</div>
      <div className="status-counts">
        <span style={{ color: 'var(--success)' }}>Online ({online})</span>
        <span style={{ color: 'var(--warning)' }}>Away ({away})</span>
        <span style={{ color: 'var(--text-muted)' }}>Offline ({offline})</span>
        <span style={{ color: 'var(--danger)' }}>Banned ({banned})</span>
      </div>
      <div className="split-bar">
        <div style={{ flex: online || 0.001, background: 'var(--success)', minWidth: 2 }} />
        <div style={{ flex: away || 0.001, background: 'var(--warning)', minWidth: 2 }} />
        <div style={{ flex: offline || 0.001, background: 'var(--text-muted)', minWidth: 2 }} />
        <div style={{ flex: banned || 0.001, background: 'var(--danger)', minWidth: 2 }} />
      </div>
      <div style={{ color: 'var(--text-muted)', fontSize: 11, textAlign: 'center', marginTop: 6 }}>
        Tracked clients: {users.length}
      </div>
    </div>
  );
}
