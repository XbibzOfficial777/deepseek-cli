import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Download,
  Database,
  RefreshCw,
  UserCog,
  GitBranch,
  KeyRound,
  LogOut,
} from 'lucide-react';
import { LoginOverlay } from './components/LoginOverlay';
import { AdminChip } from './components/AdminChip';
import { ThemeDots } from './components/ThemeDots';
import { StatsCards } from './components/StatsCards';
import { Charts } from './components/Charts';
import { FilterBar } from './components/FilterBar';
import { UserTable, type ActionKind } from './components/UserTable';
import { Drawer } from './components/Drawer';
import { ConfirmModal } from './components/ConfirmModal';
import { LimitModal } from './components/LimitModal';
import { PasscodeModal } from './components/PasscodeModal';
import { VersionModal } from './components/VersionModal';
import { CliUsersModal } from './components/CliUsersModal';
import { ToastProvider, useToast } from './components/Toast';
import {
  useDashboardData,
  useAdminAction,
  useChangePasscode,
  useVersionInfo,
  usePublishVersion,
  useCliUsers,
  useUserAction,
} from './hooks/useDashboardData';
import { getPasscode, setPasscode } from './api/client';
import { getStoredTheme } from './lib/theme';

import type {
  UserRecord,
  ThemeKey,
  StatusFilter,
  SortColumn,
  SortDirection,
  CliUser,
} from './lib/types';
import { applyThemeVars } from './lib/themeVars';

function DashboardInner() {
  const { showToast } = useToast();
  const [authed, setAuthed] = useState(!!getPasscode());
  const [theme, setTheme] = useState<ThemeKey>(getStoredTheme());
  const [polling, setPolling] = useState(true);
  const [pollInterval, setPollInterval] = useState(15000);
  const [lastSync, setLastSync] = useState<Date | null>(null);

  // Filter & sort state
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [sort, setSort] = useState<{ column: SortColumn; direction: SortDirection }>({
    column: 'tokens',
    direction: 'desc',
  });

  // Selected user for drawer
  const [selectedUser, setSelectedUser] = useState<UserRecord | null>(null);

  // Modal state
  const [confirmState, setConfirmState] = useState<{
    isOpen: boolean;
    title: string;
    message: string;
    confirmText: string;
    onConfirm: () => void;
  }>({ isOpen: false, title: '', message: '', confirmText: '', onConfirm: () => {} });

  const [limitUser, setLimitUser] = useState<UserRecord | null>(null);
  const [passcodeOpen, setPasscodeOpen] = useState(false);
  const [versionOpen, setVersionOpen] = useState(false);
  const [cliUsersOpen, setCliUsersOpen] = useState(false);

  // React Query hooks
  const dataQuery = useDashboardData();
  const adminAction = useAdminAction();
  const changePasscode = useChangePasscode();
  const versionQuery = useVersionInfo();
  const publishVersion = usePublishVersion();
  const cliUsersQuery = useCliUsers();
  const userAction = useUserAction();

  // Apply theme CSS variables on theme change
  useEffect(() => {
    applyThemeVars(theme);
  }, [theme]);

  // Handle auth errors
  useEffect(() => {
    if (dataQuery.error && (dataQuery.error as Error).message?.includes('Invalid passcode')) {
      setAuthed(false);
      setPasscode('');
    }
  }, [dataQuery.error]);

  // Polling
  useEffect(() => {
    if (!authed || !polling) return;
    const id = setInterval(() => {
      dataQuery.refetch().then(() => {
        setLastSync(new Date());
      }).catch(() => {});
    }, pollInterval);
    return () => clearInterval(id);
  }, [authed, polling, pollInterval, dataQuery]);

  // Sync handler
  const handleSync = useCallback(async () => {
    try {
      await dataQuery.refetch();
      setLastSync(new Date());
      showToast('Data loaded successfully', 'success');
    } catch (e) {
      showToast('Sync error: ' + (e as Error).message, 'error');
    }
  }, [dataQuery, showToast]);

  // Action handler
  const handleAction = useCallback((kind: ActionKind, username: string) => {
    const user = dataQuery.data?.find((u) => u.username === username);
    if (!user) {
      showToast('User not found in cache', 'error');
      return;
    }
    switch (kind) {
      case 'ban':
        setConfirmState({
          isOpen: true,
          title: user.banned ? 'Unban IP Address' : 'Ban IP Address',
          message: `Are you sure you want to ${user.banned ? 'unban' : 'ban'} IP ${user.ip}?`,
          confirmText: user.banned ? 'Restore Access' : 'Confirm Ban',
          onConfirm: () => handleAdminAction({ action: 'toggle_ban', username }),
        });
        break;
      case 'limit':
        setLimitUser(user);
        break;
      case 'delete':
        setConfirmState({
          isOpen: true,
          title: 'Delete IP Record',
          message: `Are you sure you want to delete the IP record for "${username}"? This will erase all token usage statistics and status logs.`,
          confirmText: 'Delete Record',
          onConfirm: () => handleAdminAction({ action: 'delete', username }),
        });
        break;
    }
  }, [dataQuery.data, showToast]);

  const handleAdminAction = useCallback(async (params: { action: 'toggle_ban' | 'delete' | 'update_limit'; username?: string; limit_val?: number }) => {
    showToast('Saving action to Cloudflare Worker...', 'info');
    try {
      await adminAction.mutateAsync(params);
      showToast('Action processed successfully', 'success');
    } catch (e) {
      showToast('Action Error: ' + (e as Error).message, 'error');
    }
  }, [adminAction, showToast]);

  // Filtered & sorted users
  const filteredUsers = useMemo(() => {
    if (!dataQuery.data) return [];
    let result = dataQuery.data.filter((user) => {
      const searchLower = search.toLowerCase();
      const matchesSearch =
        (user.username || '').toLowerCase().includes(searchLower) ||
        (user.ip || '').includes(search) ||
        (user.last_tool || '').toLowerCase().includes(searchLower);

      let matchesFilter = true;
      switch (statusFilter) {
        case 'banned':
          matchesFilter = user.banned === true;
          break;
        case 'limited':
          matchesFilter = !!(user.tokens?.limit && user.tokens.limit > 0 && user.tokens.total >= user.tokens.limit);
          break;
        case 'online':
        case 'away':
        case 'offline':
          matchesFilter = user.status === statusFilter && !user.banned;
          break;
        case 'all':
        default:
          matchesFilter = true;
      }
      return matchesSearch && matchesFilter;
    });

    result.sort((a, b) => {
      let valA: string | number = 0;
      let valB: string | number = 0;
      switch (sort.column) {
        case 'username':
          valA = (a.username || '').toLowerCase();
          valB = (b.username || '').toLowerCase();
          break;
        case 'ip':
          valA = a.ip || '';
          valB = b.ip || '';
          break;
        case 'status': {
          const priority: Record<string, number> = { online: 3, away: 2, offline: 1 };
          valA = a.banned ? 0 : priority[a.status] || 0;
          valB = b.banned ? 0 : priority[b.status] || 0;
          break;
        }
        case 'tokens':
          valA = a.tokens?.total || 0;
          valB = b.tokens?.total || 0;
          break;
        case 'last_tool':
          valA = (a.last_tool || '').toLowerCase();
          valB = (b.last_tool || '').toLowerCase();
          break;
        case 'total_calls':
          valA = a.total_calls || 0;
          valB = b.total_calls || 0;
          break;
        case 'last_online':
          valA = new Date(a.last_online || 0).getTime();
          valB = new Date(b.last_online || 0).getTime();
          break;
      }
      if (valA < valB) return sort.direction === 'asc' ? -1 : 1;
      if (valA > valB) return sort.direction === 'asc' ? 1 : -1;
      return 0;
    });

    return result;
  }, [dataQuery.data, search, statusFilter, sort]);

  // Export handlers
  const handleExport = (format: 'csv' | 'json') => {
    if (!dataQuery.data || dataQuery.data.length === 0) {
      showToast('No client data to export', 'error');
      return;
    }
    let content: string;
    let mimeType: string;
    let filename: string;

    if (format === 'json') {
      content = JSON.stringify(dataQuery.data, null, 2);
      mimeType = 'application/json';
      filename = `deepseek_client_report_${Date.now()}.json`;
    } else {
      const headers = ['Username', 'IP', 'Status', 'Tokens Input', 'Tokens Output', 'Tokens Total', 'Limit', 'Last Tool', 'Total Calls', 'Last Online', 'Version', 'Banned'];
      const rows = dataQuery.data.map((u) => [
        u.username || '',
        u.ip || '',
        u.banned ? 'banned' : (u.status || ''),
        u.tokens?.input || 0,
        u.tokens?.output || 0,
        u.tokens?.total || 0,
        u.tokens?.limit || 0,
        u.last_tool || '',
        u.total_calls || 0,
        u.last_online || '',
        u.version || 'Unknown',
        u.banned ? 'true' : 'false',
      ]);
      const csvRows = [headers.join(',')];
      rows.forEach((r) => {
        csvRows.push(r.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(','));
      });
      content = csvRows.join('\n');
      mimeType = 'text/csv';
      filename = `deepseek_client_report_${Date.now()}.csv`;
    }

    const blob = new Blob([content], { type: `${mimeType};charset=utf-8;` });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast(`Exported as ${format.toUpperCase()}`, 'success');
  };

  // Version modal handler
  const openVersionModal = async () => {
    setVersionOpen(true);
    try {
      await versionQuery.refetch();
    } catch (e) {
      showToast('Version load error: ' + (e as Error).message, 'error');
    }
  };

  const handlePublishVersion = async (version: string) => {
    showToast('Publishing version to registry Gist…', 'info');
    try {
      await publishVersion.mutateAsync(version);
      showToast(`Published v${version} — all clients will see the update`, 'success');
      setVersionOpen(false);
    } catch (e) {
      showToast('Error: ' + (e as Error).message, 'error');
    }
  };

  const handlePasscodeChange = async (newPasscode: string) => {
    try {
      await changePasscode.mutateAsync(newPasscode);
      setPasscode(newPasscode);
      showToast('Passcode updated', 'success');
      setPasscodeOpen(false);
    } catch (e) {
      showToast('Error: ' + (e as Error).message, 'error');
    }
  };

  const handleCliUserAction = async (action: 'ban' | 'unban' | 'delete', uid: string) => {
    try {
      await userAction.mutateAsync({ action, uid });
      showToast(`User ${action} done`, 'success');
    } catch (e) {
      showToast('Error: ' + (e as Error).message, 'error');
    }
  };

  const handleLogout = () => {
    setPasscode('');
    setAuthed(false);
    showToast('Dashboard locked', 'info');
  };

  // Show login if not authed
  if (!authed) {
    return <LoginOverlay onSuccess={() => setAuthed(true)} />;
  }

  const passcode = getPasscode();
  const users = filteredUsers;
  const isOnline = !!dataQuery.data;
  const isFetching = dataQuery.isFetching;

  return (
    <>
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />

      <div className="dash-container">
        <header className="header">
          <div className="header-left">
            <div className="logo-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <div className="header-titles">
              <div className="header-title">DeepSeek Admin Console</div>
              <div className="header-sub">
                Secure Gist Sync
                <span className="version-pill">v7.7 · React</span>
              </div>
            </div>
          </div>
          <div className="header-right">
            <ThemeDots active={theme} onChange={setTheme} />
            <div className="badge">
              <RefreshCw size={13} /> <span>Worker Sync</span>
            </div>
            <div className="poll-group">
              <button
                id="poll-toggle-btn"
                type="button"
                className={`poll-toggle ${polling ? 'on' : ''}`}
                onClick={() => setPolling(!polling)}
              >
                {polling ? 'Auto' : 'Manual'}
              </button>
              <select
                id="poll-interval"
                value={pollInterval}
                onChange={(e) => setPollInterval(Number(e.target.value))}
              >
                <option value={5000}>5s</option>
                <option value={15000}>15s</option>
                <option value={30000}>30s</option>
              </select>
            </div>
            <button id="export-csv-btn" type="button" className="btn" onClick={() => handleExport('csv')}>
              <Download size={13} /> CSV
            </button>
            <button id="export-json-btn" type="button" className="btn" onClick={() => handleExport('json')}>
              <Database size={13} /> JSON
            </button>
            <button id="sync-btn" type="button" className="btn btn-primary" onClick={handleSync} disabled={isFetching}>
              <RefreshCw size={13} className={isFetching ? 'spinning' : ''} /> Sync
            </button>
            <button id="users-btn" type="button" className="btn" onClick={() => setCliUsersOpen(true)}>
              <UserCog size={13} /> CLI Users
            </button>
            <button id="version-btn" type="button" className="btn" onClick={openVersionModal}>
              <GitBranch size={13} /> Version
            </button>
            <button id="passcode-btn" type="button" className="btn" onClick={() => setPasscodeOpen(true)}>
              <KeyRound size={13} /> Passcode
            </button>
            <AdminChip passcode={passcode} />
            <button id="logout-btn" type="button" className="btn" onClick={handleLogout}>
              <LogOut size={13} /> Lock
            </button>
          </div>
        </header>

        <StatsCards users={dataQuery.data || []} />
        <Charts users={dataQuery.data || []} theme={theme} />
        <FilterBar
          searchValue={search}
          onSearchChange={setSearch}
          statusFilter={statusFilter}
          onStatusFilterChange={setStatusFilter}
          sort={sort}
          onSortChange={setSort}
        />
        <UserTable
          users={users}
          onAction={handleAction}
          onRowClick={setSelectedUser}
        />

        <footer className="footer">
          <div className="network-status">
            <span className={`net-dot ${isFetching ? 'fetching' : isOnline ? 'online' : 'error'}`} />
            <span>
              {isFetching ? 'Syncing...' : isOnline ? 'Connected (Worker API)' : 'Connection Error'}
            </span>
          </div>
          <span>Last synced: {lastSync ? lastSync.toLocaleTimeString() : 'Never'}</span>
        </footer>
      </div>

      {/* Drawer */}
      <Drawer
        user={selectedUser}
        onClose={() => setSelectedUser(null)}
        onBan={(u) => {
          setSelectedUser(null);
          setConfirmState({
            isOpen: true,
            title: u.banned ? 'Unban IP Address' : 'Ban IP Address',
            message: `Are you sure you want to ${u.banned ? 'unban' : 'ban'} IP ${u.ip}?`,
            confirmText: u.banned ? 'Restore Access' : 'Confirm Ban',
            onConfirm: () => handleAdminAction({ action: 'toggle_ban', username: u.username }),
          });
        }}
        onLimit={(u) => {
          setSelectedUser(null);
          setLimitUser(u);
        }}
      />

      {/* Modals */}
      <ConfirmModal
        isOpen={confirmState.isOpen}
        title={confirmState.title}
        message={confirmState.message}
        confirmText={confirmState.confirmText}
        onConfirm={confirmState.onConfirm}
        onClose={() => setConfirmState((s) => ({ ...s, isOpen: false }))}
      />

      <LimitModal
        isOpen={!!limitUser}
        user={limitUser}
        onClose={() => setLimitUser(null)}
        onSave={(value) => {
          if (limitUser) {
            handleAdminAction({ action: 'update_limit', username: limitUser.username, limit_val: value });
            setLimitUser(null);
          }
        }}
      />

      <PasscodeModal
        isOpen={passcodeOpen}
        onClose={() => setPasscodeOpen(false)}
        onSave={handlePasscodeChange}
      />

      <VersionModal
        isOpen={versionOpen}
        currentVersion={versionQuery.data?.registry?.latest_version || null}
        apiUrl={versionQuery.data?.registry?.api_url || null}
        loading={versionQuery.isFetching}
        onClose={() => setVersionOpen(false)}
        onPublish={handlePublishVersion}
      />

      <CliUsersModal
        isOpen={cliUsersOpen}
        users={cliUsersQuery.data?.users as CliUser[] | undefined}
        loading={cliUsersQuery.isFetching}
        error={cliUsersQuery.error ? (cliUsersQuery.error as Error).message : null}
        onClose={() => setCliUsersOpen(false)}
        onRefresh={async () => { await cliUsersQuery.refetch(); }}
        onSearch={() => {}}
        onBanToggle={(uid, banned) => handleCliUserAction(banned ? 'unban' : 'ban', uid)}
        onDelete={(uid) => {
          setConfirmState({
            isOpen: true,
            title: 'Delete CLI User',
            message: "Remove this user's profile record? They can re-register later.",
            confirmText: 'Delete',
            onConfirm: () => handleCliUserAction('delete', uid),
          });
        }}
      />
    </>
  );
}

export default function App() {
  return (
    <ToastProvider>
      <DashboardInner />
    </ToastProvider>
  );
}
