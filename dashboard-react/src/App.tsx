import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Download,
  Database,
  RefreshCw,
  UserCog,
  GitBranch,
  KeyRound,
  LogOut,
  User as UserIcon,
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
import { NotifyModal } from './components/NotifyModal';
import { UserDashboard } from './components/UserDashboard';
import { Landing } from './components/Landing';
import { ToastProvider, useToast } from './components/Toast';
import {
  useDashboardData,
  useAdminAction,
  useChangePasscode,
  useVersionInfo,
  usePublishVersion,
  useCliUsers,
  useUserAction,
  useSendNotification,
} from './hooks/useDashboardData';
import { getPasscode, setPasscode } from './api/client';
import { getStoredTheme } from './lib/theme';
import { getAdminDisplayName, getCycleCountdown } from './lib/format';

import type {
  UserRecord,
  ThemeKey,
  StatusFilter,
  SortColumn,
  SortDirection,
  CliUser,
} from './lib/types';
import { applyThemeVars } from './lib/themeVars';

// ── URL-based routing (separate admin vs user) ─────────────────────
function getRoute(): 'landing' | 'admin' | 'account' {
  const p = (typeof window !== 'undefined' ? window.location.pathname : '/').toLowerCase();
  if (p.startsWith('/admin')) return 'admin';
  if (p.startsWith('/account') || p.startsWith('/me') || p.startsWith('/user')) return 'account';
  return 'landing';
}

function navigate(to: string) {
  if (typeof window !== 'undefined') {
    window.history.pushState({}, '', to);
    window.dispatchEvent(new PopStateEvent('popstate'));
  }
}

// ── Admin-only console (isolated so hooks only run on /admin) ─────
function AdminConsole() {
  const { showToast } = useToast();
  const [theme, setTheme] = useState<ThemeKey>(getStoredTheme());
  const [polling, setPolling] = useState(true);
  const [pollInterval, setPollInterval] = useState(15000);
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [sort, setSort] = useState<{ column: SortColumn; direction: SortDirection }>({
    column: 'tokens',
    direction: 'desc',
  });
  const [selectedUser, setSelectedUser] = useState<UserRecord | null>(null);
  const [confirmState, setConfirmState] = useState<{
    isOpen: boolean; title: string; message: string; confirmText: string; onConfirm: () => void;
  }>({ isOpen: false, title: '', message: '', confirmText: '', onConfirm: () => {} });
  const [limitUser, setLimitUser] = useState<UserRecord | null>(null);
  const [passcodeOpen, setPasscodeOpen] = useState(false);
  const [versionOpen, setVersionOpen] = useState(false);
  const [cliUsersOpen, setCliUsersOpen] = useState(false);
  const [notifyOpen, setNotifyOpen] = useState(false);
  const [notifyTarget, setNotifyTarget] = useState<{ uid: string; name: string } | null>(null);

  const dataQuery = useDashboardData();
  const adminAction = useAdminAction();
  const changePasscode = useChangePasscode();
  const versionQuery = useVersionInfo();
  const publishVersion = usePublishVersion();
  const cliUsersQuery = useCliUsers();
  const userAction = useUserAction();
  const sendNotification = useSendNotification();

  useEffect(() => { applyThemeVars(theme); }, [theme]);
  useEffect(() => {
    if (!polling) return;
    const id = setInterval(() => {
      dataQuery.refetch().then(() => setLastSync(new Date())).catch(() => {});
    }, pollInterval);
    return () => clearInterval(id);
  }, [polling, pollInterval, dataQuery]);

  const handleSync = useCallback(async () => {
    try { await dataQuery.refetch(); setLastSync(new Date()); showToast('Data loaded', 'success'); }
    catch (e) { showToast('Sync error: ' + (e as Error).message, 'error'); }
  }, [dataQuery, showToast]);

  const handleAdminAction = useCallback(async (params: { action: 'toggle_ban' | 'delete' | 'update_limit'; username?: string; limit_val?: number }) => {
    showToast('Saving…', 'info');
    try { await adminAction.mutateAsync(params); showToast('Done', 'success'); }
    catch (e) { showToast('Error: ' + (e as Error).message, 'error'); }
  }, [adminAction, showToast]);

  const handleAction = useCallback((kind: ActionKind, username: string) => {
    const user = dataQuery.data?.find((u) => u.username === username);
    if (!user) { showToast('User not found', 'error'); return; }
    switch (kind) {
      case 'ban':
        setConfirmState({
          isOpen: true,
          title: user.banned ? 'Unban IP' : 'Ban IP',
          message: `Are you sure you want to ${user.banned ? 'unban' : 'ban'} IP ${user.ip}?`,
          confirmText: user.banned ? 'Restore' : 'Confirm',
          onConfirm: () => handleAdminAction({ action: 'toggle_ban', username }),
        }); break;
      case 'limit': setLimitUser(user); break;
      case 'delete':
        setConfirmState({
          isOpen: true, title: 'Delete IP Record',
          message: `Delete record for "${username}"? This erases all token stats.`,
          confirmText: 'Delete',
          onConfirm: () => handleAdminAction({ action: 'delete', username }),
        }); break;
    }
  }, [dataQuery.data, showToast, handleAdminAction]);

  const activeCycle = useMemo(() => {
    if (!dataQuery.data) return null;
    return dataQuery.data.filter((u) => !u.banned && u.cycle_start).sort(
      (a, b) => new Date(a.cycle_start!).getTime() - new Date(b.cycle_start!).getTime()
    )[0] || null;
  }, [dataQuery.data]);

  const filteredUsers = useMemo(() => {
    if (!dataQuery.data) return [];
    let result = dataQuery.data.filter((user) => {
      const s = search.toLowerCase();
      const ms = (user.username || '').toLowerCase().includes(s) || (user.ip || '').includes(search) || (user.last_tool || '').toLowerCase().includes(s);
      let mf = true;
      switch (statusFilter) {
        case 'banned': mf = user.banned === true; break;
        case 'limited': mf = !!(user.tokens?.limit && user.tokens.limit > 0 && user.tokens.total >= user.tokens.limit); break;
        case 'online': case 'away': case 'offline': mf = user.status === statusFilter && !user.banned; break;
        default: mf = true;
      }
      return ms && mf;
    });
    result.sort((a, b) => {
      let va: string | number = 0, vb: string | number = 0;
      switch (sort.column) {
        case 'username': va = (a.username || '').toLowerCase(); vb = (b.username || '').toLowerCase(); break;
        case 'ip': va = a.ip || ''; vb = b.ip || ''; break;
        case 'status': { const p: Record<string, number> = { online: 3, away: 2, offline: 1 }; va = a.banned ? 0 : p[a.status] || 0; vb = b.banned ? 0 : p[b.status] || 0; break; }
        case 'tokens': va = a.tokens?.total || 0; vb = b.tokens?.total || 0; break;
        case 'last_tool': va = (a.last_tool || '').toLowerCase(); vb = (b.last_tool || '').toLowerCase(); break;
        case 'total_calls': va = a.total_calls || 0; vb = b.total_calls || 0; break;
        case 'last_online': va = new Date(a.last_online || 0).getTime(); vb = new Date(b.last_online || 0).getTime(); break;
      }
      if (va < vb) return sort.direction === 'asc' ? -1 : 1;
      if (va > vb) return sort.direction === 'asc' ? 1 : -1;
      return 0;
    });
    return result;
  }, [dataQuery.data, search, statusFilter, sort]);

  const handleExport = (format: 'csv' | 'json') => {
    if (!dataQuery.data?.length) { showToast('No data', 'error'); return; }
    let content: string, mime: string, fn: string;
    if (format === 'json') { content = JSON.stringify(dataQuery.data, null, 2); mime = 'application/json'; fn = `report_${Date.now()}.json`; }
    else {
      const h = ['Username','IP','Status','Tokens Input','Tokens Output','Tokens Total','Limit','Last Tool','Total Calls','Last Online','Version','Banned'];
      const rows = dataQuery.data.map((u) => [u.username||'', u.ip||'', u.banned?'banned':(u.status||''), u.tokens?.input||0, u.tokens?.output||0, u.tokens?.total||0, u.tokens?.limit||0, u.last_tool||'', u.total_calls||0, u.last_online||'', u.version||'Unknown', u.banned?'true':'false']);
      const csv = [h.join(','), ...rows.map((r) => r.map((v) => `"${String(v).replace(/"/g,'""')}"`).join(','))].join('\n');
      content = csv; mime = 'text/csv'; fn = `report_${Date.now()}.csv`;
    }
    const blob = new Blob([content], { type: `${mime};charset=utf-8;` });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a'); a.href = url; a.download = fn; a.style.visibility = 'hidden'; document.body.appendChild(a); a.click(); document.body.removeChild(a);
    showToast(`Exported ${format.toUpperCase()}`, 'success');
  };

  const openVersionModal = async () => { setVersionOpen(true); try { await versionQuery.refetch(); } catch (e) { showToast('Version load error: ' + (e as Error).message, 'error'); } };
  const handlePublishVersion = async (version: string) => { showToast('Publishing…', 'info'); try { await publishVersion.mutateAsync(version); showToast(`Published v${version}`, 'success'); setVersionOpen(false); } catch (e) { showToast('Error: ' + (e as Error).message, 'error'); } };
  const handlePasscodeChange = async (np: string) => { try { await changePasscode.mutateAsync(np); setPasscode(np); showToast('Passcode updated', 'success'); setPasscodeOpen(false); } catch (e) { showToast('Error: ' + (e as Error).message, 'error'); } };
  const handleCliUserAction = async (action: 'ban'|'unban'|'delete', uid: string) => { try { await userAction.mutateAsync({ action, uid }); showToast(`User ${action} done`, 'success'); } catch (e) { showToast('Error: ' + (e as Error).message, 'error'); } };
  const handleOpenNotify = (uid: string, name: string) => { setNotifyTarget({ uid, name }); setNotifyOpen(true); };
  const handleSendNotify = async (params: { target_uid: string; title: string; message: string; type: string; severity: 'info'|'warning'|'danger'|'success' }) => { try { await sendNotification.mutateAsync(params); showToast('Notification sent', 'success'); setNotifyOpen(false); } catch (e) { showToast('Error: ' + (e as Error).message, 'error'); } };
  const handleLogout = () => { setPasscode(''); showToast('Locked', 'info'); window.location.reload(); };

  const passcode = getPasscode();
  const isOnline = !!dataQuery.data;
  const isFetching = dataQuery.isFetching;
  const adminName = getAdminDisplayName(passcode);
  const cycleCountdown = getCycleCountdown(activeCycle?.cycle_start);

  if (!passcode) return <LoginOverlay onSuccess={() => window.location.reload()} />;

  return (
    <>
      <div className="bg-glow bg-glow-1" />
      <div className="bg-glow bg-glow-2" />
      <div className="dash-container">
        <header className="header">
          <div className="header-left">
            <div className="logo-icon">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /></svg>
            </div>
            <div className="header-titles">
              <div className="header-title">DeepSeek Admin Console<span className="version-pill">v7.7</span></div>
              <div className="header-sub">Signed in as <span className="header-admin-name">{adminName}</span><span className="header-sub-sep">·</span><span>Secure Gist Sync</span></div>
            </div>
          </div>
          <div className="header-right">
            <ThemeDots active={theme} onChange={setTheme} />
            <div className="header-divider" />
            <div className="poll-group">
              <button type="button" className={`poll-toggle ${polling ? 'on' : ''}`} onClick={() => setPolling(!polling)}>{polling ? 'Auto' : 'Manual'}</button>
              <select value={pollInterval} onChange={(e) => setPollInterval(Number(e.target.value))}><option value={5000}>5s</option><option value={15000}>15s</option><option value={30000}>30s</option></select>
            </div>
            <div className="header-divider" />
            <button type="button" className="btn" onClick={() => handleExport('csv')}><Download size={13} /> CSV</button>
            <button type="button" className="btn" onClick={() => handleExport('json')}><Database size={13} /> JSON</button>
            <div className="header-divider" />
            <button type="button" className="btn btn-primary" onClick={handleSync} disabled={isFetching}><RefreshCw size={13} className={isFetching ? 'spinning' : ''} /> Sync</button>
            <div className="header-divider" />
            <button type="button" className="btn" onClick={() => setCliUsersOpen(true)}><UserCog size={13} /> CLI Users</button>
            <button type="button" className="btn" onClick={openVersionModal}><GitBranch size={13} /> Version</button>
            <button type="button" className="btn" onClick={() => setPasscodeOpen(true)}><KeyRound size={13} /> Passcode</button>
            <div className="header-divider" />
            <AdminChip passcode={passcode} />
            <a href="/account" className="btn"><UserIcon size={13} /> My Account</a>
            <a href="/" className="btn">← Home</a>
            <button type="button" className="btn btn-icon" onClick={handleLogout}><LogOut size={13} /></button>
          </div>
        </header>
        <StatsCards users={dataQuery.data || []} />
        <Charts users={dataQuery.data || []} theme={theme} />
        <FilterBar searchValue={search} onSearchChange={setSearch} statusFilter={statusFilter} onStatusFilterChange={setStatusFilter} sort={sort} onSortChange={setSort} />
        <UserTable users={filteredUsers} onAction={handleAction} onRowClick={setSelectedUser} />
        <footer className="footer">
          <div className="network-status">
            <span className={`net-dot ${isFetching ? 'fetching' : isOnline ? 'online' : 'error'}`} />
            <span>{isFetching ? 'Syncing…' : isOnline ? 'Connected · Worker API' : 'Connection Error'}</span>
            {isOnline && <><span className="footer-sep">·</span><span className="footer-cycle">Cycle resets in {cycleCountdown}</span></>}
          </div>
          <span className="footer-sync">Last synced {lastSync ? lastSync.toLocaleTimeString() : '—'}</span>
        </footer>
      </div>
      <Drawer user={selectedUser} onClose={() => setSelectedUser(null)} onBan={(u) => { setSelectedUser(null); setConfirmState({ isOpen: true, title: u.banned ? 'Unban IP' : 'Ban IP', message: `${u.banned ? 'Unban' : 'Ban'} IP ${u.ip}?`, confirmText: u.banned ? 'Restore' : 'Confirm', onConfirm: () => handleAdminAction({ action: 'toggle_ban', username: u.username }) }); }} onLimit={(u) => { setSelectedUser(null); setLimitUser(u); }} />
      <ConfirmModal isOpen={confirmState.isOpen} title={confirmState.title} message={confirmState.message} confirmText={confirmState.confirmText} onConfirm={confirmState.onConfirm} onClose={() => setConfirmState((s) => ({ ...s, isOpen: false }))} />
      <LimitModal isOpen={!!limitUser} user={limitUser} onClose={() => setLimitUser(null)} onSave={(v) => { if (limitUser) { handleAdminAction({ action: 'update_limit', username: limitUser.username, limit_val: v }); setLimitUser(null); } }} />
      <PasscodeModal isOpen={passcodeOpen} onClose={() => setPasscodeOpen(false)} onSave={handlePasscodeChange} />
      <VersionModal isOpen={versionOpen} currentVersion={versionQuery.data?.registry?.latest_version || null} apiUrl={versionQuery.data?.registry?.api_url || null} loading={versionQuery.isFetching} onClose={() => setVersionOpen(false)} onPublish={handlePublishVersion} />
      <CliUsersModal isOpen={cliUsersOpen} users={cliUsersQuery.data?.users as CliUser[] | undefined} loading={cliUsersQuery.isFetching} error={cliUsersQuery.error ? (cliUsersQuery.error as Error).message : null} onClose={() => setCliUsersOpen(false)} onRefresh={async () => { await cliUsersQuery.refetch(); }} onSearch={() => {}} onBanToggle={(uid, banned) => handleCliUserAction(banned ? 'unban' : 'ban', uid)} onDelete={(uid) => { setConfirmState({ isOpen: true, title: 'Delete CLI User', message: "Remove this user's profile? They can re-register later.", confirmText: 'Delete', onConfirm: () => handleCliUserAction('delete', uid) }); }} onNotify={handleOpenNotify} />
      {notifyTarget && <NotifyModal isOpen={notifyOpen} targetUid={notifyTarget.uid} targetName={notifyTarget.name} onClose={() => setNotifyOpen(false)} onSend={handleSendNotify} loading={sendNotification.isPending} />}
    </>
  );
}

// ── Router shell (no admin hooks here!) ───────────────────────────
function DashboardInner() {
  const [route, setRoute] = useState(getRoute());
  useEffect(() => {
    const onPop = () => setRoute(getRoute());
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  if (route === 'landing') return <Landing onAdmin={() => { navigate('/admin'); setRoute('admin'); }} onUser={() => { navigate('/account'); setRoute('account'); }} />;
  if (route === 'account') {
    return (
      <div className="account-route">
        <div className="bg-glow bg-glow-1" />
        <div className="bg-glow-2" />
        <div className="dash-container account-container">
          <UserDashboard />
        </div>
      </div>
    );
  }
  return <AdminConsole />;
}

export default function App() {
  return (
    <ToastProvider>
      <DashboardInner />
    </ToastProvider>
  );
}
