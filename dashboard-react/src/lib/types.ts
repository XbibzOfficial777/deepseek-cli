// ── Types matching Worker API ────────────────────────────────────

export interface UserTokens {
  input: number;
  output: number;
  total: number;
  limit: number;
}

export interface UserRecord {
  username: string;
  ip: string;
  avatar_url?: string;
  tokens: UserTokens;
  last_tool: string;
  last_online: string;
  status: 'online' | 'away' | 'offline';
  total_calls: number;
  banned: boolean;
  cycle_start: string;
  version: string;
  hostname: string;
  platform: string;
  arch: string;
  os_release: string;
  device_name: string;
}

export interface ApiCheckResponse {
  banned: boolean;
  limit_exceeded: boolean;
  found: boolean;
  usage?: number;
  limit?: number;
  last_tool?: string;
  total_calls?: number;
  username?: string;
}

export interface ApiVersionResponse {
  latest_version: string | null;
  api_url: string | null;
}

export interface ApiAdminVersionResponse {
  success: boolean;
  registry: { latest_version?: string; api_url?: string };
}

export interface ApiActionRequest {
  action: 'toggle_ban' | 'delete' | 'update_limit' | 'refresh';
  username?: string;
  limit_val?: number;
}

export interface ApiActionResponse {
  success: boolean;
  records?: UserRecord[];
  error?: string;
}

export interface ApiPasscodeChangeRequest {
  new_passcode: string;
}

export interface ApiPasscodeChangeResponse {
  success: boolean;
  message?: string;
  error?: string;
}

export interface CliUser {
  uid: string;
  username?: string;
  email?: string;
  email_verified?: boolean;
  emailVerified?: boolean;
  disabled?: boolean;
  lastRefreshAt?: string;
  banned?: boolean;
  created_at?: string;
  last_login?: string;
}

export interface ApiCliUsersResponse {
  success: boolean;
  users: CliUser[];
}

export interface ApiUserActionRequest {
  action: 'ban' | 'unban' | 'delete';
  uid: string;
}

export interface ApiAdminNotifyRequest {
  target_uid: string;
  title: string;
  message?: string;
  type?: string;
  severity?: 'info' | 'warning' | 'danger' | 'success';
}

export interface ApiAdminNotifyResponse {
  success: boolean;
  notification_id?: string;
  error?: string;
}

export interface ApiUserNotificationDeleteRequest {
  id: string;
}

export interface ApiUserNotificationDeleteResponse {
  success: boolean;
  message?: string;
  error?: string;
}

export type ThemeKey = 'blue' | 'violet' | 'emerald' | 'rose';

export interface Theme {
  name: string;
  primary: string;
  primaryGlow: string;
  primaryBorder: string;
  primaryBg: string;
  accent: string;
}

export type SortColumn = 'username' | 'ip' | 'status' | 'tokens' | 'last_tool' | 'total_calls' | 'last_online';
export type SortDirection = 'asc' | 'desc';
export type StatusFilter = 'all' | 'online' | 'away' | 'offline' | 'banned' | 'limited';
