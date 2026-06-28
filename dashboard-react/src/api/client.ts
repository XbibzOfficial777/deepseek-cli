import type {
  UserRecord,
  ApiCheckResponse,
  ApiVersionResponse,
  ApiAdminVersionResponse,
  ApiActionResponse,
  ApiActionRequest,
  ApiPasscodeChangeRequest,
  ApiPasscodeChangeResponse,
  ApiCliUsersResponse,
  ApiUserActionRequest,
  ApiAdminNotifyRequest,
  ApiAdminNotifyResponse,
  ApiUserNotificationDeleteRequest,
  ApiUserNotificationDeleteResponse,
} from '../lib/types';

const STORAGE_KEY = 'deepseek_admin_passcode';

export function getPasscode(): string {
  return localStorage.getItem(STORAGE_KEY) || '';
}

export function setPasscode(value: string): void {
  if (value) localStorage.setItem(STORAGE_KEY, value);
  else localStorage.removeItem(STORAGE_KEY);
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
    this.name = 'ApiError';
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  const passcode = getPasscode();
  // Only send admin passcode for admin endpoints to avoid leaking it to user endpoints
  const isAdminPath = path.startsWith('/api/admin');
  if (isAdminPath && passcode && !headers.has('X-Admin-Passcode')) {
    headers.set('X-Admin-Passcode', passcode);
  }
  if (options.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(path, {
    ...options,
    headers,
    cache: 'no-cache',
  });

  if (response.status === 401) {
    const errorBody = await response.json().catch(() => ({}));
    const message = errorBody.error || 'Unauthorized';
    // Only clear admin passcode for admin endpoint 401s
    if (isAdminPath) {
      setPasscode('');
      throw new ApiError('Invalid passcode. Access denied.', 401);
    }
    throw new ApiError(message, 401);
  }

  if (!response.ok) {
    const errorBody = await response.json().catch(() => ({}));
    throw new ApiError(errorBody.error || `Server responded with status ${response.status}`, response.status);
  }

  return response.json() as Promise<T>;
}

export const api = {
  // Public endpoints
  check: (ip: string) => request<ApiCheckResponse>(`/api/check?ip=${encodeURIComponent(ip)}`),
  getVersion: () => request<ApiVersionResponse>('/api/version'),

  // Admin endpoints (require passcode)
  getData: () => request<UserRecord[]>('/api/admin/data'),
  performAction: (req: ApiActionRequest) =>
    request<ApiActionResponse>('/api/admin/action', {
      method: 'POST',
      body: JSON.stringify(req),
    }),
  changePasscode: (req: ApiPasscodeChangeRequest) =>
    request<ApiPasscodeChangeResponse>('/api/admin/change_password', {
      method: 'POST',
      body: JSON.stringify(req),
    }),
  getVersionAdmin: () => request<ApiAdminVersionResponse>('/api/admin/version'),
  publishVersion: (version: string) =>
    request<ApiAdminVersionResponse>('/api/admin/version', {
      method: 'POST',
      body: JSON.stringify({ latest_version: version }),
    }),
  getCliUsers: () => request<ApiCliUsersResponse>('/api/admin/users'),
  performUserAction: (req: ApiUserActionRequest) =>
    request<ApiCliUsersResponse>('/api/admin/user_action', {
      method: 'POST',
      body: JSON.stringify(req),
    }),
  sendNotification: (req: ApiAdminNotifyRequest) =>
    request<ApiAdminNotifyResponse>('/api/admin/notify', {
      method: 'POST',
      body: JSON.stringify(req),
    }),
  deleteNotification: (req: ApiUserNotificationDeleteRequest) =>
    request<ApiUserNotificationDeleteResponse>('/api/user/notification_delete', {
      method: 'POST',
      body: JSON.stringify(req),
    }),
};
