import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { api, setPasscode } from '../api/client';
import type { ApiActionRequest } from '../lib/types';

export function useDashboardData() {
  return useQuery({
    queryKey: ['admin-data'],
    queryFn: () => api.getData(),
    staleTime: 5_000,
    refetchInterval: false, // manual polling via refetch
    retry: (failureCount, error) => {
      if (error instanceof Error && error.message.includes('Invalid passcode')) return false;
      return failureCount < 2;
    },
  });
}

export function useAdminAction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (req: ApiActionRequest) => api.performAction(req),
    onSuccess: (data) => {
      if (data.records) {
        queryClient.setQueryData(['admin-data'], data.records);
      }
    },
  });
}

export function useChangePasscode() {
  return useMutation({
    mutationFn: (newPasscode: string) => {
      setPasscode(newPasscode);
      return api.changePasscode({ new_passcode: newPasscode });
    },
  });
}

export function useVersionInfo() {
  return useQuery({
    queryKey: ['admin-version'],
    queryFn: () => api.getVersionAdmin(),
    enabled: false, // only fetch on demand
    staleTime: 30_000,
  });
}

export function usePublishVersion() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (version: string) => api.publishVersion(version),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-version'] });
    },
  });
}

export function useCliUsers() {
  return useQuery({
    queryKey: ['cli-users'],
    queryFn: () => api.getCliUsers(),
    staleTime: 0,
    gcTime: 0,
    enabled: false, // fetch on modal open (always fresh)
  });
}

export function useUserAction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (req: { action: 'ban' | 'unban' | 'delete'; uid: string }) =>
      api.performUserAction(req),
    onSuccess: (data) => {
      if (data.users) {
        queryClient.invalidateQueries({ queryKey: ["cli-users"] });
      }
    },
  });
}
