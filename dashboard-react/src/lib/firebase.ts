// Firebase client SDK for user-facing dashboard
// Uses the same Firebase project as the admin dashboard.
// Loaded as ES module via Firebase v10+ modular SDK from CDN.

import { initializeApp } from 'firebase/app';
import {
  getAuth,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  sendPasswordResetEmail,
  signOut as fbSignOut,
  onAuthStateChanged,
  type User,
} from 'firebase/auth';

const FIREBASE_CONFIG = {
  apiKey: 'AIzaSyDfdWsO1H11PjSY7IecaX_QICc14yLOtpQ',
  authDomain: 'xbibzstorage.firebaseapp.com',
  databaseURL: 'https://xbibzstorage-default-rtdb.asia-southeast1.firebasedatabase.app',
  projectId: 'xbibzstorage',
  storageBucket: 'xbibzstorage.appspot.com',
  messagingSenderId: '0',
  appId: '1:0:web:user-dashboard',
};

let app: ReturnType<typeof initializeApp> | null = null;
let authInstance: ReturnType<typeof getAuth> | null = null;

export function getFirebaseAuth() {
  if (!authInstance) {
    app = initializeApp(FIREBASE_CONFIG);
    authInstance = getAuth(app);
  }
  return authInstance;
}

export async function signIn(email: string, password: string) {
  const auth = getFirebaseAuth();
  const cred = await signInWithEmailAndPassword(auth, email, password);
  return cred.user;
}

export async function signUp(email: string, password: string) {
  const auth = getFirebaseAuth();
  const cred = await createUserWithEmailAndPassword(auth, email, password);
  return cred.user;
}

export async function resetPassword(email: string) {
  const auth = getFirebaseAuth();
  await sendPasswordResetEmail(auth, email);
}

export async function signOut() {
  const auth = getFirebaseAuth();
  await fbSignOut(auth);
}

export async function getIdToken(user: User): Promise<string> {
  return user.getIdToken();
}

export function onAuthChange(callback: (user: User | null) => void) {
  const auth = getFirebaseAuth();
  return onAuthStateChanged(auth, callback);
}

const API_BASE = 'https://deepseek-dash.bibzflow.workers.dev';

async function apiCall(path: string, user: User, options: RequestInit = {}) {
  const token = await getIdToken(user);
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
      ...(options.headers || {}),
    },
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

export const api = {
  getProfile: (user: User) => apiCall('/api/user/profile', user),
  updateUsername: (user: User, username: string) =>
    apiCall('/api/user/update_username', user, {
      method: 'POST',
      body: JSON.stringify({ username }),
    }),
  getStats: (user: User) => apiCall('/api/user/stats', user),
  getNotifications: (user: User) =>
    apiCall('/api/user/notifications', user),
  deleteAccount: (user: User) =>
    apiCall('/api/user/delete_account', user, {
      method: 'POST',
      body: JSON.stringify({ confirm: 'DELETE' }),
    }),
  deleteNotification: (user: User, id: string) =>
    apiCall('/api/user/notification_delete', user, {
      method: 'POST',
      body: JSON.stringify({ id }),
    }),
};
