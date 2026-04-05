import { create } from 'zustand';
import { api } from '../api/client';
import type { TokenStatus, CacheStatus } from '../types/settings';

const SCALE_MAP: Record<string, Record<string, string>> = {
  compact: { '--font-size-xs': '9px', '--font-size-sm': '10px', '--font-size-md': '11px', '--font-size-lg': '14px' },
  default: { '--font-size-xs': '11px', '--font-size-sm': '12px', '--font-size-md': '13px', '--font-size-lg': '16px' },
  large: { '--font-size-xs': '13px', '--font-size-sm': '14px', '--font-size-md': '15px', '--font-size-lg': '18px' },
};

function applyScale(scale: string) {
  const vars = SCALE_MAP[scale] ?? SCALE_MAP.default;
  const root = document.documentElement;
  for (const [key, val] of Object.entries(vars)) {
    root.style.setProperty(key, val);
  }
}

interface SettingsState {
  isOpen: boolean;
  tokenStatus: TokenStatus | null;
  cacheStatus: CacheStatus | null;
  devicePreference: string;
  uiScale: string;
  isValidating: boolean;
  error: string | null;

  openSettings: () => void;
  closeSettings: () => void;
  setDevicePreference: (device: string) => void;
  setUiScale: (scale: string) => void;
  updateToken: (token: string) => Promise<void>;
  clearToken: () => Promise<void>;
  fetchTokenStatus: () => Promise<void>;
  fetchCacheStatus: () => Promise<void>;
  clearCache: () => Promise<void>;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  isOpen: false,
  tokenStatus: null,
  cacheStatus: null,
  devicePreference: localStorage.getItem('nmri-device') ?? 'auto',
  uiScale: localStorage.getItem('nmri-ui-scale') ?? 'default',
  isValidating: false,
  error: null,

  openSettings: () => {
    set({ isOpen: true });
    // Refresh status when opening
    useSettingsStore.getState().fetchTokenStatus();
    useSettingsStore.getState().fetchCacheStatus();
  },
  closeSettings: () => set({ isOpen: false, error: null }),

  setDevicePreference: (device) => {
    localStorage.setItem('nmri-device', device);
    set({ devicePreference: device });
  },

  setUiScale: (scale) => {
    localStorage.setItem('nmri-ui-scale', scale);
    applyScale(scale);
    set({ uiScale: scale });
  },

  updateToken: async (token) => {
    set({ isValidating: true, error: null });
    try {
      const status = await api.settings.updateToken(token);
      set({ tokenStatus: status, isValidating: false });
    } catch (e) {
      set({ error: (e as Error).message, isValidating: false });
    }
  },

  clearToken: async () => {
    try {
      const status = await api.settings.clearToken();
      set({ tokenStatus: status, error: null });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  fetchTokenStatus: async () => {
    try {
      const status = await api.settings.tokenStatus();
      set({ tokenStatus: status });
    } catch {
      // not critical
    }
  },

  fetchCacheStatus: async () => {
    try {
      const status = await api.settings.cacheStatus();
      set({ cacheStatus: status });
    } catch {
      // not critical
    }
  },

  clearCache: async () => {
    try {
      await api.settings.clearCache();
      const prev = useSettingsStore.getState().cacheStatus;
      set({
        cacheStatus: {
          entry_count: 0,
          max_entries: prev?.max_entries ?? 5,
        },
      });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },
}));
