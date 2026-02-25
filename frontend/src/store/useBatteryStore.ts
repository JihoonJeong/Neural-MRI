import { create } from 'zustand';
import type { BatteryResult } from '../types/battery';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';
import { useLocaleStore } from './useLocaleStore';

interface BatteryState {
  result: BatteryResult | null;
  isRunning: boolean;
  isDetailOpen: boolean;
  error: string | null;

  runBattery: (categories?: string[]) => Promise<void>;
  openDetail: () => void;
  closeDetail: () => void;
}

export const useBatteryStore = create<BatteryState>((set) => ({
  result: null,
  isRunning: false,
  isDetailOpen: false,
  error: null,

  runBattery: async (categories?: string[]) => {
    set({ isRunning: true, error: null });
    const addLog = useScanStore.getState().addLog;
    addLog('Running functional test battery...');

    try {
      const locale = useLocaleStore.getState().locale;
      const result = await api.battery.run(categories, locale);
      set({ result });
      addLog(`Battery complete: ${result.passed}/${result.total_tests} passed`);
    } catch (e) {
      const msg = (e as Error).message;
      set({ error: msg });
      addLog(`Battery failed: ${msg}`);
    } finally {
      set({ isRunning: false });
    }
  },

  openDetail: () => set({ isDetailOpen: true }),
  closeDetail: () => set({ isDetailOpen: false }),
}));
