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
  includeSAE: boolean;
  saeLayer: number | null;

  runBattery: (categories?: string[]) => Promise<void>;
  openDetail: () => void;
  closeDetail: () => void;
  setIncludeSAE: (v: boolean) => void;
  setSaeLayer: (v: number | null) => void;
}

export const useBatteryStore = create<BatteryState>((set, get) => ({
  result: null,
  isRunning: false,
  isDetailOpen: false,
  error: null,
  includeSAE: false,
  saeLayer: null,

  runBattery: async (categories?: string[]) => {
    const { includeSAE, saeLayer } = get();
    set({ isRunning: true, error: null });
    const addLog = useScanStore.getState().addLog;
    addLog('Running functional test battery' + (includeSAE ? ' + SAE...' : '...'));

    try {
      const locale = useLocaleStore.getState().locale;
      const result = await api.battery.run(categories, locale, includeSAE, saeLayer);
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
  setIncludeSAE: (v: boolean) => set({ includeSAE: v }),
  setSaeLayer: (v: number | null) => set({ saeLayer: v }),
}));
