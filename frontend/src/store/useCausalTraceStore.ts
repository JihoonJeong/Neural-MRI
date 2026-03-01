import { create } from 'zustand';
import type { CausalTraceResult } from '../types/causalTrace';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';

interface CausalTraceState {
  result: CausalTraceResult | null;
  isTracing: boolean;
  cleanPrompt: string;
  corruptPrompt: string;
  error: string | null;

  setCleanPrompt: (p: string) => void;
  setCorruptPrompt: (p: string) => void;
  runTrace: () => Promise<void>;
  clear: () => void;
}

export const useCausalTraceStore = create<CausalTraceState>((set, get) => ({
  result: null,
  isTracing: false,
  cleanPrompt: '',
  corruptPrompt: '',
  error: null,

  setCleanPrompt: (cleanPrompt) => set({ cleanPrompt }),
  setCorruptPrompt: (corruptPrompt) => set({ corruptPrompt }),

  runTrace: async () => {
    const { cleanPrompt, corruptPrompt } = get();
    const prompt = cleanPrompt || useScanStore.getState().prompt;
    if (!prompt || !corruptPrompt) return;

    set({ isTracing: true, error: null });
    useScanStore.getState().addLog(`Causal trace: "${prompt}" vs "${corruptPrompt}"...`);

    try {
      const result = await api.perturb.causalTrace(prompt, corruptPrompt);
      set({ result, isTracing: false });
      useScanStore.getState().addLog(`Causal trace complete: ${result.cells.length} components`);
    } catch (e) {
      const msg = (e as Error).message;
      set({ error: msg, isTracing: false });
      useScanStore.getState().addLog(`Causal trace failed: ${msg}`);
    }
  },

  clear: () => set({ result: null, error: null, isTracing: false }),
}));
