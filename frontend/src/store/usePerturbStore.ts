import { create } from 'zustand';
import type { PerturbResult } from '../types/perturb';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';

export type PerturbationType = 'zero' | 'amplify' | 'ablate';

interface PerturbState {
  result: PerturbResult | null;
  isPerturbing: boolean;
  isRescanAnimating: boolean;
  error: string | null;

  runPerturbation: (type: PerturbationType, component: string, factor?: number) => Promise<void>;
  reset: () => void;
}

export const usePerturbStore = create<PerturbState>((set) => ({
  result: null,
  isPerturbing: false,
  isRescanAnimating: false,
  error: null,

  runPerturbation: async (type, component, factor = 2.0) => {
    const prompt = useScanStore.getState().prompt;
    const addLog = useScanStore.getState().addLog;
    set({ isPerturbing: true, error: null });
    addLog(`Perturbation: ${type} on ${component}`);

    try {
      let result: PerturbResult;
      switch (type) {
        case 'zero':
          result = await api.perturb.zero(component, prompt);
          break;
        case 'amplify':
          result = await api.perturb.amplify(component, prompt, factor);
          break;
        case 'ablate':
          result = await api.perturb.ablate(component, prompt);
          break;
        default:
          throw new Error(`Unknown perturbation type: ${type}`);
      }
      set({ result, isPerturbing: false, isRescanAnimating: true });
      addLog(
        `${type}: "${result.original.token}" -> "${result.perturbed.token}" (KL: ${result.kl_divergence.toFixed(3)})`,
      );

      // End rescan animation after 300ms
      setTimeout(() => set({ isRescanAnimating: false }), 300);
    } catch (e) {
      set({ error: (e as Error).message, isPerturbing: false });
      addLog(`Perturbation failed: ${(e as Error).message}`);
    }
  },

  reset: () => {
    set({ result: null, error: null, isRescanAnimating: false });
    useScanStore.getState().addLog('Perturbations reset');
  },
}));
