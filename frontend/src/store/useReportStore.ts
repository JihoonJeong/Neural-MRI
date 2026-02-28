import { create } from 'zustand';
import type { DiagnosticReport } from '../types/report';
import type { ReportRequest } from '../types/report';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';
import { useBatteryStore } from './useBatteryStore';
import { useSAEStore } from './useSAEStore';
import { useLocaleStore } from './useLocaleStore';

interface ReportState {
  report: DiagnosticReport | null;
  isGenerating: boolean;
  isOpen: boolean;
  error: string | null;

  generateReport: () => Promise<void>;
  openReport: () => void;
  closeReport: () => void;
}

export const useReportStore = create<ReportState>((set) => ({
  report: null,
  isGenerating: false,
  isOpen: false,
  error: null,

  generateReport: async () => {
    set({ isGenerating: true, error: null });
    const addLog = useScanStore.getState().addLog;
    addLog('Generating diagnostic report...');

    try {
      const scanState = useScanStore.getState();
      const batteryState = useBatteryStore.getState();

      const req: ReportRequest = {
        prompt: scanState.prompt,
        locale: useLocaleStore.getState().locale,
      };

      // Pass cached scan data to avoid re-running scans
      if (scanState.structuralData) {
        req.cached_t1 = scanState.structuralData as unknown as Record<string, unknown>;
      }
      if (scanState.weightData) {
        req.cached_t2 = scanState.weightData as unknown as Record<string, unknown>;
      }
      if (scanState.activationData) {
        req.cached_fmri = scanState.activationData as unknown as Record<string, unknown>;
      }
      if (scanState.circuitData) {
        req.cached_dti = scanState.circuitData as unknown as Record<string, unknown>;
      }
      if (scanState.anomalyData) {
        req.cached_flair = scanState.anomalyData as unknown as Record<string, unknown>;
      }

      // Include battery results if available
      if (batteryState.result) {
        req.cached_battery = batteryState.result as unknown as Record<string, unknown>;
      }

      // Include SAE results if available
      const saeState = useSAEStore.getState();
      if (saeState.saeData) {
        req.cached_sae = saeState.saeData as unknown as Record<string, unknown>;
      }

      const report = await api.report.generate(req);
      set({ report, isOpen: true });
      addLog(`Report complete: ${report.findings.length} findings`);
    } catch (e) {
      const msg = (e as Error).message;
      set({ error: msg });
      addLog(`Report failed: ${msg}`);
    } finally {
      set({ isGenerating: false });
    }
  },

  openReport: () => set({ isOpen: true }),
  closeReport: () => set({ isOpen: false }),
}));
