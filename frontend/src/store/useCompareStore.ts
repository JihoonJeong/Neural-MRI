import { create } from 'zustand';
import type { ActivationData, AnomalyData, CircuitData } from '../types/scan';
import type { CompareData } from '../types/compare';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';
import { computeLayerDiffs } from '../utils/compareDiff';

export interface CompareDataB {
  activationData: ActivationData | null;
  circuitData: CircuitData | null;
  anomalyData: AnomalyData | null;
}

interface CompareState {
  isCompareMode: boolean;
  promptB: string;
  dataB: CompareDataB;
  diffData: CompareData | null;
  isScanningB: boolean;
  selectedTokenIdx: number;
  tokenCountB: number;

  toggleCompare: () => void;
  setPromptB: (p: string) => void;
  runCompare: () => Promise<void>;
  setSelectedTokenIdx: (idx: number) => void;
  stepToken: (delta: number) => void;
  clear: () => void;
}

export const useCompareStore = create<CompareState>((set, get) => ({
  isCompareMode: false,
  promptB: 'The capital of Germany is',
  dataB: { activationData: null, circuitData: null, anomalyData: null },
  diffData: null,
  isScanningB: false,
  selectedTokenIdx: 0,
  tokenCountB: 0,

  toggleCompare: () => {
    const next = !get().isCompareMode;
    set({ isCompareMode: next });
    if (!next) get().clear();
  },

  setPromptB: (promptB) => set({ promptB }),

  runCompare: async () => {
    const scanState = useScanStore.getState();
    const { promptB } = get();
    const mode = scanState.mode;

    if (mode === 'T1' || mode === 'T2') return;

    set({ isScanningB: true });
    scanState.addLog(`Compare: scanning B "${promptB}"...`);

    // Run A scan if no data yet
    if (
      (mode === 'fMRI' && !scanState.activationData) ||
      (mode === 'DTI' && !scanState.circuitData) ||
      (mode === 'FLAIR' && !scanState.anomalyData)
    ) {
      await scanState.runScan();
    }

    try {
      // Ensure structural data exists
      if (!scanState.structuralData) {
        const sData = await api.scan.structural();
        useScanStore.setState({ structuralData: sData });
      }

      const newDataB: CompareDataB = { activationData: null, circuitData: null, anomalyData: null };

      if (mode === 'fMRI') {
        const data = await api.scan.activation(promptB);
        newDataB.activationData = data;
        set({ tokenCountB: data.tokens.length });
      } else if (mode === 'DTI') {
        const data = await api.scan.circuits(promptB);
        newDataB.circuitData = data;
        set({ tokenCountB: data.tokens.length });
      } else if (mode === 'FLAIR') {
        const data = await api.scan.anomaly(promptB);
        newDataB.anomalyData = data;
        set({ tokenCountB: data.tokens.length });
      }

      const freshScanState = useScanStore.getState();
      const dataA = {
        activationData: freshScanState.activationData,
        circuitData: freshScanState.circuitData,
        anomalyData: freshScanState.anomalyData,
      };
      const tokensA = freshScanState.activationData?.tokens ?? freshScanState.circuitData?.tokens ?? freshScanState.anomalyData?.tokens ?? [];
      const tokensB = newDataB.activationData?.tokens ?? newDataB.circuitData?.tokens ?? newDataB.anomalyData?.tokens ?? [];
      const diffData = computeLayerDiffs(mode, freshScanState.prompt, promptB, tokensA, tokensB, dataA, newDataB);

      set({ dataB: newDataB, diffData, selectedTokenIdx: 0 });
      scanState.addLog(`Compare complete`);
    } catch (e) {
      scanState.addLog(`Compare failed: ${(e as Error).message}`);
    } finally {
      set({ isScanningB: false });
    }
  },

  setSelectedTokenIdx: (idx) => {
    const { tokenCountB } = get();
    if (idx >= 0 && idx < tokenCountB) {
      set({ selectedTokenIdx: idx });
    }
  },

  stepToken: (delta) => {
    const { selectedTokenIdx, tokenCountB } = get();
    const next = selectedTokenIdx + delta;
    if (next >= 0 && next < tokenCountB) {
      set({ selectedTokenIdx: next });
    }
  },

  clear: () => set({
    dataB: { activationData: null, circuitData: null, anomalyData: null },
    diffData: null,
    isScanningB: false,
    selectedTokenIdx: 0,
    tokenCountB: 0,
  }),
}));
