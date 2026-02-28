import { create } from 'zustand';
import type { ActivationData, AnomalyData, CircuitData } from '../types/scan';
import type { CompareData, LayerDiff } from '../types/compare';
import type { ScanMode } from '../types/model';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';

interface CompareDataB {
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

function computeDiff(
  mode: ScanMode,
  promptA: string,
  promptB: string,
  scanStore: ReturnType<typeof useScanStore.getState>,
  dataB: CompareDataB,
): CompareData | null {
  if (mode === 'fMRI' && scanStore.activationData && dataB.activationData) {
    const a = scanStore.activationData;
    const b = dataB.activationData;
    const diffs: LayerDiff[] = [];
    for (let i = 0; i < Math.min(a.layers.length, b.layers.length); i++) {
      const aActs = a.layers[i].activations;
      const bActs = b.layers[i].activations;
      const avgA = aActs.reduce((s, v) => s + v, 0) / Math.max(aActs.length, 1);
      const avgB = bActs.reduce((s, v) => s + v, 0) / Math.max(bActs.length, 1);
      diffs.push({ layer_id: a.layers[i].layer_id, delta: avgB - avgA });
    }
    const maxAbs = Math.max(...diffs.map((d) => Math.abs(d.delta)), 0.001);
    return {
      promptA, promptB,
      tokens_a: a.tokens, tokens_b: b.tokens,
      layerDiffs: diffs, maxAbsDelta: maxAbs,
    };
  }

  if (mode === 'DTI' && scanStore.circuitData && dataB.circuitData) {
    const a = scanStore.circuitData;
    const b = dataB.circuitData;
    const aMap = new Map(a.components.map((c) => [c.layer_id, c.importance]));
    const diffs: LayerDiff[] = b.components.map((c) => ({
      layer_id: c.layer_id,
      delta: c.importance - (aMap.get(c.layer_id) ?? 0),
    }));
    const maxAbs = Math.max(...diffs.map((d) => Math.abs(d.delta)), 0.001);
    return {
      promptA, promptB,
      tokens_a: a.tokens, tokens_b: b.tokens,
      layerDiffs: diffs, maxAbsDelta: maxAbs,
    };
  }

  if (mode === 'FLAIR' && scanStore.anomalyData && dataB.anomalyData) {
    const a = scanStore.anomalyData;
    const b = dataB.anomalyData;
    const diffs: LayerDiff[] = [];
    for (let i = 0; i < Math.min(a.layers.length, b.layers.length); i++) {
      const aScores = a.layers[i].anomaly_scores;
      const bScores = b.layers[i].anomaly_scores;
      const avgA = aScores.reduce((s, v) => s + v, 0) / Math.max(aScores.length, 1);
      const avgB = bScores.reduce((s, v) => s + v, 0) / Math.max(bScores.length, 1);
      diffs.push({ layer_id: a.layers[i].layer_id, delta: avgB - avgA });
    }
    const maxAbs = Math.max(...diffs.map((d) => Math.abs(d.delta)), 0.001);
    return {
      promptA, promptB,
      tokens_a: a.tokens, tokens_b: b.tokens,
      layerDiffs: diffs, maxAbsDelta: maxAbs,
    };
  }

  return null;
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
      const diffData = computeDiff(mode, freshScanState.prompt, promptB, freshScanState, newDataB);

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
