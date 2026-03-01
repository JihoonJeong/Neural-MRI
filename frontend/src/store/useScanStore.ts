import { create } from 'zustand';
import type { ScanMode } from '../types/model';
import type { ActivationData, AnomalyData, CircuitData, StructuralData, WeightData } from '../types/scan';
import { api } from '../api/client';

interface LogEntry {
  time: string;
  msg: string;
}

export type LayoutMode = 'vertical' | 'brain' | 'network' | 'radial';

interface ScanState {
  mode: ScanMode;
  layoutMode: LayoutMode;
  structuralData: StructuralData | null;
  weightData: WeightData | null;
  activationData: ActivationData | null;
  circuitData: CircuitData | null;
  anomalyData: AnomalyData | null;
  isScanning: boolean;
  selectedLayerId: string | null;
  selectedTokenIdx: number;
  tokenCount: number;
  prompt: string;
  logs: LogEntry[];

  setMode: (mode: ScanMode) => void;
  setLayoutMode: (mode: LayoutMode) => void;
  setPrompt: (prompt: string) => void;
  runScan: () => Promise<void>;
  clearScanData: () => void;
  selectLayer: (layerId: string | null) => void;
  setSelectedTokenIdx: (idx: number) => void;
  stepToken: (delta: number) => void;
  addLog: (msg: string) => void;
}

// Module-level abort controller for cancelling in-flight scan requests
let _scanAbortController: AbortController | null = null;

function timestamp(): string {
  const now = new Date();
  return `${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
}

export const useScanStore = create<ScanState>((set, get) => ({
  mode: 'T1',
  layoutMode: 'vertical',
  structuralData: null,
  weightData: null,
  activationData: null,
  circuitData: null,
  anomalyData: null,
  isScanning: false,
  selectedLayerId: null,
  selectedTokenIdx: 0,
  tokenCount: 0,
  prompt: 'The capital of France is',
  logs: [{ time: '00:00', msg: 'Neural MRI Scanner initialized' }],

  setMode: (mode) => {
    set({ mode });
    get().addLog(`Mode: ${mode}`);
  },

  setLayoutMode: (layoutMode) => set({ layoutMode }),

  setPrompt: (prompt) => set({ prompt }),

  runScan: async () => {
    const { mode, prompt, addLog, structuralData } = get();

    // Cancel any in-flight scan
    if (_scanAbortController) {
      _scanAbortController.abort();
    }
    _scanAbortController = new AbortController();
    const { signal } = _scanAbortController;

    set({ isScanning: true });
    addLog(`Scanning ${mode}...`);

    try {
      if (mode === 'T1') {
        const data = await api.scan.structural(signal);
        set({ structuralData: data });
        addLog(`T1 complete: ${data.layers.length} components`);
      } else if (mode === 'T2') {
        const data = await api.scan.weights(undefined, signal);
        set({ weightData: data });
        addLog(`T2 complete: ${data.layers.length} weight tensors`);
      } else if (mode === 'fMRI') {
        // Ensure structural data is loaded (needed for node layout)
        if (!structuralData) {
          const sData = await api.scan.structural(signal);
          set({ structuralData: sData });
          addLog(`T1 auto-loaded for layout`);
        }
        const data = await api.scan.activation(prompt, signal);
        set({
          activationData: data,
          tokenCount: data.tokens.length,
          selectedTokenIdx: 0,
        });
        addLog(`fMRI complete: ${data.tokens.length} tokens, ${data.layers.length} layers`);
      } else if (mode === 'DTI') {
        // Ensure structural data is loaded
        if (!structuralData) {
          const sData = await api.scan.structural(signal);
          set({ structuralData: sData });
          addLog(`T1 auto-loaded for layout`);
        }
        const data = await api.scan.circuits(prompt, -1, signal);
        set({
          circuitData: data,
          tokenCount: data.tokens.length,
          selectedTokenIdx: 0,
        });
        addLog(`DTI complete: ${data.components.length} components, ${data.connections.filter((c) => c.is_pathway).length} pathways`);
      } else if (mode === 'FLAIR') {
        if (!structuralData) {
          const sData = await api.scan.structural(signal);
          set({ structuralData: sData });
          addLog(`T1 auto-loaded for layout`);
        }
        const data = await api.scan.anomaly(prompt, signal);
        set({
          anomalyData: data,
          tokenCount: data.tokens.length,
          selectedTokenIdx: 0,
        });
        addLog(`FLAIR complete: ${data.tokens.length} tokens, ${data.layers.length} layers`);
      }
    } catch (e) {
      if ((e as Error).name === 'AbortError') return; // silently ignore aborted
      addLog(`Scan failed: ${(e as Error).message}`);
    } finally {
      // Only clear scanning state if this is still the active controller
      if (_scanAbortController?.signal === signal) {
        set({ isScanning: false });
      }
    }
  },

  clearScanData: () => {
    if (_scanAbortController) {
      _scanAbortController.abort();
      _scanAbortController = null;
    }
    set({
      structuralData: null,
      weightData: null,
      activationData: null,
      circuitData: null,
      anomalyData: null,
      isScanning: false,
      selectedLayerId: null,
      selectedTokenIdx: 0,
      tokenCount: 0,
    });
    get().addLog('Scan data cleared (model switched)');
  },

  selectLayer: (layerId) => set({ selectedLayerId: layerId }),

  setSelectedTokenIdx: (idx) => {
    const { tokenCount } = get();
    if (idx >= 0 && idx < tokenCount) {
      set({ selectedTokenIdx: idx });
    }
  },

  stepToken: (delta) => {
    const { selectedTokenIdx, tokenCount } = get();
    const next = selectedTokenIdx + delta;
    if (next >= 0 && next < tokenCount) {
      set({ selectedTokenIdx: next });
    }
  },

  addLog: (msg) => {
    set((s) => ({
      logs: [...s.logs.slice(-50), { time: timestamp(), msg }],
    }));
  },
}));
