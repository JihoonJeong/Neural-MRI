import { create } from 'zustand';
import type { ScanMode } from '../types/model';
import type { ActivationData, AnomalyData, CircuitData, StructuralData, WeightData } from '../types/scan';
import { api } from '../api/client';

interface LogEntry {
  time: string;
  msg: string;
}

interface ScanState {
  mode: ScanMode;
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
  setPrompt: (prompt: string) => void;
  runScan: () => Promise<void>;
  selectLayer: (layerId: string | null) => void;
  setSelectedTokenIdx: (idx: number) => void;
  stepToken: (delta: number) => void;
  addLog: (msg: string) => void;
}

function timestamp(): string {
  const now = new Date();
  return `${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
}

export const useScanStore = create<ScanState>((set, get) => ({
  mode: 'T1',
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

  setPrompt: (prompt) => set({ prompt }),

  runScan: async () => {
    const { mode, prompt, addLog, structuralData } = get();
    set({ isScanning: true });
    addLog(`Scanning ${mode}...`);

    try {
      if (mode === 'T1') {
        const data = await api.scan.structural();
        set({ structuralData: data });
        addLog(`T1 complete: ${data.layers.length} components`);
      } else if (mode === 'T2') {
        const data = await api.scan.weights();
        set({ weightData: data });
        addLog(`T2 complete: ${data.layers.length} weight tensors`);
      } else if (mode === 'fMRI') {
        // Ensure structural data is loaded (needed for node layout)
        if (!structuralData) {
          const sData = await api.scan.structural();
          set({ structuralData: sData });
          addLog(`T1 auto-loaded for layout`);
        }
        const data = await api.scan.activation(prompt);
        set({
          activationData: data,
          tokenCount: data.tokens.length,
          selectedTokenIdx: 0,
        });
        addLog(`fMRI complete: ${data.tokens.length} tokens, ${data.layers.length} layers`);
      } else if (mode === 'DTI') {
        // Ensure structural data is loaded
        if (!structuralData) {
          const sData = await api.scan.structural();
          set({ structuralData: sData });
          addLog(`T1 auto-loaded for layout`);
        }
        const data = await api.scan.circuits(prompt);
        set({
          circuitData: data,
          tokenCount: data.tokens.length,
          selectedTokenIdx: 0,
        });
        addLog(`DTI complete: ${data.components.length} components, ${data.connections.filter((c) => c.is_pathway).length} pathways`);
      } else if (mode === 'FLAIR') {
        if (!structuralData) {
          const sData = await api.scan.structural();
          set({ structuralData: sData });
          addLog(`T1 auto-loaded for layout`);
        }
        const data = await api.scan.anomaly(prompt);
        set({
          anomalyData: data,
          tokenCount: data.tokens.length,
          selectedTokenIdx: 0,
        });
        addLog(`FLAIR complete: ${data.tokens.length} tokens, ${data.layers.length} layers`);
      }
    } catch (e) {
      addLog(`Scan failed: ${(e as Error).message}`);
    } finally {
      set({ isScanning: false });
    }
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
