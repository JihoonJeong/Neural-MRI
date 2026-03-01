import { create } from 'zustand';
import type { ActivationData, AnomalyData, CircuitData } from '../types/scan';
import type { CompareData } from '../types/compare';
import type { ScanMode } from '../types/model';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';
import { computeLayerDiffs } from '../utils/compareDiff';

export interface CrossModelData {
  activationData: ActivationData | null;
  circuitData: CircuitData | null;
  anomalyData: AnomalyData | null;
}

type Phase = 'idle' | 'scanning_a' | 'switching' | 'scanning_b' | 'done';

interface CrossModelState {
  isCrossModelMode: boolean;
  modelIdA: string | null;
  modelIdB: string | null;
  dataA: CrossModelData;
  dataB: CrossModelData;
  diffData: CompareData | null;
  prompt: string;
  phase: Phase;
  error: string | null;

  toggleCrossModel: () => void;
  setModelIdB: (id: string) => void;
  runCrossCompare: () => Promise<void>;
  clear: () => void;
}

const emptyData = (): CrossModelData => ({
  activationData: null,
  circuitData: null,
  anomalyData: null,
});

export const useCrossModelStore = create<CrossModelState>((set, get) => ({
  isCrossModelMode: false,
  modelIdA: null,
  modelIdB: null,
  dataA: emptyData(),
  dataB: emptyData(),
  diffData: null,
  prompt: '',
  phase: 'idle',
  error: null,

  toggleCrossModel: () => {
    const next = !get().isCrossModelMode;
    set({ isCrossModelMode: next });
    if (!next) get().clear();
  },

  setModelIdB: (modelIdB) => set({ modelIdB }),

  runCrossCompare: async () => {
    const scanState = useScanStore.getState();
    const mode = scanState.mode;
    const prompt = scanState.prompt;
    const { modelIdB } = get();

    if (mode === 'T1' || mode === 'T2' || !modelIdB || !prompt) return;

    const modelIdA = scanState.activationData?.model_id
      ?? scanState.circuitData?.model_id
      ?? scanState.anomalyData?.model_id
      ?? null;

    set({ phase: 'scanning_a', error: null, prompt, modelIdA });
    scanState.addLog(`Cross-model: scanning ${modelIdA ?? 'current model'}...`);

    try {
      // Phase 1: Ensure model A is scanned
      if (
        (mode === 'fMRI' && !scanState.activationData) ||
        (mode === 'DTI' && !scanState.circuitData) ||
        (mode === 'FLAIR' && !scanState.anomalyData)
      ) {
        await scanState.runScan();
      }

      const freshScan = useScanStore.getState();
      const dataA: CrossModelData = {
        activationData: freshScan.activationData,
        circuitData: freshScan.circuitData,
        anomalyData: freshScan.anomalyData,
      };
      set({ dataA, modelIdA: freshScan.activationData?.model_id ?? freshScan.circuitData?.model_id ?? freshScan.anomalyData?.model_id ?? modelIdA });

      // Phase 2: Switch to model B
      set({ phase: 'switching' });
      scanState.addLog(`Cross-model: switching to ${modelIdB}...`);
      await api.model.load(modelIdB);

      // Phase 3: Scan model B
      set({ phase: 'scanning_b' });
      scanState.addLog(`Cross-model: scanning ${modelIdB}...`);

      const dataB: CrossModelData = { activationData: null, circuitData: null, anomalyData: null };
      if (mode === 'fMRI') {
        dataB.activationData = await api.scan.activation(prompt);
      } else if (mode === 'DTI') {
        dataB.circuitData = await api.scan.circuits(prompt);
      } else if (mode === 'FLAIR') {
        dataB.anomalyData = await api.scan.anomaly(prompt);
      }

      // Phase 4: Compute diff
      const tokensA = dataA.activationData?.tokens ?? dataA.circuitData?.tokens ?? dataA.anomalyData?.tokens ?? [];
      const tokensB = dataB.activationData?.tokens ?? dataB.circuitData?.tokens ?? dataB.anomalyData?.tokens ?? [];
      const finalModelIdA = get().modelIdA ?? 'Model A';
      const diffData = computeLayerDiffs(mode as ScanMode, finalModelIdA, modelIdB, tokensA, tokensB, dataA, dataB);

      set({ dataB, diffData, phase: 'done' });
      scanState.addLog('Cross-model comparison complete');
    } catch (e) {
      const msg = (e as Error).message;
      set({ error: msg, phase: 'idle' });
      scanState.addLog(`Cross-model failed: ${msg}`);
    }
  },

  clear: () => set({
    dataA: emptyData(),
    dataB: emptyData(),
    diffData: null,
    phase: 'idle',
    error: null,
    modelIdA: null,
    modelIdB: null,
    prompt: '',
  }),
}));
