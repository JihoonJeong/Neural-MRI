import { create } from 'zustand';

import { api } from '../api/client';
import type {
  EmotionListResponse,
  ExtractProbesResponse,
  LayerEvolutionResponse,
  PCAResponse,
  ProjectResponse,
  SteerResponse,
  SweepResponse,
} from '../types/emotion';
import { useScanStore } from './useScanStore';

interface EmotionState {
  // Tab state
  tabActive: boolean;
  setTabActive: (active: boolean) => void;

  // Data
  emotionList: EmotionListResponse | null;
  probeResult: ExtractProbesResponse | null;
  steerResult: SteerResponse | null;
  projectResult: ProjectResponse | null;
  pcaResult: PCAResponse | null;
  sweepResult: SweepResponse | null;
  layerEvoResult: LayerEvolutionResponse | null;

  // Loading states
  isExtracting: boolean;
  isSteering: boolean;
  isProjecting: boolean;
  isSweeping: boolean;

  error: string | null;
  selectedEmotion: string;
  strength: number;

  // Actions
  fetchList: () => Promise<void>;
  extractProbes: () => Promise<void>;
  steer: (prompt: string) => Promise<void>;
  project: (prompt: string) => Promise<void>;
  fetchPCA: () => Promise<void>;
  sweep: (prompt: string) => Promise<void>;
  layerEvolution: (prompt: string, tokenIdx?: number) => Promise<void>;
  setSelectedEmotion: (emotion: string) => void;
  setStrength: (strength: number) => void;
  reset: () => void;
}

export const useEmotionStore = create<EmotionState>((set, get) => ({
  tabActive: false,
  setTabActive: (active) => set({ tabActive: active }),

  emotionList: null,
  probeResult: null,
  steerResult: null,
  projectResult: null,
  pcaResult: null,
  sweepResult: null,
  layerEvoResult: null,

  isExtracting: false,
  isSteering: false,
  isProjecting: false,
  isSweeping: false,

  error: null,
  selectedEmotion: 'calm',
  strength: 0.05,

  fetchList: async () => {
    try {
      const list = await api.emotion.list();
      set({ emotionList: list });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  extractProbes: async () => {
    set({ isExtracting: true, error: null });
    try {
      const result = await api.emotion.extractProbes();
      set({ probeResult: result, isExtracting: false });
      useScanStore.getState().addLog(`Emotion probes: ${result.n_emotions} extracted (layer ${result.layer_idx})`);
    } catch (e) {
      const msg = (e as Error).message;
      set({ error: msg, isExtracting: false });
      useScanStore.getState().addLog(`Emotion probe extraction failed: ${msg}`);
    }
  },

  steer: async (prompt: string) => {
    const { selectedEmotion, strength } = get();
    set({ isSteering: true, error: null });
    try {
      const result = await api.emotion.steer(prompt, selectedEmotion, strength);
      set({ steerResult: result, isSteering: false });
      useScanStore.getState().addLog(`Emotion steer: ${selectedEmotion} (${strength > 0 ? '+' : ''}${strength})`);
    } catch (e) {
      const msg = (e as Error).message;
      set({ error: msg, isSteering: false });
      useScanStore.getState().addLog(`Emotion steer failed: ${msg}`);
    }
  },

  project: async (prompt: string) => {
    set({ isProjecting: true, error: null });
    try {
      const result = await api.emotion.project(prompt);
      set({ projectResult: result, isProjecting: false });
      useScanStore.getState().addLog(`Emotion project: ${result.emotions.length} emotions × ${result.tokens.length} tokens`);
    } catch (e) {
      const msg = (e as Error).message;
      set({ error: msg, isProjecting: false });
    }
  },

  fetchPCA: async () => {
    try {
      const result = await api.emotion.pca();
      set({ pcaResult: result });
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  sweep: async (prompt: string) => {
    const { selectedEmotion } = get();
    set({ isSweeping: true, error: null });
    try {
      const result = await api.emotion.sweep(prompt, selectedEmotion);
      set({ sweepResult: result, isSweeping: false });
      useScanStore.getState().addLog(`Sweep: ${selectedEmotion}, ${result.points.length} points`);
    } catch (e) {
      const msg = (e as Error).message;
      set({ error: msg, isSweeping: false });
    }
  },

  layerEvolution: async (prompt: string, tokenIdx = -1) => {
    try {
      const result = await api.emotion.layerEvolution(prompt, tokenIdx);
      set({ layerEvoResult: result });
      useScanStore.getState().addLog(`Layer evolution: ${result.emotions.length} emotions × ${result.layers.length} layers`);
    } catch (e) {
      set({ error: (e as Error).message });
    }
  },

  setSelectedEmotion: (emotion) => set({ selectedEmotion: emotion }),
  setStrength: (strength) => set({ strength }),
  reset: () =>
    set({
      probeResult: null,
      steerResult: null,
      projectResult: null,
      pcaResult: null,
      sweepResult: null,
      layerEvoResult: null,
      error: null,
      isExtracting: false,
      isSteering: false,
      isProjecting: false,
      isSweeping: false,
    }),
}));
