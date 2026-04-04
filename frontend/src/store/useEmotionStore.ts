import { create } from 'zustand';

import { api } from '../api/client';
import type {
  EmotionListResponse,
  ExtractProbesResponse,
  SteerResponse,
} from '../types/emotion';
import { useScanStore } from './useScanStore';

interface EmotionState {
  emotionList: EmotionListResponse | null;
  probeResult: ExtractProbesResponse | null;
  steerResult: SteerResponse | null;
  isExtracting: boolean;
  isSteering: boolean;
  error: string | null;
  selectedEmotion: string;
  strength: number;

  fetchList: () => Promise<void>;
  extractProbes: () => Promise<void>;
  steer: (prompt: string) => Promise<void>;
  setSelectedEmotion: (emotion: string) => void;
  setStrength: (strength: number) => void;
  reset: () => void;
}

export const useEmotionStore = create<EmotionState>((set, get) => ({
  emotionList: null,
  probeResult: null,
  steerResult: null,
  isExtracting: false,
  isSteering: false,
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

  setSelectedEmotion: (emotion) => set({ selectedEmotion: emotion }),
  setStrength: (strength) => set({ strength }),
  reset: () =>
    set({
      probeResult: null,
      steerResult: null,
      error: null,
      isExtracting: false,
      isSteering: false,
    }),
}));
