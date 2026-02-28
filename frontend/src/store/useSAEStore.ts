import { create } from 'zustand';
import type { SAEData, SAEInfoResponse } from '../types/scan';
import { api } from '../api/client';

interface SAEState {
  saeInfo: SAEInfoResponse | null;
  saeData: SAEData | null;
  isScanning: boolean;
  error: string | null;
  selectedLayer: number;
  selectedFeatureIdx: number | null;
  topK: number;

  fetchInfo: () => Promise<void>;
  runScan: (prompt: string) => Promise<void>;
  setSelectedLayer: (layer: number) => void;
  setSelectedFeatureIdx: (idx: number | null) => void;
  reset: () => void;
}

export const useSAEStore = create<SAEState>((set, get) => ({
  saeInfo: null,
  saeData: null,
  isScanning: false,
  error: null,
  selectedLayer: 0,
  selectedFeatureIdx: null,
  topK: 20,

  fetchInfo: async () => {
    try {
      const info = await api.sae.info();
      set({ saeInfo: info, error: null });
      // Reset layer selection to first available
      if (info.available && info.layers.length > 0) {
        set({ selectedLayer: info.layers[0] });
      }
    } catch (e) {
      set({ saeInfo: null, error: (e as Error).message });
    }
  },

  runScan: async (prompt: string) => {
    const { selectedLayer, topK } = get();
    set({ isScanning: true, error: null });
    try {
      const data = await api.sae.scan(prompt, selectedLayer, topK);
      set({ saeData: data, isScanning: false });
    } catch (e) {
      set({ error: (e as Error).message, isScanning: false });
    }
  },

  setSelectedLayer: (layer: number) => {
    set({ selectedLayer: layer, saeData: null, selectedFeatureIdx: null });
  },

  setSelectedFeatureIdx: (idx: number | null) => {
    set({ selectedFeatureIdx: idx });
  },

  reset: () => {
    set({
      saeInfo: null,
      saeData: null,
      isScanning: false,
      error: null,
      selectedLayer: 0,
      selectedFeatureIdx: null,
    });
  },
}));
