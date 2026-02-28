import { create } from 'zustand';
import type { ModelInfo } from '../types/model';
import { api } from '../api/client';
import type { ModelListEntry } from '../api/client';
import { useSAEStore } from './useSAEStore';

interface ModelState {
  modelInfo: ModelInfo | null;
  isLoading: boolean;
  error: string | null;
  availableModels: ModelListEntry[];
  loadModel: (modelId: string) => Promise<void>;
  fetchModelInfo: () => Promise<void>;
  fetchModels: () => Promise<void>;
}

export const useModelStore = create<ModelState>((set) => ({
  modelInfo: null,
  isLoading: false,
  error: null,
  availableModels: [],

  loadModel: async (modelId) => {
    set({ isLoading: true, error: null });
    try {
      const info = await api.model.load(modelId);
      set({ modelInfo: info, isLoading: false });
      // Refresh model list to update is_loaded flags
      try {
        const models = await api.model.list();
        set({ availableModels: models });
      } catch {
        // non-critical
      }
      // Fetch SAE availability for the new model
      useSAEStore.getState().reset();
      useSAEStore.getState().fetchInfo();
    } catch (e) {
      set({ error: (e as Error).message, isLoading: false });
    }
  },

  fetchModelInfo: async () => {
    try {
      const info = await api.model.info();
      set({ modelInfo: info, error: null });
      // Fetch SAE availability
      useSAEStore.getState().fetchInfo();
    } catch {
      // Model not yet loaded — that's ok on initial load
    }
  },

  fetchModels: async () => {
    try {
      const models = await api.model.list();
      set({ availableModels: models });
    } catch {
      // Server may not be ready yet
    }
  },
}));
