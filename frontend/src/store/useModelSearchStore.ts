import { create } from 'zustand';
import { api } from '../api/client';
import type { HubSearchResult } from '../types/settings';

interface ModelSearchState {
  query: string;
  results: HubSearchResult[];
  recentModels: string[];
  isSearching: boolean;
  error: string | null;
  tlOnly: boolean;

  setQuery: (q: string) => void;
  setTlOnly: (v: boolean) => void;
  search: (q: string) => Promise<void>;
  addRecentModel: (modelId: string) => void;
  clearResults: () => void;
}

export const useModelSearchStore = create<ModelSearchState>((set, get) => ({
  query: '',
  results: [],
  recentModels: JSON.parse(
    localStorage.getItem('nmri-recent-models') ?? '[]',
  ),
  isSearching: false,
  error: null,
  tlOnly: false,

  setQuery: (q) => set({ query: q }),
  setTlOnly: (v) => set({ tlOnly: v }),

  search: async (q) => {
    if (!q.trim()) {
      set({ results: [] });
      return;
    }
    set({ isSearching: true, error: null });
    try {
      const results = await api.model.search(q, 20, get().tlOnly);
      set({ results, isSearching: false });
    } catch (e) {
      set({ error: (e as Error).message, isSearching: false });
    }
  },

  addRecentModel: (modelId) => {
    const current = get().recentModels.filter((id) => id !== modelId);
    const updated = [modelId, ...current].slice(0, 10);
    localStorage.setItem('nmri-recent-models', JSON.stringify(updated));
    set({ recentModels: updated });
  },

  clearResults: () => set({ results: [], query: '' }),
}));
