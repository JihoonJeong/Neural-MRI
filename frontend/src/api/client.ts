import type { ModelInfo } from '../types/model';
import type { ActivationData, AnomalyData, CircuitData, SAEData, SAEInfoResponse, StructuralData, WeightData } from '../types/scan';
import type { PerturbResult, PatchResult } from '../types/perturb';
import type { CausalTraceResult } from '../types/causalTrace';
import type { DiagnosticReport, ReportRequest } from '../types/report';
import type { BatteryResult, TestCase } from '../types/battery';
import type { TokenStatus, CacheStatus, HubSearchResult } from '../types/settings';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api';

class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new ApiError(res.status, body.detail || res.statusText);
  }
  return res.json();
}

export interface ModelListEntry {
  model_id: string;
  display_name: string;
  family: string;
  params: string;
  tl_compat: boolean;
  gated: boolean;
  is_loaded: boolean;
  source?: 'registry' | 'dynamic';
}

export const api = {
  model: {
    list: () => request<ModelListEntry[]>('/model/list'),
    load: (model_id: string, device = 'auto') =>
      request<ModelInfo>('/model/load', {
        method: 'POST',
        body: JSON.stringify({ model_id, device }),
      }),
    info: () => request<ModelInfo>('/model/info'),
    unload: () => request<{ status: string }>('/model/unload', { method: 'DELETE' }),
    search: (q: string, limit = 20, tlOnly = false) =>
      request<HubSearchResult[]>(
        `/model/search?q=${encodeURIComponent(q)}&limit=${limit}&tl_only=${tlOnly}`,
      ),
  },
  scan: {
    structural: () =>
      request<StructuralData>('/scan/structural', { method: 'POST' }),
    weights: (layers?: string[]) =>
      request<WeightData>('/scan/weights', {
        method: 'POST',
        body: JSON.stringify({ layers: layers ?? null }),
      }),
    activation: (prompt: string) =>
      request<ActivationData>('/scan/activation', {
        method: 'POST',
        body: JSON.stringify({ prompt }),
      }),
    circuits: (prompt: string, targetTokenIdx = -1) =>
      request<CircuitData>('/scan/circuits', {
        method: 'POST',
        body: JSON.stringify({ prompt, target_token_idx: targetTokenIdx }),
      }),
    anomaly: (prompt: string) =>
      request<AnomalyData>('/scan/anomaly', {
        method: 'POST',
        body: JSON.stringify({ prompt }),
      }),
  },
  perturb: {
    zero: (component: string, prompt: string) =>
      request<PerturbResult>('/perturb/zero', {
        method: 'POST',
        body: JSON.stringify({ component, prompt }),
      }),
    amplify: (component: string, prompt: string, factor = 2.0) =>
      request<PerturbResult>('/perturb/amplify', {
        method: 'POST',
        body: JSON.stringify({ component, factor, prompt }),
      }),
    ablate: (component: string, prompt: string) =>
      request<PerturbResult>('/perturb/ablate', {
        method: 'POST',
        body: JSON.stringify({ component, prompt }),
      }),
    patch: (cleanPrompt: string, corruptPrompt: string, component: string, targetIdx = -1) =>
      request<PatchResult>('/perturb/patch', {
        method: 'POST',
        body: JSON.stringify({
          clean_prompt: cleanPrompt,
          corrupt_prompt: corruptPrompt,
          component,
          target_token_idx: targetIdx,
        }),
      }),
    causalTrace: (cleanPrompt: string, corruptPrompt: string, targetIdx = -1) =>
      request<CausalTraceResult>('/perturb/causal-trace', {
        method: 'POST',
        body: JSON.stringify({
          clean_prompt: cleanPrompt,
          corrupt_prompt: corruptPrompt,
          target_token_idx: targetIdx,
        }),
      }),
    reset: () =>
      request<{ status: string }>('/perturb/reset', { method: 'POST' }),
  },
  report: {
    generate: (req: ReportRequest = {}) =>
      request<DiagnosticReport>('/report/generate', {
        method: 'POST',
        body: JSON.stringify(req),
      }),
  },
  battery: {
    run: (categories?: string[], locale?: string, includeSae?: boolean, saeLayer?: number | null) =>
      request<BatteryResult>('/battery/run', {
        method: 'POST',
        body: JSON.stringify({
          categories: categories ?? null,
          locale: locale ?? 'en',
          include_sae: includeSae ?? false,
          sae_layer: saeLayer ?? null,
        }),
      }),
    tests: () => request<TestCase[]>('/battery/tests'),
  },
  sae: {
    info: () => request<SAEInfoResponse>('/sae/info'),
    support: () => request<Record<string, boolean>>('/sae/support'),
    scan: (prompt: string, layerIdx: number, topK = 20) =>
      request<SAEData>('/sae/scan', {
        method: 'POST',
        body: JSON.stringify({ prompt, layer_idx: layerIdx, top_k: topK }),
      }),
  },
  collab: {
    create: (displayName = 'Host') =>
      request<{ session_id: string; host_id: string; join_url: string }>(
        `/collab/create?display_name=${encodeURIComponent(displayName)}`,
        { method: 'POST' },
      ),
    get: (sessionId: string) =>
      request<{
        session_id: string;
        host_id: string;
        host_name: string;
        participant_count: number;
      }>(`/collab/${sessionId}`),
    delete: (sessionId: string) =>
      request<{ status: string }>(`/collab/${sessionId}`, { method: 'DELETE' }),
    list: () =>
      request<
        Array<{
          session_id: string;
          host_name: string;
          participant_count: number;
          created_at: string;
        }>
      >('/collab/list'),
  },
  settings: {
    updateToken: (token: string) =>
      request<TokenStatus>('/settings/token', {
        method: 'POST',
        body: JSON.stringify({ token }),
      }),
    clearToken: () =>
      request<TokenStatus>('/settings/token', { method: 'DELETE' }),
    tokenStatus: () => request<TokenStatus>('/settings/token/status'),
    cacheStatus: () => request<CacheStatus>('/settings/cache'),
    clearCache: () =>
      request<{ status: string }>('/settings/cache', { method: 'DELETE' }),
  },
};
