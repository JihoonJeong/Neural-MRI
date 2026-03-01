import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('API client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('model.list calls GET /api/model/list', async () => {
    const mockData = [{ model_id: 'gpt2', display_name: 'GPT-2' }];
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    });

    const { api } = await import('../api/client');
    const result = await api.model.list();
    expect(result).toEqual(mockData);
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/model/list'),
      expect.objectContaining({ headers: { 'Content-Type': 'application/json' } }),
    );
  });

  it('scan.structural calls POST /api/scan/structural', async () => {
    const mockData = { layers: [] };
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockData),
    });

    const { api } = await import('../api/client');
    const result = await api.scan.structural();
    expect(result).toEqual(mockData);
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/scan/structural'),
      expect.objectContaining({ method: 'POST' }),
    );
  });

  it('throws ApiError on non-ok response', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: () => Promise.resolve({ detail: 'Model not loaded' }),
    });

    const { api } = await import('../api/client');
    await expect(api.model.info()).rejects.toThrow('Model not loaded');
  });

  it('scan.activation sends prompt in body', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ tokens: [], layers: [] }),
    });

    const { api } = await import('../api/client');
    await api.scan.activation('Hello world');
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/scan/activation'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ prompt: 'Hello world' }),
      }),
    );
  });

  it('perturb.zero sends component and prompt', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ original_logits: {}, perturbed_logits: {}, kl_divergence: 0 }),
    });

    const { api } = await import('../api/client');
    await api.perturb.zero('blocks.3.attn', 'test prompt');
    expect(fetch).toHaveBeenCalledWith(
      expect.stringContaining('/perturb/zero'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ component: 'blocks.3.attn', prompt: 'test prompt' }),
      }),
    );
  });
});
