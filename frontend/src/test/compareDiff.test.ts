import { describe, it, expect } from 'vitest';
import { computeLayerDiffs, type ScanDataPair } from '../utils/compareDiff';
import type { ActivationData, CircuitData, AnomalyData } from '../types/scan';

function empty(): ScanDataPair {
  return { activationData: null, circuitData: null, anomalyData: null };
}

describe('computeLayerDiffs', () => {
  it('returns null for T1 mode (no diff supported)', () => {
    const result = computeLayerDiffs('T1', 'A', 'B', [], [], empty(), empty());
    expect(result).toBeNull();
  });

  it('returns null for T2 mode (no diff supported)', () => {
    const result = computeLayerDiffs('T2', 'A', 'B', [], [], empty(), empty());
    expect(result).toBeNull();
  });

  it('computes fMRI diffs correctly', () => {
    const dataA = {
      tokens: ['The', 'cat'],
      layers: [
        { layer_id: 'embed', activations: [0.2, 0.3] },
        { layer_id: 'blocks.0.attn', activations: [0.4, 0.6] },
      ],
    } as unknown as ActivationData;
    const dataB = {
      tokens: ['The', 'dog'],
      layers: [
        { layer_id: 'embed', activations: [0.5, 0.5] },
        { layer_id: 'blocks.0.attn', activations: [0.8, 0.2] },
      ],
    } as unknown as ActivationData;

    const result = computeLayerDiffs(
      'fMRI', 'prompt A', 'prompt B',
      dataA.tokens, dataB.tokens,
      { ...empty(), activationData: dataA },
      { ...empty(), activationData: dataB },
    );

    expect(result).not.toBeNull();
    expect(result!.promptA).toBe('prompt A');
    expect(result!.promptB).toBe('prompt B');
    expect(result!.layerDiffs).toHaveLength(2);
    // embed: avgA = 0.25, avgB = 0.5, delta = 0.25
    expect(result!.layerDiffs[0].delta).toBeCloseTo(0.25);
    // blocks.0.attn: avgA = 0.5, avgB = 0.5, delta = 0
    expect(result!.layerDiffs[1].delta).toBeCloseTo(0);
    expect(result!.maxAbsDelta).toBeCloseTo(0.25);
  });

  it('computes DTI diffs correctly', () => {
    const dataA = {
      tokens: ['a'],
      components: [
        { layer_id: 'blocks.0.attn', importance: 0.8 },
        { layer_id: 'blocks.0.mlp', importance: 0.3 },
      ],
      connections: [],
    } as unknown as CircuitData;
    const dataB = {
      tokens: ['b'],
      components: [
        { layer_id: 'blocks.0.attn', importance: 0.5 },
        { layer_id: 'blocks.0.mlp', importance: 0.9 },
      ],
      connections: [],
    } as unknown as CircuitData;

    const result = computeLayerDiffs(
      'DTI', 'A', 'B', ['a'], ['b'],
      { ...empty(), circuitData: dataA },
      { ...empty(), circuitData: dataB },
    );

    expect(result).not.toBeNull();
    expect(result!.layerDiffs).toHaveLength(2);
    expect(result!.layerDiffs[0].delta).toBeCloseTo(-0.3);
    expect(result!.layerDiffs[1].delta).toBeCloseTo(0.6);
    expect(result!.maxAbsDelta).toBeCloseTo(0.6);
  });

  it('computes FLAIR diffs correctly', () => {
    const dataA = {
      tokens: ['x'],
      layers: [{ layer_id: 'blocks.0', anomaly_scores: [0.1, 0.2], kl_scores: [], entropy_scores: [] }],
    } as unknown as AnomalyData;
    const dataB = {
      tokens: ['y'],
      layers: [{ layer_id: 'blocks.0', anomaly_scores: [0.5, 0.6], kl_scores: [], entropy_scores: [] }],
    } as unknown as AnomalyData;

    const result = computeLayerDiffs(
      'FLAIR', 'A', 'B', ['x'], ['y'],
      { ...empty(), anomalyData: dataA },
      { ...empty(), anomalyData: dataB },
    );

    expect(result).not.toBeNull();
    expect(result!.layerDiffs[0].delta).toBeCloseTo(0.4);
  });

  it('returns null when data is missing for the mode', () => {
    const result = computeLayerDiffs('fMRI', 'A', 'B', [], [], empty(), empty());
    expect(result).toBeNull();
  });
});
