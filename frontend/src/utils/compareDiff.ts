import type { ActivationData, AnomalyData, CircuitData } from '../types/scan';
import type { CompareData, LayerDiff } from '../types/compare';
import type { ScanMode } from '../types/model';

export interface ScanDataPair {
  activationData: ActivationData | null;
  circuitData: CircuitData | null;
  anomalyData: AnomalyData | null;
}

/**
 * Compute layer-wise diffs between two scan datasets.
 * Works for both multi-prompt (same model) and cross-model comparisons.
 */
export function computeLayerDiffs(
  mode: ScanMode,
  labelA: string,
  labelB: string,
  tokensA: string[],
  tokensB: string[],
  dataA: ScanDataPair,
  dataB: ScanDataPair,
): CompareData | null {
  if (mode === 'fMRI' && dataA.activationData && dataB.activationData) {
    const a = dataA.activationData;
    const b = dataB.activationData;
    const diffs: LayerDiff[] = [];
    for (let i = 0; i < Math.min(a.layers.length, b.layers.length); i++) {
      const aActs = a.layers[i].activations;
      const bActs = b.layers[i].activations;
      const avgA = aActs.reduce((s, v) => s + v, 0) / Math.max(aActs.length, 1);
      const avgB = bActs.reduce((s, v) => s + v, 0) / Math.max(bActs.length, 1);
      diffs.push({ layer_id: a.layers[i].layer_id, delta: avgB - avgA });
    }
    const maxAbs = Math.max(...diffs.map((d) => Math.abs(d.delta)), 0.001);
    return { promptA: labelA, promptB: labelB, tokens_a: tokensA, tokens_b: tokensB, layerDiffs: diffs, maxAbsDelta: maxAbs };
  }

  if (mode === 'DTI' && dataA.circuitData && dataB.circuitData) {
    const a = dataA.circuitData;
    const b = dataB.circuitData;
    const aMap = new Map(a.components.map((c) => [c.layer_id, c.importance]));
    const diffs: LayerDiff[] = b.components.map((c) => ({
      layer_id: c.layer_id,
      delta: c.importance - (aMap.get(c.layer_id) ?? 0),
    }));
    const maxAbs = Math.max(...diffs.map((d) => Math.abs(d.delta)), 0.001);
    return { promptA: labelA, promptB: labelB, tokens_a: tokensA, tokens_b: tokensB, layerDiffs: diffs, maxAbsDelta: maxAbs };
  }

  if (mode === 'FLAIR' && dataA.anomalyData && dataB.anomalyData) {
    const a = dataA.anomalyData;
    const b = dataB.anomalyData;
    const diffs: LayerDiff[] = [];
    for (let i = 0; i < Math.min(a.layers.length, b.layers.length); i++) {
      const aScores = a.layers[i].anomaly_scores;
      const bScores = b.layers[i].anomaly_scores;
      const avgA = aScores.reduce((s, v) => s + v, 0) / Math.max(aScores.length, 1);
      const avgB = bScores.reduce((s, v) => s + v, 0) / Math.max(bScores.length, 1);
      diffs.push({ layer_id: a.layers[i].layer_id, delta: avgB - avgA });
    }
    const maxAbs = Math.max(...diffs.map((d) => Math.abs(d.delta)), 0.001);
    return { promptA: labelA, promptB: labelB, tokens_a: tokensA, tokens_b: tokensB, layerDiffs: diffs, maxAbsDelta: maxAbs };
  }

  return null;
}
