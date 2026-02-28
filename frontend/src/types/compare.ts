export interface LayerDiff {
  layer_id: string;
  delta: number; // B - A (positive = B stronger)
}

export interface CompareData {
  promptA: string;
  promptB: string;
  tokens_a: string[];
  tokens_b: string[];
  layerDiffs: LayerDiff[];
  maxAbsDelta: number;
}
