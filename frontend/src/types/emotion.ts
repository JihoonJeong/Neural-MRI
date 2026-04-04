// --- Emotion Vector Analysis ---

export interface EmotionActivation {
  emotion: string;
  activation: number;
  activation_normalized: number;
}

export interface ExtractProbesResponse {
  model_id: string;
  layer_idx: number;
  mode: string;
  n_emotions: number;
  emotions: string[];
  metadata: Record<string, unknown>;
}

export interface SteerComparison {
  original_text: string;
  steered_text: string;
  emotion: string;
  strength: number;
  layer_range: number[];
}

export interface SteerResponse {
  model_id: string;
  prompt: string;
  comparison: SteerComparison;
  original_emotions: EmotionActivation[];
  steered_emotions: EmotionActivation[];
  metadata: Record<string, unknown>;
}

export interface EmotionListResponse {
  emotions: string[];
  n_emotions: number;
  passages_per_emotion: Record<string, number>;
  model_id: string | null;
  has_probes: boolean;
  probe_layers: number[];
}
