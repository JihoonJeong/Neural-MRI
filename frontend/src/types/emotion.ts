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

// --- Token-level Projection ---

export interface TokenEmotionProfile {
  token_idx: number;
  token_str: string;
  activations: Record<string, number>;
}

export interface DisplayToken {
  label: string;
  token_indices: number[];
  activations: Record<string, number>;
}

export interface ProjectResponse {
  model_id: string;
  prompt: string;
  layer_idx: number;
  emotions: string[];
  tokens: TokenEmotionProfile[];
  display_tokens: DisplayToken[];
  metadata: Record<string, unknown>;
}

// --- PCA ---

export interface PCAResponse {
  model_id: string;
  layer_idx: number;
  emotions: string[];
  pc1: number[];
  pc2: number[];
  variance_explained: number[];
  metadata: Record<string, unknown>;
}

// --- Sweep ---

export interface SweepPoint {
  strength: number;
  target_emotion_activation: number;
  generated_text: string;
}

export interface SweepResponse {
  model_id: string;
  prompt: string;
  emotion: string;
  points: SweepPoint[];
  metadata: Record<string, unknown>;
}

// --- Layer Evolution ---

export interface LayerEmotionPoint {
  layer_idx: number;
  activations: Record<string, number>;
}

export interface LayerEvolutionResponse {
  model_id: string;
  prompt: string;
  token_idx: number;
  token_str: string;
  emotions: string[];
  layers: LayerEmotionPoint[];
  metadata: Record<string, unknown>;
}

// --- Steer + SAE ---

export interface SAEFeatureDiff {
  feature_idx: number;
  original_activation: number;
  steered_activation: number;
  diff: number;
}

export interface SteerSAEResponse {
  model_id: string;
  prompt: string;
  emotion: string;
  strength: number;
  sae_layer_idx: number;
  sae_hook_name: string;
  d_sae: number;
  original_top_features: SAEFeatureDiff[];
  steered_top_features: SAEFeatureDiff[];
  top_changed_features: SAEFeatureDiff[];
  metadata: Record<string, unknown>;
}

// --- List ---

export interface EmotionListResponse {
  emotions: string[];
  n_emotions: number;
  passages_per_emotion: Record<string, number>;
  model_id: string | null;
  has_probes: boolean;
  probe_layers: number[];
}
