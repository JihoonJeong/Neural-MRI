export interface LayerStructure {
  layer_id: string;
  layer_type: 'embedding' | 'attention' | 'mlp' | 'output';
  layer_index: number | null;
  param_count: number;
  shape_info: Record<string, number>;
}

export interface ConnectionInfo {
  from_id: string;
  to_id: string;
  type: string;
}

export interface StructuralData {
  model_id: string;
  scan_mode: 'T1';
  total_params: number;
  layers: LayerStructure[];
  connections: ConnectionInfo[];
  metadata: Record<string, unknown>;
}

export interface LayerWeightStats {
  layer_id: string;
  component: string;
  mean: number;
  std: number;
  min_val: number;
  max_val: number;
  l2_norm: number;
  shape: number[];
  num_outliers: number;
  histogram: number[];
}

export interface WeightData {
  model_id: string;
  scan_mode: 'T2';
  layers: LayerWeightStats[];
  metadata: Record<string, unknown>;
}

// --- fMRI: Activation Scan ---

export interface LayerActivation {
  layer_id: string;
  activations: number[]; // per-token scalar (0-1 normalized)
  per_head?: number[][] | null; // [n_heads][n_tokens] for attn layers
}

export interface ActivationData {
  model_id: string;
  scan_mode: 'fMRI';
  tokens: string[];
  layers: LayerActivation[];
  metadata: Record<string, unknown>;
}

// --- DTI: Circuit Scan ---

export interface PathwayConnection {
  from_id: string;
  to_id: string;
  strength: number;
  is_pathway: boolean;
}

export interface ComponentImportance {
  layer_id: string;
  importance: number;
  is_pathway: boolean;
}

export interface AttentionHead {
  layer_idx: number;
  head_idx: number;
  pattern: number[][];
}

export interface CircuitData {
  model_id: string;
  scan_mode: 'DTI';
  tokens: string[];
  target_token_idx: number;
  connections: PathwayConnection[];
  components: ComponentImportance[];
  attention_heads: AttentionHead[];
  metadata: Record<string, unknown>;
}

// --- FLAIR: Anomaly Scan ---

export interface TokenPredictionLens {
  token: string;
  prob: number;
}

export interface LayerAnomaly {
  layer_id: string;
  anomaly_scores: number[];
  kl_scores: number[];
  entropy_scores: number[];
  top_predictions?: TokenPredictionLens[][] | null; // [seq_len][top_k]
}

export interface AnomalyData {
  model_id: string;
  scan_mode: 'FLAIR';
  tokens: string[];
  layers: LayerAnomaly[];
  metadata: Record<string, unknown>;
}

// --- SAE: Sparse Autoencoder Feature Scan ---

export interface SAEFeatureInfo {
  feature_idx: number;
  activation: number;
  activation_normalized: number;
  neuronpedia_url: string | null;
}

export interface SAETokenFeatures {
  token_idx: number;
  token_str: string;
  top_features: SAEFeatureInfo[];
}

export interface SAEData {
  model_id: string;
  scan_mode: 'SAE';
  prompt: string;
  layer_idx: number;
  hook_name: string;
  d_sae: number;
  tokens: string[];
  token_features: SAETokenFeatures[];
  reconstruction_loss: number;
  sparsity: number;
  heatmap_feature_indices: number[];
  heatmap_values: number[][];
  metadata: Record<string, unknown>;
}

export interface SAEInfoResponse {
  available: boolean;
  model_id: string | null;
  release?: string;
  layers: number[];
  d_sae: number;
  has_neuronpedia?: boolean;
}

// --- WebSocket Messages ---

export type WsMessageIn =
  | { type: 'scan_stream'; mode: 'fMRI' | 'DTI'; prompt: string }
  | { type: 'ping' };

export type WsMessageOut =
  | { type: 'info'; message: string }
  | { type: 'scan_start'; mode: string; tokens: string[]; n_layers: number; seq_len: number }
  | { type: 'activation_frame'; token_idx: number; layers: { layer_id: string; activation: number }[] }
  | { type: 'attention_pattern'; layer_idx: number; head_idx: number; pattern: number[][] }
  | { type: 'component_importance'; layer_id: string; importance: number; is_pathway: boolean }
  | { type: 'scan_complete'; compute_time_ms: number }
  | { type: 'error'; message: string }
  | { type: 'pong' };
