export interface CausalTraceCell {
  component: string;
  layer_idx: number;
  component_type: 'attn' | 'mlp' | 'embed';
  recovery_score: number;
}

export interface CausalTraceResult {
  model_id: string;
  clean_prompt: string;
  corrupt_prompt: string;
  target_token_idx: number;
  clean_prediction: string;
  corrupt_prediction: string;
  cells: CausalTraceCell[];
  n_layers: number;
  metadata: Record<string, unknown>;
}
