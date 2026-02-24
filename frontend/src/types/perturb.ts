export interface TokenPrediction {
  token: string;
  logit: number;
  prob: number;
}

export interface PerturbResult {
  model_id: string;
  component: string;
  perturbation_type: string;
  original: TokenPrediction;
  perturbed: TokenPrediction;
  top_k_original: TokenPrediction[];
  top_k_perturbed: TokenPrediction[];
  logit_diff: number;
  kl_divergence: number;
  metadata: Record<string, unknown>;
}

export interface PatchResult {
  model_id: string;
  component: string;
  clean_prompt: string;
  corrupt_prompt: string;
  clean_prediction: TokenPrediction;
  corrupt_prediction: TokenPrediction;
  patched_prediction: TokenPrediction;
  recovery_score: number;
  metadata: Record<string, unknown>;
}
