export interface TestCase {
  test_id: string;
  category: string;
  name: string;
  prompt: string;
  expected_behavior: string;
  expected_tokens?: string[];
  unexpected_tokens?: string[];
  compare_prompts?: string[];
}

export interface TestPrediction {
  token: string;
  prob: number;
}

export interface ActivationSummary {
  peak_layers: number[];
  peak_activation: number;
  active_layer_count: number;
}

export interface CompareResult {
  prompt: string;
  top_k: TestPrediction[];
  pronoun_probs?: Record<string, number>;
}

export interface SAEFeatureBrief {
  feature_idx: number;
  activation: number;
  activation_normalized: number;
  neuronpedia_url: string | null;
}

export interface TestSAEResult {
  layer_idx: number;
  hook_name: string;
  top_features: SAEFeatureBrief[];
}

export interface CrossTestFeature {
  feature_idx: number;
  neuronpedia_url: string | null;
  test_ids: string[];
  categories: string[];
  count: number;
  avg_activation: number;
  max_activation: number;
}

export interface BatterySAESummary {
  layer_idx: number;
  d_sae: number;
  total_unique_features: number;
  cross_test_features: CrossTestFeature[];
  per_category_top_features: Record<string, number[]>;
  interpretation: string;
}

export interface TestResult {
  test_id: string;
  category: string;
  name: string;
  prompt: string;
  passed: boolean;
  top_k: TestPrediction[];
  actual_token: string;
  actual_prob: number;
  expected_prob?: number;
  activation_summary: ActivationSummary;
  interpretation: string;
  compare_results?: CompareResult[];
  sae_features?: TestSAEResult;
}

export interface BatteryResult {
  model_id: string;
  total_tests: number;
  passed: number;
  failed: number;
  results: TestResult[];
  summary: string;
  sae_summary?: BatterySAESummary;
}
