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
}

export interface BatteryResult {
  model_id: string;
  total_tests: number;
  passed: number;
  failed: number;
  results: TestResult[];
  summary: string;
}
