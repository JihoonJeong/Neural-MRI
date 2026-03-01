export interface TokenStatus {
  is_set: boolean;
  is_valid: boolean | null;
  source: 'runtime' | 'env' | 'none';
}

export interface CacheStatus {
  entry_count: number;
  max_entries: number;
}

export interface HubSearchResult {
  model_id: string;
  author: string | null;
  downloads: number;
  likes: number;
  pipeline_tag: string | null;
  gated: boolean | string;
  tl_compat: boolean | null;
  architectures: string[];
}
