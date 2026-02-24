export interface LayerConfig {
  layer_index: number;
  layer_type: string;
  components: string[];
  num_heads: number;
  d_model: number;
  d_head: number;
  d_mlp: number;
}

export interface ModelInfo {
  model_id: string;
  model_name: string;
  n_params: number;
  n_layers: number;
  d_model: number;
  d_vocab: number;
  n_heads: number;
  d_head: number;
  d_mlp: number;
  max_seq_len: number;
  device: string;
  layers: LayerConfig[];
  dtype: string;
}

export type ScanMode = 'T1' | 'T2' | 'fMRI' | 'DTI' | 'FLAIR';

export const SCAN_MODES: Record<ScanMode, { label: string; desc: string; color: string }> = {
  T1: { label: 'T1 Topology', desc: 'Topology Layer 1 — Architecture', color: '#e0e0e0' },
  T2: { label: 'T2 Tensor', desc: 'Tensor Layer 2 — Weight Distribution', color: '#aaccee' },
  fMRI: { label: 'fMRI', desc: 'functional Model Resonance Imaging', color: '#ff6644' },
  DTI: { label: 'DTI', desc: 'Data Tractography Imaging', color: '#44ddaa' },
  FLAIR: { label: 'FLAIR', desc: 'Feature-Level Anomaly Identification', color: '#ff4466' },
};
