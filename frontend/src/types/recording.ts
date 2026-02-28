import type { ScanMode } from './model';
import type { ActivationData, AnomalyData, CircuitData, StructuralData, WeightData } from './scan';

export interface RecordingFrame {
  timestamp: number; // ms elapsed since recording start
  mode: ScanMode;
  selectedTokenIdx: number;
  structuralData: StructuralData | null;
  weightData: WeightData | null;
  activationData: ActivationData | null;
  circuitData: CircuitData | null;
  anomalyData: AnomalyData | null;
}

export interface ScanRecording {
  version: 1;
  modelId: string;
  prompt: string;
  createdAt: string; // ISO timestamp
  duration: number; // total ms
  frames: RecordingFrame[];
}

export type PlaybackSpeed = 0.5 | 1 | 2 | 4;
