import type { ScanMode } from './model';
import type {
  ActivationData,
  AnomalyData,
  CircuitData,
  StructuralData,
  WeightData,
} from './scan';

export interface Participant {
  id: string;
  display_name: string;
  role: 'host' | 'viewer';
  color: string;
}

export interface PeerCursor {
  participantId: string;
  displayName: string;
  color: string;
  x: number;
  y: number;
  lastUpdated: number;
}

export interface ScanState {
  mode: ScanMode;
  prompt: string;
  structuralData: StructuralData | null;
  weightData: WeightData | null;
  activationData: ActivationData | null;
  circuitData: CircuitData | null;
  anomalyData: AnomalyData | null;
  selectedTokenIdx: number;
  isScanning: boolean;
}

// ── Client → Server ──

export type CollabC2S =
  | { type: 'c2s_join'; display_name: string; host_id?: string }
  | { type: 'c2s_leave' }
  | { type: 'c2s_scan_state'; payload: ScanState }
  | { type: 'c2s_cursor'; x: number; y: number }
  | { type: 'c2s_select_layer'; layer_id: string | null }
  | { type: 'c2s_select_token'; token_idx: number }
  | { type: 'c2s_ping' };

// ── Server → Client ──

export type CollabS2C =
  | {
      type: 's2c_joined';
      session_id: string;
      participant_id: string;
      role: 'host' | 'viewer';
      participants: Participant[];
      scan_state: ScanState | null;
    }
  | { type: 's2c_participant_joined'; participant: Participant }
  | { type: 's2c_participant_left'; participant_id: string }
  | { type: 's2c_scan_state'; payload: ScanState }
  | {
      type: 's2c_cursor';
      participant_id: string;
      display_name: string;
      x: number;
      y: number;
    }
  | { type: 's2c_select_layer'; participant_id: string; layer_id: string | null }
  | { type: 's2c_select_token'; participant_id: string; token_idx: number }
  | { type: 's2c_error'; message: string }
  | { type: 's2c_session_closed'; reason: string }
  | { type: 's2c_pong' };
