import { create } from 'zustand';
import { ScanSocket } from '../api/ws';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';
import type { WsMessageOut } from '../types/scan';
import type {
  LayerActivation,
  ActivationData,
  CircuitData,
  AttentionHead,
  ComponentImportance,
  PathwayConnection,
} from '../types/scan';

export type StreamStatus = 'idle' | 'connecting' | 'streaming' | 'complete' | 'error';

interface StreamState {
  status: StreamStatus;
  currentToken: number;
  totalTokens: number;
  tokens: string[];
  errorMessage: string | null;
  isStreamPreferred: boolean;

  startStream: (mode: 'fMRI' | 'DTI', prompt: string) => Promise<void>;
  cancelStream: () => void;
  setStreamPreferred: (v: boolean) => void;
}

// Module-level socket reference
let _socket: ScanSocket | null = null;

// Frame accumulation state (module-level, reset per stream)
let _layerMap: Map<string, number[]> = new Map();
let _attentionHeads: AttentionHead[] = [];
let _componentImportances: ComponentImportance[] = [];
let _streamTokens: string[] = [];
let _nLayers = 0;

function _resetAccumulators() {
  _layerMap = new Map();
  _attentionHeads = [];
  _componentImportances = [];
  _streamTokens = [];
  _nLayers = 0;
}

function _buildActivationData(): ActivationData {
  const modelId = useScanStore.getState().structuralData?.model_id ?? 'unknown';
  const layers: LayerActivation[] = [];

  for (const [layerId, activations] of _layerMap) {
    layers.push({ layer_id: layerId, activations });
  }

  return {
    model_id: modelId,
    scan_mode: 'fMRI',
    tokens: _streamTokens,
    layers,
    metadata: { source: 'stream' },
  };
}

function _buildCircuitData(tokenIdx: number): CircuitData {
  const modelId = useScanStore.getState().structuralData?.model_id ?? 'unknown';

  // Build pathway connections from component importances
  const connections: PathwayConnection[] = [];
  const compIds = _componentImportances.map((c) => c.layer_id);

  for (let i = 0; i < compIds.length - 1; i++) {
    const fromImp = _componentImportances[i];
    const toImp = _componentImportances[i + 1];
    const strength = (fromImp.importance + toImp.importance) / 2;
    connections.push({
      from_id: fromImp.layer_id,
      to_id: toImp.layer_id,
      strength,
      is_pathway: fromImp.is_pathway && toImp.is_pathway,
    });
  }

  return {
    model_id: modelId,
    scan_mode: 'DTI',
    tokens: _streamTokens,
    target_token_idx: tokenIdx,
    connections,
    components: [..._componentImportances],
    attention_heads: [..._attentionHeads],
    metadata: { source: 'stream' },
  };
}

export const useStreamStore = create<StreamState>((set, get) => ({
  status: 'idle',
  currentToken: 0,
  totalTokens: 0,
  tokens: [],
  errorMessage: null,
  isStreamPreferred: localStorage.getItem('nmri_stream_preferred') === 'true',

  setStreamPreferred: (v) => {
    localStorage.setItem('nmri_stream_preferred', String(v));
    set({ isStreamPreferred: v });
  },

  cancelStream: () => {
    _socket?.close();
    _socket = null;
    _resetAccumulators();
    useScanStore.setState({ isScanning: false });
    set({ status: 'idle', currentToken: 0, totalTokens: 0, tokens: [], errorMessage: null });
  },

  startStream: async (mode, prompt) => {
    const scanStore = useScanStore.getState();

    // Cancel any existing stream
    if (_socket) {
      _socket.close();
      _socket = null;
    }
    _resetAccumulators();

    set({ status: 'connecting', currentToken: 0, totalTokens: 0, tokens: [], errorMessage: null });
    scanStore.addLog(`Stream ${mode}: connecting...`);

    // Ensure structural data is loaded
    if (!scanStore.structuralData) {
      try {
        const sData = await api.scan.structural();
        useScanStore.setState({ structuralData: sData });
        scanStore.addLog('T1 auto-loaded for layout');
      } catch (e) {
        set({ status: 'error', errorMessage: (e as Error).message });
        scanStore.addLog(`Stream failed: ${(e as Error).message}`);
        return;
      }
    }

    useScanStore.setState({ isScanning: true });

    const streamMode = mode;

    const handleMessage = (msg: WsMessageOut) => {
      switch (msg.type) {
        case 'scan_start': {
          _streamTokens = msg.tokens;
          _nLayers = msg.n_layers;
          set({
            status: 'streaming',
            tokens: msg.tokens,
            totalTokens: streamMode === 'fMRI' ? msg.seq_len : _nLayers * 2 + 1,
            currentToken: 0,
          });
          useScanStore.setState({ tokenCount: msg.tokens.length });
          scanStore.addLog(`Stream started: ${msg.tokens.length} tokens, ${msg.n_layers} layers`);
          break;
        }

        case 'activation_frame': {
          // Accumulate activation data
          for (const { layer_id, activation } of msg.layers) {
            if (!_layerMap.has(layer_id)) _layerMap.set(layer_id, []);
            _layerMap.get(layer_id)![msg.token_idx] = activation;
          }

          const tokenIdx = msg.token_idx;
          set({ currentToken: tokenIdx + 1 });

          // Build and publish partial ActivationData
          const activationData = _buildActivationData();
          useScanStore.setState({
            activationData,
            selectedTokenIdx: tokenIdx,
          });
          break;
        }

        case 'attention_pattern': {
          _attentionHeads.push({
            layer_idx: msg.layer_idx,
            head_idx: msg.head_idx,
            pattern: msg.pattern,
          });
          // Update current progress (attention patterns come first in DTI)
          const attnProgress = Math.floor(_attentionHeads.length / ((_nLayers > 0 ? _nLayers : 1) * 12));
          set({ currentToken: Math.min(attnProgress, get().totalTokens) });
          break;
        }

        case 'component_importance': {
          _componentImportances.push({
            layer_id: msg.layer_id,
            importance: msg.importance,
            is_pathway: msg.is_pathway,
          });

          set({ currentToken: _componentImportances.length });

          // Build and publish partial CircuitData
          const targetIdx = _streamTokens.length > 0 ? _streamTokens.length - 1 : 0;
          const circuitData = _buildCircuitData(targetIdx);
          useScanStore.setState({
            circuitData,
            selectedTokenIdx: 0,
          });
          break;
        }

        case 'scan_complete': {
          set({ status: 'complete' });
          useScanStore.setState({ isScanning: false });
          scanStore.addLog(`Stream complete (${msg.compute_time_ms}ms)`);
          _socket?.close();
          _socket = null;
          break;
        }

        case 'error': {
          set({ status: 'error', errorMessage: msg.message });
          useScanStore.setState({ isScanning: false });
          scanStore.addLog(`Stream error: ${msg.message}`);
          _socket?.close();
          _socket = null;
          break;
        }

        case 'info':
        case 'pong':
          break;
      }
    };

    const handleDisconnect = () => {
      const { status } = get();
      if (status === 'streaming' || status === 'connecting') {
        set({ status: 'error', errorMessage: 'Connection lost' });
        useScanStore.setState({ isScanning: false });
        scanStore.addLog('Stream disconnected unexpectedly');
      }
    };

    _socket = new ScanSocket(handleMessage, handleDisconnect);

    try {
      await _socket.connect();
      _socket.send({ type: 'scan_stream', mode, prompt });
    } catch (e) {
      set({ status: 'error', errorMessage: (e as Error).message });
      useScanStore.setState({ isScanning: false });
      scanStore.addLog(`Stream connection failed: ${(e as Error).message}`);
      _socket = null;
    }
  },
}));
