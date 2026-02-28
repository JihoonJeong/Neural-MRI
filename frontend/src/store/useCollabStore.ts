import { create } from 'zustand';
import type { Participant, PeerCursor, ScanState } from '../types/collab';
import { CollabSocket } from '../api/collabSocket';
import { api } from '../api/client';
import { useScanStore } from './useScanStore';

interface CollabState {
  // Session
  isInSession: boolean;
  sessionId: string | null;
  participantId: string | null;
  hostId: string | null;
  role: 'host' | 'viewer' | null;
  participants: Participant[];
  peerCursors: Map<string, PeerCursor>;

  // Remote scan state (viewers only)
  remoteScanState: ScanState | null;

  // Connection
  socket: CollabSocket | null;
  isConnecting: boolean;
  error: string | null;

  // Actions
  createSession: (displayName?: string) => Promise<string>;
  joinSession: (sessionId: string, displayName?: string) => Promise<void>;
  leaveSession: () => void;
  broadcastScanState: () => void;
  sendCursor: (x: number, y: number) => void;
  sendLayerSelection: (layerId: string | null) => void;
  sendTokenSelection: (tokenIdx: number) => void;
}

let _scanUnsub: (() => void) | null = null;
let _lastCursorSend = 0;

function _buildScanState(): ScanState {
  const s = useScanStore.getState();
  return {
    mode: s.mode,
    prompt: s.prompt,
    structuralData: s.structuralData,
    weightData: s.weightData,
    activationData: s.activationData,
    circuitData: s.circuitData,
    anomalyData: s.anomalyData,
    selectedTokenIdx: s.selectedTokenIdx,
    isScanning: s.isScanning,
  };
}

export const useCollabStore = create<CollabState>((set, get) => ({
  isInSession: false,
  sessionId: null,
  participantId: null,
  hostId: null,
  role: null,
  participants: [],
  peerCursors: new Map(),
  remoteScanState: null,
  socket: null,
  isConnecting: false,
  error: null,

  createSession: async (displayName = 'Host') => {
    set({ isConnecting: true, error: null });
    try {
      const { session_id, host_id } = await api.collab.create(displayName);

      const socket = new CollabSocket(
        (msg) => _handleMessage(msg, set, get),
        () => _handleDisconnect(set),
      );
      await socket.connect(session_id);
      socket.send({ type: 'c2s_join', display_name: displayName, host_id });

      set({
        socket,
        sessionId: session_id,
        hostId: host_id,
        isConnecting: false,
      });

      // Subscribe to scan store changes as host
      _setupHostSubscription(get);

      return session_id;
    } catch (e) {
      set({ isConnecting: false, error: (e as Error).message });
      throw e;
    }
  },

  joinSession: async (sessionId, displayName = 'Viewer') => {
    set({ isConnecting: true, error: null });
    try {
      const socket = new CollabSocket(
        (msg) => _handleMessage(msg, set, get),
        () => _handleDisconnect(set),
      );
      await socket.connect(sessionId);
      socket.send({ type: 'c2s_join', display_name: displayName });

      set({ socket, sessionId, isConnecting: false });
    } catch (e) {
      set({ isConnecting: false, error: (e as Error).message });
      throw e;
    }
  },

  leaveSession: () => {
    const { socket } = get();
    if (socket) {
      socket.send({ type: 'c2s_leave' });
      socket.close();
    }
    if (_scanUnsub) {
      _scanUnsub();
      _scanUnsub = null;
    }
    set({
      isInSession: false,
      sessionId: null,
      participantId: null,
      hostId: null,
      role: null,
      participants: [],
      peerCursors: new Map(),
      remoteScanState: null,
      socket: null,
      error: null,
    });
  },

  broadcastScanState: () => {
    const { socket, role } = get();
    if (role !== 'host' || !socket?.connected) return;
    socket.send({ type: 'c2s_scan_state', payload: _buildScanState() });
  },

  sendCursor: (x, y) => {
    const now = Date.now();
    if (now - _lastCursorSend < 100) return;
    _lastCursorSend = now;
    get().socket?.send({ type: 'c2s_cursor', x, y });
  },

  sendLayerSelection: (layerId) => {
    get().socket?.send({ type: 'c2s_select_layer', layer_id: layerId });
  },

  sendTokenSelection: (tokenIdx) => {
    get().socket?.send({ type: 'c2s_select_token', token_idx: tokenIdx });
  },
}));

// ── Message handler ──

type Setter = (partial: Partial<CollabState> | ((s: CollabState) => Partial<CollabState>)) => void;
type Getter = () => CollabState;

function _handleMessage(msg: import('../types/collab').CollabS2C, set: Setter, get: Getter) {
  switch (msg.type) {
    case 's2c_joined':
      set({
        isInSession: true,
        participantId: msg.participant_id,
        role: msg.role,
        participants: msg.participants,
        remoteScanState: msg.scan_state,
      });
      break;

    case 's2c_participant_joined':
      set((s) => ({
        participants: [...s.participants, msg.participant],
      }));
      break;

    case 's2c_participant_left':
      set((s) => ({
        participants: s.participants.filter((p) => p.id !== msg.participant_id),
        peerCursors: (() => {
          const next = new Map(s.peerCursors);
          next.delete(msg.participant_id);
          return next;
        })(),
      }));
      break;

    case 's2c_scan_state':
      set({ remoteScanState: msg.payload });
      break;

    case 's2c_cursor': {
      const cursor: PeerCursor = {
        participantId: msg.participant_id,
        displayName: msg.display_name,
        color: get().participants.find((p) => p.id === msg.participant_id)?.color ?? '#aaa',
        x: msg.x,
        y: msg.y,
        lastUpdated: Date.now(),
      };
      set((s) => {
        const next = new Map(s.peerCursors);
        next.set(msg.participant_id, cursor);
        return { peerCursors: next };
      });
      break;
    }

    case 's2c_select_layer':
      // Could be used for highlighting peer selections
      break;

    case 's2c_select_token':
      // Could be used for syncing token stepper
      break;

    case 's2c_session_closed':
      get().leaveSession();
      break;

    case 's2c_error':
      set({ error: msg.message });
      break;

    case 's2c_pong':
      break;
  }
}

function _handleDisconnect(set: Setter) {
  set({
    isInSession: false,
    socket: null,
    error: 'Connection lost',
  });
  if (_scanUnsub) {
    _scanUnsub();
    _scanUnsub = null;
  }
}

function _setupHostSubscription(get: Getter) {
  if (_scanUnsub) _scanUnsub();

  let prevIsScanning = useScanStore.getState().isScanning;
  let prevMode = useScanStore.getState().mode;
  let prevTokenIdx = useScanStore.getState().selectedTokenIdx;

  _scanUnsub = useScanStore.subscribe((state) => {
    const collabState = get();
    if (collabState.role !== 'host' || !collabState.socket?.connected) return;

    // Broadcast when scan completes
    if (prevIsScanning && !state.isScanning) {
      collabState.broadcastScanState();
    }

    // Broadcast when scan starts
    if (!prevIsScanning && state.isScanning) {
      collabState.broadcastScanState();
    }

    // Broadcast mode change
    if (prevMode !== state.mode) {
      collabState.broadcastScanState();
    }

    // Broadcast token selection change
    if (prevTokenIdx !== state.selectedTokenIdx) {
      collabState.socket.send({
        type: 'c2s_select_token',
        token_idx: state.selectedTokenIdx,
      });
    }

    prevIsScanning = state.isScanning;
    prevMode = state.mode;
    prevTokenIdx = state.selectedTokenIdx;
  });
}
