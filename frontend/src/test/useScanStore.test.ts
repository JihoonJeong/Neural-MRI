import { describe, it, expect, beforeEach } from 'vitest';
import { useScanStore } from '../store/useScanStore';

describe('useScanStore', () => {
  beforeEach(() => {
    useScanStore.setState({
      mode: 'T1',
      layoutMode: 'vertical',
      structuralData: null,
      weightData: null,
      activationData: null,
      circuitData: null,
      anomalyData: null,
      isScanning: false,
      selectedLayerId: null,
      selectedTokenIdx: 0,
      tokenCount: 0,
      prompt: 'The capital of France is',
      logs: [{ time: '00:00', msg: 'Neural MRI Scanner initialized' }],
    });
  });

  it('has correct initial state', () => {
    const state = useScanStore.getState();
    expect(state.mode).toBe('T1');
    expect(state.layoutMode).toBe('vertical');
    expect(state.prompt).toBe('The capital of France is');
    expect(state.isScanning).toBe(false);
    expect(state.selectedTokenIdx).toBe(0);
  });

  it('setMode updates mode and adds log', () => {
    useScanStore.getState().setMode('fMRI');
    const state = useScanStore.getState();
    expect(state.mode).toBe('fMRI');
    expect(state.logs.some((l) => l.msg.includes('fMRI'))).toBe(true);
  });

  it('setLayoutMode updates layout mode', () => {
    useScanStore.getState().setLayoutMode('brain');
    expect(useScanStore.getState().layoutMode).toBe('brain');
  });

  it('setPrompt updates prompt', () => {
    useScanStore.getState().setPrompt('Hello world');
    expect(useScanStore.getState().prompt).toBe('Hello world');
  });

  it('selectLayer sets selectedLayerId', () => {
    useScanStore.getState().selectLayer('blocks.3.attn');
    expect(useScanStore.getState().selectedLayerId).toBe('blocks.3.attn');

    useScanStore.getState().selectLayer(null);
    expect(useScanStore.getState().selectedLayerId).toBeNull();
  });

  it('stepToken respects bounds', () => {
    useScanStore.setState({ tokenCount: 5, selectedTokenIdx: 2 });

    useScanStore.getState().stepToken(1);
    expect(useScanStore.getState().selectedTokenIdx).toBe(3);

    useScanStore.getState().stepToken(-2);
    expect(useScanStore.getState().selectedTokenIdx).toBe(1);

    // Can't go below 0
    useScanStore.getState().stepToken(-10);
    expect(useScanStore.getState().selectedTokenIdx).toBe(1);

    // Can't exceed tokenCount
    useScanStore.setState({ selectedTokenIdx: 4 });
    useScanStore.getState().stepToken(1);
    expect(useScanStore.getState().selectedTokenIdx).toBe(4);
  });

  it('setSelectedTokenIdx respects bounds', () => {
    useScanStore.setState({ tokenCount: 5 });

    useScanStore.getState().setSelectedTokenIdx(3);
    expect(useScanStore.getState().selectedTokenIdx).toBe(3);

    // Out of bounds — no change
    useScanStore.getState().setSelectedTokenIdx(-1);
    expect(useScanStore.getState().selectedTokenIdx).toBe(3);

    useScanStore.getState().setSelectedTokenIdx(5);
    expect(useScanStore.getState().selectedTokenIdx).toBe(3);
  });

  it('clearScanData resets all scan data', () => {
    useScanStore.setState({
      structuralData: { layers: [] } as never,
      weightData: { layers: [] } as never,
      activationData: { tokens: [], layers: [] } as never,
      selectedLayerId: 'blocks.0.attn',
      selectedTokenIdx: 3,
      tokenCount: 5,
    });

    useScanStore.getState().clearScanData();
    const state = useScanStore.getState();
    expect(state.structuralData).toBeNull();
    expect(state.weightData).toBeNull();
    expect(state.activationData).toBeNull();
    expect(state.circuitData).toBeNull();
    expect(state.anomalyData).toBeNull();
    expect(state.selectedLayerId).toBeNull();
    expect(state.selectedTokenIdx).toBe(0);
    expect(state.tokenCount).toBe(0);
  });

  it('addLog keeps max 51 entries', () => {
    for (let i = 0; i < 60; i++) {
      useScanStore.getState().addLog(`msg ${i}`);
    }
    // Initial log + 60 added, but capped at 51 (slice -50 + new)
    expect(useScanStore.getState().logs.length).toBeLessThanOrEqual(51);
  });
});
