import { useEffect, useCallback } from 'react';
import { useScanStore } from '../store/useScanStore';
import type { LayoutMode } from '../store/useScanStore';
import { useCompareStore } from '../store/useCompareStore';
import { useCrossModelStore } from '../store/useCrossModelStore';
import { useRecordingStore } from '../store/useRecordingStore';
import { useStreamStore } from '../store/useStreamStore';
import type { ScanMode } from '../types/model';

const LAYOUT_ORDER: LayoutMode[] = ['vertical', 'brain', 'network', 'radial'];

export function TokenStepper() {
  const { mode, activationData, circuitData, anomalyData, selectedTokenIdx, setSelectedTokenIdx, stepToken } =
    useScanStore();
  const streamStatus = useStreamStore((s) => s.status);
  const streamCurrentToken = useStreamStore((s) => s.currentToken);

  const isStreaming = streamStatus === 'streaming' || streamStatus === 'connecting';

  // Show for fMRI/DTI/FLAIR modes
  const visible = mode === 'fMRI' || mode === 'DTI' || mode === 'FLAIR';
  const tokens =
    mode === 'fMRI' ? activationData?.tokens :
    mode === 'DTI' ? circuitData?.tokens :
    anomalyData?.tokens;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      // Skip when typing in input/textarea/select
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

      // Token stepping (disabled during streaming)
      if (visible && tokens && !isStreaming) {
        if (e.key === 'ArrowLeft') { e.preventDefault(); stepToken(-1); return; }
        if (e.key === 'ArrowRight') { e.preventDefault(); stepToken(1); return; }
      }

      // Mode switching: 1-5
      const modeMap: Record<string, ScanMode> = { '1': 'T1', '2': 'T2', '3': 'fMRI', '4': 'DTI', '5': 'FLAIR' };
      if (modeMap[e.key]) {
        e.preventDefault();
        useScanStore.getState().setMode(modeMap[e.key]);
        return;
      }

      // R: toggle recording
      if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        const rec = useRecordingStore.getState();
        if (rec.isRecording) rec.stopRecording();
        else rec.startRecording();
        return;
      }

      // Space: play/pause playback
      if (e.key === ' ') {
        const rec = useRecordingStore.getState();
        if (rec.recording) {
          e.preventDefault();
          if (rec.isPlaying) rec.pause();
          else rec.play();
        }
        return;
      }

      // L: cycle layout
      if (e.key === 'l' || e.key === 'L') {
        e.preventDefault();
        const s = useScanStore.getState();
        const idx = LAYOUT_ORDER.indexOf(s.layoutMode);
        s.setLayoutMode(LAYOUT_ORDER[(idx + 1) % LAYOUT_ORDER.length]);
        return;
      }

      // C: toggle compare
      if (e.key === 'c' || e.key === 'C') {
        e.preventDefault();
        if (e.shiftKey) {
          useCrossModelStore.getState().toggleCrossModel();
        } else {
          useCompareStore.getState().toggleCompare();
        }
        return;
      }
    },
    [visible, tokens, stepToken, isStreaming],
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  if (!visible || !tokens || tokens.length === 0) return null;

  return (
    <div
      className="flex items-center gap-1 px-3 py-2 overflow-x-auto"
      style={{ borderTop: '1px solid var(--border)' }}
    >
      <span
        className="shrink-0 mr-1"
        style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
      >
        TOKEN:
      </span>
      <button
        onClick={() => stepToken(-1)}
        disabled={isStreaming || selectedTokenIdx <= 0}
        className="shrink-0 rounded"
        style={{
          width: 22,
          height: 22,
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid var(--border)',
          color: !isStreaming && selectedTokenIdx > 0 ? 'var(--text-primary)' : 'var(--text-secondary)',
          fontSize: 'var(--font-size-xs)',
          cursor: !isStreaming && selectedTokenIdx > 0 ? 'pointer' : 'default',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        &lt;
      </button>
      <div className="flex gap-1 overflow-x-auto">
        {tokens.map((token, idx) => {
          const isSelected = idx === selectedTokenIdx;
          const displayToken = token.replace(/^▁/, ' ').replace(/^Ġ/, ' ');

          // Streaming awareness
          const arrived = !isStreaming || idx < streamCurrentToken;
          const isCurrent = isStreaming && idx === streamCurrentToken - 1;

          return (
            <button
              key={idx}
              onClick={() => !isStreaming && setSelectedTokenIdx(idx)}
              className="shrink-0 rounded"
              style={{
                padding: '2px 6px',
                fontSize: 'var(--font-size-xs)',
                fontFamily: 'var(--font-primary)',
                background: isCurrent
                  ? 'rgba(0,170,255,0.15)'
                  : isSelected && !isStreaming
                    ? 'rgba(0,255,170,0.15)'
                    : 'rgba(255,255,255,0.04)',
                border: isCurrent
                  ? '1px solid rgba(0,170,255,0.5)'
                  : isSelected && !isStreaming
                    ? '1px solid rgba(0,255,170,0.5)'
                    : '1px solid var(--border)',
                color: isCurrent
                  ? '#00aaff'
                  : isSelected && !isStreaming
                    ? 'var(--accent-active)'
                    : 'var(--text-data)',
                cursor: isStreaming ? 'default' : 'pointer',
                whiteSpace: 'nowrap',
                opacity: arrived ? 1 : 0.3,
                animation: isCurrent ? 'scan-pulse 1.5s ease-in-out infinite' : 'none',
                transition: 'opacity 0.2s ease',
              }}
            >
              {displayToken}
            </button>
          );
        })}
      </div>
      <button
        onClick={() => stepToken(1)}
        disabled={isStreaming || selectedTokenIdx >= tokens.length - 1}
        className="shrink-0 rounded"
        style={{
          width: 22,
          height: 22,
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid var(--border)',
          color: !isStreaming && selectedTokenIdx < tokens.length - 1 ? 'var(--text-primary)' : 'var(--text-secondary)',
          fontSize: 'var(--font-size-xs)',
          cursor: !isStreaming && selectedTokenIdx < tokens.length - 1 ? 'pointer' : 'default',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        &gt;
      </button>
      <span
        className="shrink-0 ml-1"
        style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
      >
        {selectedTokenIdx + 1}/{tokens.length}
      </span>
    </div>
  );
}
