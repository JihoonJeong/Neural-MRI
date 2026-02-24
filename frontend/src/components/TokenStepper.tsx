import { useEffect, useCallback } from 'react';
import { useScanStore } from '../store/useScanStore';

export function TokenStepper() {
  const { mode, activationData, circuitData, anomalyData, selectedTokenIdx, setSelectedTokenIdx, stepToken } =
    useScanStore();

  // Show for fMRI/DTI/FLAIR modes
  const visible = mode === 'fMRI' || mode === 'DTI' || mode === 'FLAIR';
  const tokens =
    mode === 'fMRI' ? activationData?.tokens :
    mode === 'DTI' ? circuitData?.tokens :
    anomalyData?.tokens;

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (!visible || !tokens) return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        stepToken(-1);
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        stepToken(1);
      }
    },
    [visible, tokens, stepToken],
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
        disabled={selectedTokenIdx <= 0}
        className="shrink-0 rounded"
        style={{
          width: 22,
          height: 22,
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid var(--border)',
          color: selectedTokenIdx > 0 ? 'var(--text-primary)' : 'var(--text-secondary)',
          fontSize: 'var(--font-size-xs)',
          cursor: selectedTokenIdx > 0 ? 'pointer' : 'default',
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
          return (
            <button
              key={idx}
              onClick={() => setSelectedTokenIdx(idx)}
              className="shrink-0 rounded"
              style={{
                padding: '2px 6px',
                fontSize: 'var(--font-size-xs)',
                fontFamily: 'var(--font-primary)',
                background: isSelected ? 'rgba(0,255,170,0.15)' : 'rgba(255,255,255,0.04)',
                border: isSelected
                  ? '1px solid rgba(0,255,170,0.5)'
                  : '1px solid var(--border)',
                color: isSelected ? 'var(--accent-active)' : 'var(--text-data)',
                cursor: 'pointer',
                whiteSpace: 'nowrap',
              }}
            >
              {displayToken}
            </button>
          );
        })}
      </div>
      <button
        onClick={() => stepToken(1)}
        disabled={selectedTokenIdx >= tokens.length - 1}
        className="shrink-0 rounded"
        style={{
          width: 22,
          height: 22,
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid var(--border)',
          color: selectedTokenIdx < tokens.length - 1 ? 'var(--text-primary)' : 'var(--text-secondary)',
          fontSize: 'var(--font-size-xs)',
          cursor: selectedTokenIdx < tokens.length - 1 ? 'pointer' : 'default',
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
