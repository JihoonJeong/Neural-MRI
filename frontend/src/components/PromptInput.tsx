import { useState, useEffect } from 'react';
import { useScanStore } from '../store/useScanStore';
import { useCompareStore } from '../store/useCompareStore';
import { useLocaleStore } from '../store/useLocaleStore';
import { useCollabStore } from '../store/useCollabStore';
import type { TranslationKey } from '../i18n/translations';

const inputStyle = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid var(--border)',
  color: 'var(--text-data)',
  padding: '6px 10px',
  fontSize: 'var(--font-size-md)',
  fontFamily: 'var(--font-primary)',
  outline: 'none',
} as const;

export function PromptInput() {
  const { prompt, setPrompt, isScanning, runScan, mode } = useScanStore();
  const { isCompareMode, promptB, setPromptB, runCompare, isScanningB, toggleCompare } = useCompareStore();
  const t = useLocaleStore((s) => s.t);
  const [scanFailed, setScanFailed] = useState(false);

  const isViewer = useCollabStore((s) => s.role) === 'viewer';
  const isPromptMode = mode === 'fMRI' || mode === 'DTI' || mode === 'FLAIR';
  const busy = isScanning || isScanningB;

  // Watch for scan failure: transition from scanning to not-scanning with error
  const logs = useScanStore((s) => s.logs);
  useEffect(() => {
    if (!isScanning && logs.length > 0) {
      const lastLog = logs[logs.length - 1];
      if (lastLog.msg.startsWith('Scan failed') || lastLog.msg.startsWith('Compare failed')) {
        setScanFailed(true);
        const timer = setTimeout(() => setScanFailed(false), 1000);
        return () => clearTimeout(timer);
      }
    }
  }, [isScanning, logs]);

  const handleScan = () => {
    if (isCompareMode) {
      runCompare();
    } else {
      runScan();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !busy) {
      handleScan();
    }
  };

  const buttonLabel = (() => {
    if (busy) return isCompareMode ? 'COMPARING...' : 'SCANNING...';
    if (isCompareMode) return t('compare.scan' as TranslationKey);
    return 'SCAN';
  })();

  return (
    <div style={{ borderTop: '1px solid var(--border)' }}>
      {/* Row A */}
      <div className="flex items-center gap-2 px-3 py-2">
        {/* Compare toggle */}
        {isPromptMode && (
          <button
            onClick={toggleCompare}
            title={t('compare.title' as TranslationKey)}
            style={{
              background: isCompareMode ? 'rgba(0,255,170,0.12)' : 'none',
              border: isCompareMode ? '1px solid rgba(0,255,170,0.3)' : '1px solid var(--border)',
              color: isCompareMode ? 'var(--accent-active)' : 'var(--text-secondary)',
              padding: '4px 8px',
              fontSize: 'var(--font-size-sm)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
              borderRadius: 4,
            }}
          >
            ⇄
          </button>
        )}

        <span
          className="whitespace-nowrap"
          style={{
            fontSize: 'var(--font-size-xs)',
            color: isCompareMode ? 'var(--accent-active)' : 'var(--text-primary)',
          }}
        >
          {isCompareMode ? t('compare.promptA' as TranslationKey) + ':' : 'PROMPT:'}
        </span>
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isViewer}
          className="flex-1 rounded"
          style={{ ...inputStyle, opacity: isViewer ? 0.5 : 1 }}
        />
        {!isCompareMode && (
          <button
            onClick={handleScan}
            disabled={busy || isViewer}
            className="rounded tracking-wide"
            style={{
              background: scanFailed
                ? 'rgba(255,68,102,0.2)'
                : busy || isViewer
                  ? '#1a1c22'
                  : 'rgba(0,255,170,0.12)',
              border: scanFailed
                ? '1px solid rgba(255,68,102,0.5)'
                : '1px solid rgba(0,255,170,0.3)',
              color: scanFailed ? '#ff4466' : 'var(--accent-active)',
              padding: '6px 16px',
              fontSize: 'var(--font-size-sm)',
              fontFamily: 'var(--font-primary)',
              cursor: busy || isViewer ? 'default' : 'pointer',
              letterSpacing: '1px',
              animation: busy ? 'scan-pulse 1.5s ease-in-out infinite' : 'none',
              opacity: isViewer ? 0.5 : 1,
            }}
          >
            {isViewer ? 'VIEW ONLY' : busy ? 'SCANNING...' : 'SCAN'}
          </button>
        )}
      </div>

      {/* Row B (compare mode only) */}
      {isCompareMode && (
        <div className="flex items-center gap-2 px-3 pb-2">
          <div style={{ width: isPromptMode ? 36 : 0 }} />
          <span
            className="whitespace-nowrap"
            style={{ fontSize: 'var(--font-size-xs)', color: '#ff9494' }}
          >
            {t('compare.promptB' as TranslationKey)}:
          </span>
          <input
            value={promptB}
            onChange={(e) => setPromptB(e.target.value)}
            onKeyDown={handleKeyDown}
            className="flex-1 rounded"
            style={inputStyle}
          />
          <button
            onClick={handleScan}
            disabled={busy}
            className="rounded tracking-wide"
            style={{
              background: scanFailed
                ? 'rgba(255,68,102,0.2)'
                : busy
                  ? '#1a1c22'
                  : 'rgba(0,255,170,0.12)',
              border: scanFailed
                ? '1px solid rgba(255,68,102,0.5)'
                : '1px solid rgba(0,255,170,0.3)',
              color: scanFailed ? '#ff4466' : 'var(--accent-active)',
              padding: '6px 16px',
              fontSize: 'var(--font-size-sm)',
              fontFamily: 'var(--font-primary)',
              cursor: busy ? 'default' : 'pointer',
              letterSpacing: '1px',
              animation: busy ? 'scan-pulse 1.5s ease-in-out infinite' : 'none',
            }}
          >
            {buttonLabel}
          </button>
        </div>
      )}
    </div>
  );
}
