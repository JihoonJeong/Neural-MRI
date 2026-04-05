import { useState, useEffect } from 'react';
import { useScanStore } from '../store/useScanStore';
import { useCompareStore } from '../store/useCompareStore';
import { useLocaleStore } from '../store/useLocaleStore';
import { useCollabStore } from '../store/useCollabStore';
import { useStreamStore } from '../store/useStreamStore';
import { TemplateSelector } from './TemplateSelector';
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

  const { isStreamPreferred, setStreamPreferred, startStream, cancelStream, status: streamStatus } =
    useStreamStore();

  const isViewer = useCollabStore((s) => s.role) === 'viewer';
  const isPromptMode = mode === 'fMRI' || mode === 'DTI' || mode === 'FLAIR';
  const isStreamableMode = mode === 'fMRI' || mode === 'DTI';
  const isStreaming = streamStatus === 'streaming' || streamStatus === 'connecting';
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
    if (isStreaming) {
      cancelStream();
      return;
    }
    if (isCompareMode) {
      runCompare();
    } else if (isStreamPreferred && isStreamableMode) {
      startStream(mode as 'fMRI' | 'DTI', prompt);
    } else {
      runScan();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !busy) {
      e.preventDefault();
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
        {/* Template selector */}
        <TemplateSelector />

        {/* LIVE toggle (fMRI/DTI only) */}
        {isStreamableMode && (
          <button
            onClick={() => setStreamPreferred(!isStreamPreferred)}
            title={t('stream.live' as TranslationKey)}
            style={{
              background: isStreamPreferred ? 'rgba(0,170,255,0.12)' : 'none',
              border: isStreamPreferred
                ? '1px solid rgba(0,170,255,0.4)'
                : '1px solid var(--border)',
              color: isStreamPreferred ? '#00aaff' : 'var(--text-secondary)',
              padding: '4px 8px',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
              borderRadius: 4,
              letterSpacing: '1px',
            }}
          >
            {t('stream.live' as TranslationKey)}
          </button>
        )}

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
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isViewer}
          rows={2}
          className="flex-1 rounded"
          style={{
            ...inputStyle,
            opacity: isViewer ? 0.5 : 1,
            resize: 'vertical',
            minHeight: 36,
            maxHeight: 120,
            lineHeight: 1.4,
          }}
        />
        {!isCompareMode && (
          <button
            onClick={handleScan}
            disabled={(busy && !isStreaming) || isViewer}
            className="rounded tracking-wide"
            style={{
              background: scanFailed
                ? 'rgba(255,68,102,0.2)'
                : isStreaming
                  ? 'rgba(0,170,255,0.12)'
                  : busy || isViewer
                    ? '#1a1c22'
                    : 'rgba(0,255,170,0.12)',
              border: scanFailed
                ? '1px solid rgba(255,68,102,0.5)'
                : isStreaming
                  ? '1px solid rgba(0,170,255,0.4)'
                  : '1px solid rgba(0,255,170,0.3)',
              color: scanFailed ? '#ff4466' : isStreaming ? '#00aaff' : 'var(--accent-active)',
              padding: '6px 16px',
              fontSize: 'var(--font-size-sm)',
              fontFamily: 'var(--font-primary)',
              cursor: (busy && !isStreaming) || isViewer ? 'default' : 'pointer',
              letterSpacing: '1px',
              animation: busy || isStreaming ? 'scan-pulse 1.5s ease-in-out infinite' : 'none',
              opacity: isViewer ? 0.5 : 1,
            }}
          >
            {isViewer
              ? 'VIEW ONLY'
              : isStreaming
                ? t('stream.streaming' as TranslationKey)
                : busy
                  ? 'SCANNING...'
                  : 'SCAN'}
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
          <textarea
            value={promptB}
            onChange={(e) => setPromptB(e.target.value)}
            onKeyDown={handleKeyDown}
            rows={2}
            className="flex-1 rounded"
            style={{ ...inputStyle, resize: 'vertical', minHeight: 36, maxHeight: 120, lineHeight: 1.4 }}
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
