import { useState, useEffect } from 'react';
import { useScanStore } from '../store/useScanStore';

export function PromptInput() {
  const { prompt, setPrompt, isScanning, runScan } = useScanStore();
  const [scanFailed, setScanFailed] = useState(false);

  // Watch for scan failure: transition from scanning to not-scanning with error
  const logs = useScanStore((s) => s.logs);
  useEffect(() => {
    if (!isScanning && logs.length > 0) {
      const lastLog = logs[logs.length - 1];
      if (lastLog.msg.startsWith('Scan failed')) {
        setScanFailed(true);
        const timer = setTimeout(() => setScanFailed(false), 1000);
        return () => clearTimeout(timer);
      }
    }
  }, [isScanning, logs]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isScanning) {
      runScan();
    }
  };

  return (
    <div
      className="flex items-center gap-2 px-3 py-2"
      style={{ borderTop: '1px solid var(--border)' }}
    >
      <span
        className="whitespace-nowrap"
        style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-primary)' }}
      >
        PROMPT:
      </span>
      <input
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={handleKeyDown}
        className="flex-1 rounded"
        style={{
          background: 'rgba(255,255,255,0.03)',
          border: '1px solid var(--border)',
          color: 'var(--text-data)',
          padding: '6px 10px',
          fontSize: 'var(--font-size-md)',
          fontFamily: 'var(--font-primary)',
          outline: 'none',
        }}
      />
      <button
        onClick={runScan}
        disabled={isScanning}
        className="rounded tracking-wide"
        style={{
          background: scanFailed
            ? 'rgba(255,68,102,0.2)'
            : isScanning
              ? '#1a1c22'
              : 'rgba(0,255,170,0.12)',
          border: scanFailed
            ? '1px solid rgba(255,68,102,0.5)'
            : '1px solid rgba(0,255,170,0.3)',
          color: scanFailed ? '#ff4466' : 'var(--accent-active)',
          padding: '6px 16px',
          fontSize: 'var(--font-size-sm)',
          fontFamily: 'var(--font-primary)',
          cursor: isScanning ? 'default' : 'pointer',
          letterSpacing: '1px',
          animation: isScanning ? 'scan-pulse 1.5s ease-in-out infinite' : 'none',
        }}
      >
        {isScanning ? 'SCANNING...' : 'SCAN'}
      </button>
    </div>
  );
}
