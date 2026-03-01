import { useEffect, useRef, useState } from 'react';
import { useModelStore } from '../store/useModelStore';
import { useScanStore } from '../store/useScanStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';
import { useReportStore } from '../store/useReportStore';
import { useRecordingStore } from '../store/useRecordingStore';
import { useCrossModelStore } from '../store/useCrossModelStore';
import { useSettingsStore } from '../store/useSettingsStore';
import { exportPng, exportSvg, exportJson, exportReport } from '../utils/exportUtils';
import { exportGif, exportWebM } from '../utils/videoExport';
import { ModelPicker } from './ModelPicker';

export function TopBar() {
  const { modelInfo, isLoading, error } = useModelStore();
  const scanStore = useScanStore();
  const addLog = scanStore.addLog;
  const { locale, toggleLocale, openGuide, t } = useLocaleStore();
  const { report, generateReport, isGenerating } = useReportStore();
  const recording = useRecordingStore((s) => s.recording);
  const { isCrossModelMode, toggleCrossModel } = useCrossModelStore();
  const [showError, setShowError] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [videoExporting, setVideoExporting] = useState<string | null>(null);
  const exportRef = useRef<HTMLDivElement>(null);

  const hasScanData = !!(scanStore.structuralData || scanStore.activationData || scanStore.circuitData || scanStore.anomalyData);

  // Show error message briefly then auto-dismiss
  useEffect(() => {
    if (error) {
      setShowError(true);
      const timer = setTimeout(() => setShowError(false), 3000);
      return () => clearTimeout(timer);
    }
    setShowError(false);
  }, [error]);

  // Close dropdown on outside click
  useEffect(() => {
    if (!exportOpen) return;
    const handler = (e: MouseEvent) => {
      if (exportRef.current && !exportRef.current.contains(e.target as Node)) {
        setExportOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [exportOpen]);

  const handleExport = (type: 'png' | 'svg' | 'json' | 'report' | 'gif' | 'webm') => {
    setExportOpen(false);
    const ts = new Date().toISOString().slice(0, 10);
    const modelId = modelInfo?.model_id ?? 'scan';

    if (type === 'png') {
      exportPng(`nmri-${modelId}-${ts}.png`);
      addLog('Exported PNG');
    } else if (type === 'svg') {
      exportSvg(`nmri-${modelId}-${ts}.svg`);
      addLog('Exported SVG');
    } else if (type === 'json') {
      const data = {
        model_id: modelInfo?.model_id,
        mode: scanStore.mode,
        prompt: scanStore.prompt,
        structuralData: scanStore.structuralData,
        weightData: scanStore.weightData,
        activationData: scanStore.activationData,
        circuitData: scanStore.circuitData,
        anomalyData: scanStore.anomalyData,
        exportedAt: new Date().toISOString(),
      };
      exportJson(data, `nmri-${modelId}-${ts}.json`);
      addLog('Exported JSON');
    } else if (type === 'report' && report) {
      exportReport(report, `nmri-report-${modelId}-${ts}.md`);
      addLog('Exported Report');
    } else if (type === 'gif' && recording) {
      setVideoExporting('GIF');
      addLog('Exporting GIF...');
      exportGif(recording, {
        fps: 5,
        onProgress: (p) => { if (p >= 1) setVideoExporting(null); },
      }).then(() => {
        addLog('GIF exported');
        setVideoExporting(null);
      }).catch(() => {
        addLog('GIF export failed');
        setVideoExporting(null);
      });
    } else if (type === 'webm' && recording) {
      setVideoExporting('WebM');
      addLog('Exporting WebM...');
      exportWebM(recording, {
        fps: 10,
        onProgress: (p) => { if (p >= 1) setVideoExporting(null); },
      }).then(() => {
        addLog('WebM exported');
        setVideoExporting(null);
      }).catch(() => {
        addLog('WebM export failed');
        setVideoExporting(null);
      });
    }
  };

  const paramStr = modelInfo
    ? `${(modelInfo.n_params / 1e6).toFixed(0)}M params`
    : '';

  const menuItemStyle = (disabled = false) => ({
    display: 'block' as const,
    width: '100%',
    textAlign: 'left' as const,
    background: 'none',
    border: 'none',
    color: disabled ? 'var(--text-secondary)' : 'var(--text-data)',
    padding: '6px 12px',
    fontSize: 'var(--font-size-xs)',
    fontFamily: 'var(--font-primary)',
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.4 : 1,
  });

  return (
    <div
      className="flex items-center gap-4 px-5 py-3 border-b"
      style={{
        background: 'linear-gradient(180deg, #0f1218 0%, #0a0c10 100%)',
        borderColor: 'var(--border)',
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2">
        <div
          className="w-2.5 h-2.5 rounded-full"
          style={{
            background: '#00ffaa',
            boxShadow: '0 0 8px rgba(0,255,170,0.5)',
          }}
        />
        <span
          className="font-bold tracking-widest"
          style={{ fontSize: 'var(--font-size-lg)', color: 'var(--text-primary)' }}
        >
          NEURAL MRI
        </span>
      </div>

      <span
        className="tracking-wide"
        style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}
      >
        Model Resonance Imaging
      </span>

      {/* Right side */}
      <div className="ml-auto flex items-center gap-3">
        {paramStr && (
          <span style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-secondary)' }}>
            {paramStr}
          </span>
        )}
        <ModelPicker />
        {isLoading && (
          <span
            className="loading-dots"
            style={{ fontSize: 'var(--font-size-xs)', color: 'var(--accent-active)' }}
          >
            Loading
            <span className="dot dot1">.</span>
            <span className="dot dot2">.</span>
            <span className="dot dot3">.</span>
          </span>
        )}
        {showError && error && (
          <span
            style={{
              fontSize: 'var(--font-size-xs)',
              color: '#ff4466',
              animation: 'nmri-fade-in 0.2s ease-out',
            }}
          >
            {error.length > 40 ? error.slice(0, 40) + '...' : error}
          </span>
        )}
        {modelInfo && !isLoading && (
          <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
            {modelInfo.device.toUpperCase()}
          </span>
        )}
        <button
          onClick={toggleCrossModel}
          className="rounded"
          style={{
            background: isCrossModelMode ? 'rgba(0,255,170,0.2)' : 'rgba(255,255,255,0.04)',
            border: `1px solid ${isCrossModelMode ? 'rgba(0,255,170,0.4)' : 'var(--border)'}`,
            color: isCrossModelMode ? 'var(--accent-active)' : 'var(--text-secondary)',
            padding: '2px 8px',
            fontSize: 'var(--font-size-xs)',
            fontFamily: 'var(--font-primary)',
            cursor: 'pointer',
            letterSpacing: '1px',
          }}
        >
          {t('crossModel.title' as TranslationKey)}
        </button>
        <button
          onClick={generateReport}
          disabled={isGenerating || isLoading || !modelInfo}
          className="rounded"
          style={{
            background: isGenerating ? 'rgba(255,255,255,0.04)' : 'rgba(0,255,170,0.08)',
            border: '1px solid rgba(0,255,170,0.2)',
            color: isGenerating ? 'var(--text-secondary)' : 'var(--accent-active)',
            padding: '2px 8px',
            fontSize: 'var(--font-size-xs)',
            fontFamily: 'var(--font-primary)',
            cursor: isGenerating || isLoading || !modelInfo ? 'not-allowed' : 'pointer',
            letterSpacing: '1px',
          }}
        >
          {isGenerating ? t('report.generating' as TranslationKey) : t('report.button' as TranslationKey)}
        </button>

        {/* Export dropdown */}
        <div ref={exportRef} style={{ position: 'relative' }}>
          <button
            onClick={() => setExportOpen(!exportOpen)}
            className="rounded"
            style={{
              background: exportOpen ? 'rgba(0,255,170,0.15)' : 'rgba(0,255,170,0.08)',
              border: '1px solid rgba(0,255,170,0.2)',
              color: 'var(--accent-active)',
              padding: '2px 8px',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
              letterSpacing: '1px',
            }}
          >
            {t('export.button' as TranslationKey)} ▾
          </button>
          {exportOpen && (
            <div
              style={{
                position: 'absolute',
                top: '100%',
                right: 0,
                marginTop: 4,
                background: 'var(--bg-surface)',
                border: '1px solid var(--border)',
                borderRadius: 4,
                minWidth: 160,
                zIndex: 50,
                boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
                animation: 'nmri-fade-in 0.15s ease-out',
              }}
            >
              <button
                onClick={() => handleExport('png')}
                disabled={!hasScanData}
                style={menuItemStyle(!hasScanData)}
                onMouseEnter={(e) => { if (hasScanData) (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'none'; }}
              >
                {t('export.png' as TranslationKey)}
              </button>
              <button
                onClick={() => handleExport('svg')}
                disabled={!hasScanData}
                style={menuItemStyle(!hasScanData)}
                onMouseEnter={(e) => { if (hasScanData) (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'none'; }}
              >
                {t('export.svg' as TranslationKey)}
              </button>
              <div style={{ borderTop: '1px solid var(--border)', margin: '2px 0' }} />
              <button
                onClick={() => handleExport('json')}
                disabled={!hasScanData}
                style={menuItemStyle(!hasScanData)}
                onMouseEnter={(e) => { if (hasScanData) (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'none'; }}
              >
                {t('export.json' as TranslationKey)}
              </button>
              {report && (
                <button
                  onClick={() => handleExport('report')}
                  style={menuItemStyle()}
                  onMouseEnter={(e) => { (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                  onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'none'; }}
                >
                  {t('export.report' as TranslationKey)}
                </button>
              )}
              {recording && (
                <>
                  <div style={{ borderTop: '1px solid var(--border)', margin: '2px 0' }} />
                  <button
                    onClick={() => handleExport('gif')}
                    disabled={!!videoExporting}
                    style={menuItemStyle(!!videoExporting)}
                    onMouseEnter={(e) => { if (!videoExporting) (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                    onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'none'; }}
                  >
                    {videoExporting === 'GIF' ? 'Exporting GIF...' : t('export.gif' as TranslationKey)}
                  </button>
                  <button
                    onClick={() => handleExport('webm')}
                    disabled={!!videoExporting}
                    style={menuItemStyle(!!videoExporting)}
                    onMouseEnter={(e) => { if (!videoExporting) (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                    onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'none'; }}
                  >
                    {videoExporting === 'WebM' ? 'Exporting WebM...' : t('export.webm' as TranslationKey)}
                  </button>
                </>
              )}
            </div>
          )}
        </div>

        <button
          onClick={() => useSettingsStore.getState().openSettings()}
          className="rounded"
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid var(--border)',
            color: 'var(--text-secondary)',
            padding: '2px 6px',
            fontSize: 'var(--font-size-xs)',
            fontFamily: 'var(--font-primary)',
            cursor: 'pointer',
            letterSpacing: '1px',
          }}
        >
          {t('settings.button' as TranslationKey)}
        </button>
        <button
          onClick={toggleLocale}
          className="rounded"
          style={{
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid var(--border)',
            color: 'var(--text-secondary)',
            padding: '2px 6px',
            fontSize: 'var(--font-size-xs)',
            fontFamily: 'var(--font-primary)',
            cursor: 'pointer',
            letterSpacing: '1px',
          }}
        >
          {locale === 'en' ? 'KO' : 'EN'}
        </button>
        <button
          onClick={openGuide}
          className="rounded"
          style={{
            background: 'rgba(0,255,170,0.08)',
            border: '1px solid rgba(0,255,170,0.2)',
            color: 'var(--accent-active)',
            padding: '2px 8px',
            fontSize: 'var(--font-size-sm)',
            fontFamily: 'var(--font-primary)',
            cursor: 'pointer',
          }}
        >
          ?
        </button>
      </div>
    </div>
  );
}
