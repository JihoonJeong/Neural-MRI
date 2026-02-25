import { useEffect, useState } from 'react';
import { useModelStore } from '../store/useModelStore';
import { useScanStore } from '../store/useScanStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';
import { useReportStore } from '../store/useReportStore';

export function TopBar() {
  const { modelInfo, isLoading, error, availableModels, loadModel } = useModelStore();
  const addLog = useScanStore((s) => s.addLog);
  const { locale, toggleLocale, openGuide, t } = useLocaleStore();
  const { generateReport, isGenerating } = useReportStore();
  const [showError, setShowError] = useState(false);

  // Show error message briefly then auto-dismiss
  useEffect(() => {
    if (error) {
      setShowError(true);
      const timer = setTimeout(() => setShowError(false), 3000);
      return () => clearTimeout(timer);
    }
    setShowError(false);
  }, [error]);

  const handleModelChange = async (e: React.ChangeEvent<HTMLSelectElement>) => {
    const modelId = e.target.value;
    addLog(`Loading model: ${modelId}...`);
    await loadModel(modelId);
    const info = useModelStore.getState().modelInfo;
    if (info) {
      addLog(`Model loaded: ${modelId}`);
    }
  };

  const paramStr = modelInfo
    ? `${(modelInfo.n_params / 1e6).toFixed(0)}M params`
    : '';

  // Use dynamic model list, fallback to basic list if API hasn't responded yet
  const models = availableModels.length > 0
    ? availableModels
    : [{ model_id: 'gpt2', display_name: 'GPT-2 Small', params: '124M', family: 'gpt2', tl_compat: true, gated: false, is_loaded: false }];

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
        <select
          value={modelInfo?.model_id ?? 'gpt2'}
          onChange={handleModelChange}
          disabled={isLoading}
          className="rounded"
          style={{
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
            padding: '4px 8px',
            fontSize: 'var(--font-size-sm)',
            fontFamily: 'var(--font-primary)',
          }}
        >
          {models.map((m) => (
            <option key={m.model_id} value={m.model_id}>
              {m.gated ? '\u{1F512} ' : ''}{m.display_name} ({m.params})
            </option>
          ))}
        </select>
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
