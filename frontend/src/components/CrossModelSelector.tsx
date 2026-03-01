import { useEffect, useState } from 'react';
import { useCrossModelStore } from '../store/useCrossModelStore';
import { useModelStore } from '../store/useModelStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';
import type { ModelListEntry } from '../api/client';

const PHASE_LABELS: Record<string, string> = {
  scanning_a: 'Scanning model A...',
  switching: 'Switching model...',
  scanning_b: 'Scanning model B...',
};

export function CrossModelSelector() {
  const { modelIdB, setModelIdB, runCrossCompare, phase, error } = useCrossModelStore();
  const modelInfo = useModelStore((s) => s.modelInfo);
  const availableModels = useModelStore((s) => s.availableModels);
  const t = useLocaleStore((s) => s.t);
  const [models, setModels] = useState<ModelListEntry[]>([]);

  useEffect(() => {
    setModels(availableModels.filter((m) => m.model_id !== modelInfo?.model_id));
  }, [availableModels, modelInfo?.model_id]);

  const isBusy = phase !== 'idle' && phase !== 'done';
  const currentModelName = modelInfo?.model_id ?? 'Model A';

  return (
    <div
      className="flex items-center gap-2 px-3 py-2"
      style={{
        borderBottom: '1px solid var(--border)',
        fontSize: 'var(--font-size-xs)',
        color: 'var(--text-primary)',
      }}
    >
      <span style={{ color: 'var(--accent-active)', letterSpacing: '1px' }}>
        {t('crossModel.title' as TranslationKey)}
      </span>

      <span style={{ color: 'var(--text-secondary)' }}>
        A: {currentModelName}
      </span>

      <span style={{ color: 'var(--text-secondary)' }}>vs</span>

      <select
        value={modelIdB ?? ''}
        onChange={(e) => setModelIdB(e.target.value)}
        disabled={isBusy}
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          color: 'var(--text-primary)',
          padding: '2px 6px',
          fontSize: 'var(--font-size-xs)',
          fontFamily: 'var(--font-primary)',
        }}
      >
        <option value="">{t('crossModel.selectModel' as TranslationKey)}</option>
        {models.map((m) => (
          <option key={m.model_id} value={m.model_id}>
            {m.display_name} ({m.params})
          </option>
        ))}
      </select>

      <button
        onClick={runCrossCompare}
        disabled={isBusy || !modelIdB}
        style={{
          background: isBusy ? 'rgba(255,255,255,0.04)' : 'rgba(0,255,170,0.1)',
          border: '1px solid rgba(0,255,170,0.2)',
          color: isBusy ? 'var(--text-secondary)' : 'var(--accent-active)',
          padding: '2px 8px',
          fontSize: 'var(--font-size-xs)',
          fontFamily: 'var(--font-primary)',
          cursor: isBusy || !modelIdB ? 'not-allowed' : 'pointer',
          letterSpacing: '1px',
        }}
      >
        {isBusy ? PHASE_LABELS[phase] ?? phase : t('crossModel.compare' as TranslationKey)}
      </button>

      {error && (
        <span style={{ color: '#ff4466', fontSize: 'var(--font-size-xs)' }}>
          {error.length > 30 ? error.slice(0, 30) + '...' : error}
        </span>
      )}
    </div>
  );
}
