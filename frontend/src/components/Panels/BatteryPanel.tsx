import { useBatteryStore } from '../../store/useBatteryStore';
import { useModelStore } from '../../store/useModelStore';
import { useSAEStore } from '../../store/useSAEStore';
import { useLocaleStore } from '../../store/useLocaleStore';
import type { TranslationKey } from '../../i18n/translations';

const CATEGORY_COLORS: Record<string, string> = {
  factual_recall: '#44ddaa',
  gender_bias: '#ffaa44',
  syntactic: '#aaccee',
  negation: '#ff6644',
  repetition: '#aa88ff',
  reasoning: '#44aaff',
};

export function BatteryPanel() {
  const { result, isRunning, error, runBattery, openDetail, includeSAE, saeLayer, setIncludeSAE, setSaeLayer } = useBatteryStore();
  const isModelLoaded = useModelStore((s) => s.modelInfo !== null);
  const saeInfo = useSAEStore((s) => s.saeInfo);
  const t = useLocaleStore((s) => s.t);
  const saeAvailable = saeInfo?.available ?? false;

  return (
    <div className="px-3 py-2">
      <div className="flex items-center justify-between mb-2">
        <span
          className="tracking-wide font-bold"
          style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
        >
          {t('battery.title' as TranslationKey)}
        </span>
        <button
          onClick={() => runBattery()}
          disabled={isRunning || !isModelLoaded}
          style={{
            background: isRunning ? 'rgba(255,255,255,0.04)' : 'rgba(0,255,170,0.1)',
            border: '1px solid rgba(0,255,170,0.25)',
            color: isRunning ? 'var(--text-secondary)' : 'var(--accent-active)',
            padding: '2px 8px',
            fontSize: 'var(--font-size-xs)',
            fontFamily: 'var(--font-primary)',
            cursor: isRunning || !isModelLoaded ? 'not-allowed' : 'pointer',
            borderRadius: 3,
          }}
        >
          {isRunning ? t('battery.running' as TranslationKey) : t('battery.run' as TranslationKey)}
        </button>
      </div>

      {/* SAE toggle */}
      {saeAvailable && (
        <div className="flex items-center gap-2 mb-2" style={{ fontSize: 'var(--font-size-xs)' }}>
          <label
            className="flex items-center gap-1"
            style={{ color: includeSAE ? '#aa88ff' : 'var(--text-secondary)', cursor: 'pointer' }}
          >
            <input
              type="checkbox"
              checked={includeSAE}
              onChange={(e) => setIncludeSAE(e.target.checked)}
              style={{ accentColor: '#aa88ff' }}
            />
            {t('battery.includeSae' as TranslationKey)}
          </label>
          {includeSAE && saeInfo && (
            <select
              value={saeLayer ?? ''}
              onChange={(e) => setSaeLayer(e.target.value ? Number(e.target.value) : null)}
              style={{
                background: 'var(--bg-primary)',
                color: 'var(--text-data)',
                border: '1px solid var(--border)',
                borderRadius: 3,
                fontSize: 'var(--font-size-xs)',
                fontFamily: 'var(--font-primary)',
                padding: '1px 4px',
                width: 52,
              }}
            >
              <option value="">{t('battery.autoLayer' as TranslationKey)}</option>
              {saeInfo.layers.map((l) => (
                <option key={l} value={l}>{l}</option>
              ))}
            </select>
          )}
        </div>
      )}

      {error && (
        <div style={{ fontSize: 'var(--font-size-xs)', color: '#ff4466', marginBottom: 4 }}>
          {error.length > 40 ? error.slice(0, 40) + '...' : error}
        </div>
      )}

      {result && (
        <>
          {/* Summary */}
          <div
            className="mb-2"
            style={{
              fontSize: 'var(--font-size-sm)',
              fontWeight: 'bold',
            }}
          >
            <span style={{ color: result.passed > 0 ? '#44ddaa' : 'var(--text-secondary)' }}>
              {t('battery.passed' as TranslationKey)}: {result.passed}
            </span>
            <span style={{ color: 'var(--text-secondary)' }}> / {result.total_tests}</span>
            {result.failed > 0 && (
              <span style={{ color: '#ff4466', marginLeft: 8 }}>
                {t('battery.failed' as TranslationKey)}: {result.failed}
              </span>
            )}
            {result.sae_summary && (
              <span style={{ color: '#aa88ff', marginLeft: 8, fontSize: 'var(--font-size-xs)', fontWeight: 'normal' }}>
                SAE L{result.sae_summary.layer_idx}
              </span>
            )}
          </div>

          {/* Test list */}
          <div style={{ maxHeight: 160, overflowY: 'auto' }}>
            {result.results.map((r) => (
              <div
                key={r.test_id}
                className="flex items-center gap-2 py-0.5"
                style={{
                  fontSize: 'var(--font-size-xs)',
                  cursor: 'pointer',
                }}
                onClick={openDetail}
              >
                <span
                  style={{
                    color: CATEGORY_COLORS[r.category] || 'var(--text-secondary)',
                    width: 70,
                    flexShrink: 0,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {r.category}
                </span>
                <span
                  style={{
                    color: 'var(--text-data)',
                    flex: 1,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {r.name}
                </span>
                {r.sae_features && (
                  <span style={{ color: '#aa88ff', flexShrink: 0, fontSize: 9 }}>SAE</span>
                )}
                <span style={{ color: r.passed ? '#44ddaa' : '#ff4466', flexShrink: 0 }}>
                  {r.passed ? '\u2713' : '\u2717'}
                </span>
              </div>
            ))}
          </div>

          {/* View details link */}
          <button
            onClick={openDetail}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--accent-active)',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
              padding: '4px 0 0',
              textDecoration: 'underline',
              textUnderlineOffset: 2,
            }}
          >
            {t('battery.viewDetails' as TranslationKey)}
          </button>
        </>
      )}

      {!result && !isRunning && (
        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
          {t('battery.noResult' as TranslationKey)}
        </div>
      )}
    </div>
  );
}
