import { useScanStore } from '../../store/useScanStore';
import { useLocaleStore } from '../../store/useLocaleStore';
import type { TranslationKey } from '../../i18n/translations';

export function LogitLensPanel() {
  const anomalyData = useScanStore((s) => s.anomalyData);
  const selectedTokenIdx = useScanStore((s) => s.selectedTokenIdx);
  const t = useLocaleStore((s) => s.t);

  if (!anomalyData) {
    return (
      <div className="px-3 py-2" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
        <div className="tracking-wide mb-1" style={{ color: 'var(--accent-active)' }}>
          {t('logitLens.title' as TranslationKey)}
        </div>
        {t('logitLens.needFLAIR' as TranslationKey)}
      </div>
    );
  }

  const layers = anomalyData.layers;
  const hasPredictions = layers.some((l) => l.top_predictions && l.top_predictions.length > 0);

  if (!hasPredictions) {
    return (
      <div className="px-3 py-2" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
        <div className="tracking-wide mb-1" style={{ color: 'var(--accent-active)' }}>
          {t('logitLens.title' as TranslationKey)}
        </div>
        No prediction data available.
      </div>
    );
  }

  // Get the final layer's top prediction for the selected token (the "answer")
  const finalLayer = layers[layers.length - 1];
  const finalPred = finalLayer.top_predictions?.[selectedTokenIdx]?.[0]?.token ?? '';

  return (
    <div className="px-3 py-2" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-primary)' }}>
      <div className="tracking-wide mb-2" style={{ color: 'var(--accent-active)' }}>
        {t('logitLens.title' as TranslationKey)}
      </div>

      <div className="mb-1" style={{ color: 'var(--text-secondary)' }}>
        Token #{selectedTokenIdx}: final = "{finalPred}"
      </div>

      <div className="max-h-64 overflow-y-auto">
        {layers.map((layer, layerIdx) => {
          const preds = layer.top_predictions?.[selectedTokenIdx];
          if (!preds || preds.length === 0) return null;
          const topToken = preds[0].token;
          const topProb = preds[0].prob;
          const matchesFinal = topToken === finalPred;

          return (
            <div key={layer.layer_id} className="flex items-center gap-1 mb-1">
              <span
                className="text-right shrink-0"
                style={{ width: 28, color: 'var(--text-secondary)' }}
              >
                L{layerIdx}
              </span>
              <span
                className="shrink-0 px-1 rounded"
                style={{
                  background: matchesFinal ? 'rgba(0,255,170,0.15)' : 'rgba(255,160,80,0.15)',
                  color: matchesFinal ? 'var(--accent-active)' : '#ffa050',
                  minWidth: 60,
                  textAlign: 'center',
                }}
              >
                {topToken.length > 8 ? topToken.slice(0, 8) + '...' : topToken}
              </span>
              {/* Probability bar */}
              <div
                className="flex-1 relative"
                style={{ height: 6, background: 'rgba(255,255,255,0.03)' }}
              >
                <div
                  className="absolute h-full rounded-sm"
                  style={{
                    width: `${Math.min(topProb * 100, 100)}%`,
                    background: matchesFinal ? 'var(--accent-active)' : '#ffa050',
                    opacity: 0.6,
                  }}
                />
              </div>
              <span
                className="text-right shrink-0"
                style={{ width: 32, color: 'var(--text-secondary)' }}
              >
                {(topProb * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
