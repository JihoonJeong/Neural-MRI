import { useCompareStore } from '../../store/useCompareStore';
import { useCrossModelStore } from '../../store/useCrossModelStore';
import { useLocaleStore } from '../../store/useLocaleStore';
import type { TranslationKey } from '../../i18n/translations';

export function DiffPanel() {
  const compareDiff = useCompareStore((s) => s.diffData);
  const crossModelDiff = useCrossModelStore((s) => s.diffData);
  const isCrossModel = useCrossModelStore((s) => s.isCrossModelMode);
  const t = useLocaleStore((s) => s.t);

  const diffData = isCrossModel ? crossModelDiff : compareDiff;

  if (!diffData) return null;

  const { layerDiffs, maxAbsDelta } = diffData;

  return (
    <div className="p-3" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-primary)' }}>
      <div className="mb-2 tracking-wide">
        {t('compare.diff' as TranslationKey)}
      </div>
      <div className="max-h-64 overflow-y-auto">
        {layerDiffs.map((d) => {
          const pct = maxAbsDelta > 0 ? d.delta / maxAbsDelta : 0;
          const isPositive = d.delta >= 0;
          const barWidth = Math.abs(pct) * 50;
          const label = d.layer_id.length > 12 ? d.layer_id.replace('blocks.', 'B') : d.layer_id;

          return (
            <div key={d.layer_id} className="flex items-center gap-1 mb-1">
              <span
                className="text-right shrink-0"
                style={{ width: 50, color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)' }}
              >
                {label}
              </span>
              <div
                className="flex-1 relative"
                style={{ height: 7, background: 'rgba(255,255,255,0.03)' }}
              >
                <div
                  style={{
                    position: 'absolute',
                    left: '50%',
                    top: 0,
                    bottom: 0,
                    width: 1,
                    background: 'rgba(255,255,255,0.1)',
                  }}
                />
                <div
                  className="absolute h-full rounded-sm transition-all duration-300"
                  style={{
                    left: isPositive ? '50%' : `${50 - barWidth}%`,
                    width: `${barWidth}%`,
                    background: isPositive ? 'var(--accent-active)' : '#ff6464',
                    opacity: 0.8,
                  }}
                />
              </div>
              <span
                className="text-right shrink-0"
                style={{
                  width: 42,
                  color: isPositive ? 'var(--accent-active)' : '#ff6464',
                  fontSize: 'var(--font-size-xs)',
                }}
              >
                {isPositive ? '+' : ''}{d.delta.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
      <div
        className="flex justify-between mt-2"
        style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
      >
        <span style={{ color: '#ff6464' }}>A {t('compare.stronger' as TranslationKey)}</span>
        <span style={{ color: 'var(--accent-active)' }}>B {t('compare.stronger' as TranslationKey)}</span>
      </div>
    </div>
  );
}
