import { useScanStore } from '../store/useScanStore';
import { useCompareStore } from '../store/useCompareStore';
import { ScanCanvas } from './ScanCanvas/ScanCanvas';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';

const CANVAS_W = 270;
const CANVAS_H = 620;

export function CompareView() {
  const scanStore = useScanStore();
  const { dataB, selectedTokenIdx } = useCompareStore();
  const t = useLocaleStore((s) => s.t);

  const mode = scanStore.mode;

  return (
    <div className="flex gap-1" style={{ justifyContent: 'center' }}>
      {/* Canvas A */}
      <div>
        <div
          style={{
            fontSize: 'var(--font-size-xs)',
            color: 'var(--accent-active)',
            padding: '4px 8px',
            letterSpacing: '1px',
            textAlign: 'center',
            borderBottom: '1px solid var(--border)',
          }}
        >
          A: {scanStore.prompt.length > 28 ? scanStore.prompt.slice(0, 28) + '...' : scanStore.prompt}
        </div>
        <ScanCanvas
          width={CANVAS_W}
          height={CANVAS_H}
          dataOverride={{
            mode,
            structuralData: scanStore.structuralData,
            weightData: scanStore.weightData,
            activationData: scanStore.activationData,
            circuitData: scanStore.circuitData,
            anomalyData: scanStore.anomalyData,
            selectedTokenIdx: scanStore.selectedTokenIdx,
            selectedLayerId: scanStore.selectedLayerId,
          }}
          label="A"
          hideEmptyStates
        />
      </div>

      {/* Canvas B */}
      <div>
        <div
          style={{
            fontSize: 'var(--font-size-xs)',
            color: '#ff9494',
            padding: '4px 8px',
            letterSpacing: '1px',
            textAlign: 'center',
            borderBottom: '1px solid var(--border)',
          }}
        >
          B: {useCompareStore.getState().promptB.length > 28
            ? useCompareStore.getState().promptB.slice(0, 28) + '...'
            : useCompareStore.getState().promptB}
        </div>
        {dataB.activationData || dataB.circuitData || dataB.anomalyData ? (
          <ScanCanvas
            width={CANVAS_W}
            height={CANVAS_H}
            dataOverride={{
              mode,
              structuralData: scanStore.structuralData,
              weightData: scanStore.weightData,
              activationData: dataB.activationData,
              circuitData: dataB.circuitData,
              anomalyData: dataB.anomalyData,
              selectedTokenIdx,
            }}
            label="B"
            hideEmptyStates
          />
        ) : (
          <div
            className="flex items-center justify-center"
            style={{
              width: CANVAS_W,
              height: CANVAS_H,
              color: 'var(--text-secondary)',
              fontSize: 'var(--font-size-sm)',
            }}
          >
            {t('compare.scan' as TranslationKey)}
          </div>
        )}
      </div>
    </div>
  );
}
