import { useScanStore } from '../store/useScanStore';
import { useCompareStore } from '../store/useCompareStore';
import { useCrossModelStore } from '../store/useCrossModelStore';
import { CrossModelSelector } from './CrossModelSelector';
import { ScanCanvas } from './ScanCanvas/ScanCanvas';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';

const CANVAS_W = 270;
const CANVAS_H = 620;

export function CompareView() {
  const scanStore = useScanStore();
  const { dataB, selectedTokenIdx } = useCompareStore();
  const { isCrossModelMode, dataA: crossA, dataB: crossB, modelIdA, modelIdB, phase } = useCrossModelStore();
  const t = useLocaleStore((s) => s.t);

  const mode = scanStore.mode;
  const isCross = isCrossModelMode && (phase === 'done');

  // Labels
  const labelA = isCross
    ? (modelIdA ?? 'Model A')
    : scanStore.prompt.length > 28 ? scanStore.prompt.slice(0, 28) + '...' : scanStore.prompt;
  const labelB = isCross
    ? (modelIdB ?? 'Model B')
    : useCompareStore.getState().promptB.length > 28
      ? useCompareStore.getState().promptB.slice(0, 28) + '...'
      : useCompareStore.getState().promptB;

  // Data sources
  const overrideA = isCross ? {
    mode,
    structuralData: scanStore.structuralData,
    weightData: scanStore.weightData,
    activationData: crossA.activationData,
    circuitData: crossA.circuitData,
    anomalyData: crossA.anomalyData,
    selectedTokenIdx: scanStore.selectedTokenIdx,
    selectedLayerId: scanStore.selectedLayerId,
  } : {
    mode,
    structuralData: scanStore.structuralData,
    weightData: scanStore.weightData,
    activationData: scanStore.activationData,
    circuitData: scanStore.circuitData,
    anomalyData: scanStore.anomalyData,
    selectedTokenIdx: scanStore.selectedTokenIdx,
    selectedLayerId: scanStore.selectedLayerId,
  };

  const hasBData = isCross
    ? !!(crossB.activationData || crossB.circuitData || crossB.anomalyData)
    : !!(dataB.activationData || dataB.circuitData || dataB.anomalyData);

  const overrideB = isCross ? {
    mode,
    structuralData: null,
    weightData: null,
    activationData: crossB.activationData,
    circuitData: crossB.circuitData,
    anomalyData: crossB.anomalyData,
    selectedTokenIdx: scanStore.selectedTokenIdx,
  } : {
    mode,
    structuralData: scanStore.structuralData,
    weightData: scanStore.weightData,
    activationData: dataB.activationData,
    circuitData: dataB.circuitData,
    anomalyData: dataB.anomalyData,
    selectedTokenIdx,
  };

  return (
    <div className="flex flex-col">
      {isCrossModelMode && <CrossModelSelector />}
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
            A: {labelA}
          </div>
          <ScanCanvas
            width={CANVAS_W}
            height={CANVAS_H}
            dataOverride={overrideA}
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
            B: {labelB}
          </div>
          {hasBData ? (
            <ScanCanvas
              width={CANVAS_W}
              height={CANVAS_H}
              dataOverride={overrideB}
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
    </div>
  );
}
