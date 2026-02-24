import { useScanStore } from '../../store/useScanStore';
import { SCAN_MODES } from '../../types/model';

export function LayerSummary() {
  const { mode, structuralData, weightData, activationData, circuitData, anomalyData, selectedTokenIdx } =
    useScanStore();

  // Build per-layer summary data
  const layerData = (() => {
    if (mode === 'T1' && structuralData) {
      const maxParam = Math.max(...structuralData.layers.map((l) => l.param_count), 1);
      return structuralData.layers.map((l) => ({
        label: l.layer_id.length > 12 ? l.layer_id.replace('blocks.', 'B') : l.layer_id,
        value: l.param_count / maxParam,
        raw: `${(l.param_count / 1e6).toFixed(1)}M`,
      }));
    }
    if (mode === 'T2' && weightData) {
      // Group by layer_id, average l2_norm
      const grouped = new Map<string, number[]>();
      for (const w of weightData.layers) {
        const arr = grouped.get(w.layer_id) ?? [];
        arr.push(w.l2_norm);
        grouped.set(w.layer_id, arr);
      }
      const entries = Array.from(grouped.entries()).map(([id, norms]) => ({
        label: id.length > 12 ? id.replace('blocks.', 'B') : id,
        avg: norms.reduce((a, b) => a + b, 0) / norms.length,
      }));
      const maxNorm = Math.max(...entries.map((e) => e.avg), 1);
      return entries.map((e) => ({
        label: e.label,
        value: e.avg / maxNorm,
        raw: e.avg.toFixed(1),
      }));
    }
    if (mode === 'fMRI' && activationData) {
      // Show per-layer activation for the selected token
      return activationData.layers.map((l) => {
        const act = selectedTokenIdx < l.activations.length ? l.activations[selectedTokenIdx] : 0;
        return {
          label: l.layer_id.length > 12 ? l.layer_id.replace('blocks.', 'B') : l.layer_id,
          value: act,
          raw: act.toFixed(3),
        };
      });
    }
    if (mode === 'DTI' && circuitData) {
      // Show component importance
      return circuitData.components.map((c) => ({
        label: c.layer_id.length > 12 ? c.layer_id.replace('blocks.', 'B') : c.layer_id,
        value: c.importance,
        raw: c.importance.toFixed(3),
        isPathway: c.is_pathway,
      }));
    }
    if (mode === 'FLAIR' && anomalyData) {
      return anomalyData.layers.map((l) => {
        const score = selectedTokenIdx < l.anomaly_scores.length ? l.anomaly_scores[selectedTokenIdx] : 0;
        return {
          label: l.layer_id.length > 12 ? l.layer_id.replace('blocks.', 'B') : l.layer_id,
          value: score,
          raw: score.toFixed(3),
        };
      });
    }
    return [];
  })();

  const barColor = SCAN_MODES[mode].color;

  return (
    <div className="p-3" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-primary)' }}>
      <div className="mb-2 tracking-wide">
        LAYER SUMMARY
        {mode === 'fMRI' && activationData && (
          <span style={{ color: 'var(--text-secondary)', marginLeft: 6 }}>
            [{activationData.tokens[selectedTokenIdx] ?? ''}]
          </span>
        )}
        {mode === 'FLAIR' && anomalyData && (
          <span style={{ color: 'var(--text-secondary)', marginLeft: 6 }}>
            [{anomalyData.tokens[selectedTokenIdx] ?? ''}]
          </span>
        )}
      </div>
      <div className="max-h-64 overflow-y-auto">
        {layerData.length === 0 ? (
          <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)' }}>
            Run a scan to see layer data
          </div>
        ) : (
          layerData.map((d) => (
            <div key={d.label} className="flex items-center gap-2 mb-1">
              <span
                className="text-right shrink-0"
                style={{
                  width: 60,
                  color: 'isPathway' in d && d.isPathway ? barColor : 'var(--text-secondary)',
                  fontSize: 'var(--font-size-xs)',
                }}
              >
                {d.label}
              </span>
              <div
                className="flex-1 rounded-sm overflow-hidden"
                style={{ height: 7, background: 'rgba(255,255,255,0.03)' }}
              >
                <div
                  className="h-full rounded-sm transition-all duration-500"
                  style={{
                    width: `${d.value * 100}%`,
                    background: barColor,
                    opacity: 'isPathway' in d && d.isPathway ? 1 : 0.7,
                  }}
                />
              </div>
              <span
                className="text-right shrink-0"
                style={{ width: 40, color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)' }}
              >
                {d.raw}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
