import { useScanStore } from '../../store/useScanStore';
import { usePerturbStore } from '../../store/usePerturbStore';
import type { PerturbationType } from '../../store/usePerturbStore';

const OPERATIONS: { label: string; type: PerturbationType; factor?: number }[] = [
  { label: 'Zero-out', type: 'zero' },
  { label: 'Amplify 2x', type: 'amplify', factor: 2.0 },
  { label: 'Ablate', type: 'ablate' },
];

export function StimPanel() {
  const { selectedLayerId, structuralData } = useScanStore();
  const { isPerturbing, result, runPerturbation, reset } = usePerturbStore();

  const layer = structuralData?.layers.find((l) => l.layer_id === selectedLayerId);

  // Perturbation not available for embed/output layers
  const canPerturb =
    layer && layer.layer_type !== 'embedding' && layer.layer_type !== 'output';

  if (!layer) {
    return (
      <div
        className="p-4 text-center mt-4"
        style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-md)' }}
      >
        <div style={{ color: 'var(--text-primary)', marginBottom: 6 }}>STIMULATION MODE</div>
        <div>Click a node to inspect</div>
      </div>
    );
  }

  return (
    <div className="p-3" style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-data)' }}>
      <div
        className="mb-2 tracking-wide"
        style={{ color: 'var(--accent-active)', fontSize: 'var(--font-size-md)' }}
      >
        NODE SELECTED
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 mb-3">
        <span style={{ color: 'var(--text-primary)' }}>ID:</span>
        <span>{layer.layer_id}</span>
        <span style={{ color: 'var(--text-primary)' }}>Type:</span>
        <span>{layer.layer_type}</span>
        <span style={{ color: 'var(--text-primary)' }}>Params:</span>
        <span>{(layer.param_count / 1e6).toFixed(2)}M</span>
        {Object.entries(layer.shape_info).map(([k, v]) => (
          <div key={k} className="contents">
            <span style={{ color: 'var(--text-primary)' }}>{k}:</span>
            <span>{v}</span>
          </div>
        ))}
      </div>

      {/* Perturbation buttons */}
      <div className="mb-1 tracking-wide" style={{ color: 'var(--text-primary)', fontSize: 'var(--font-size-sm)' }}>
        PERTURBATION
      </div>
      <div className="flex flex-wrap gap-1.5">
        {OPERATIONS.map((op) => (
          <button
            key={op.label}
            disabled={!canPerturb || isPerturbing}
            onClick={() => {
              if (canPerturb && selectedLayerId) {
                runPerturbation(op.type, selectedLayerId, op.factor);
              }
            }}
            className="rounded-sm"
            style={{
              background: canPerturb ? 'rgba(0,255,170,0.12)' : 'rgba(0,255,170,0.08)',
              border: '1px solid rgba(0,255,170,0.25)',
              color: 'var(--accent-active)',
              padding: '3px 7px',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              opacity: canPerturb && !isPerturbing ? 1 : 0.4,
              cursor: canPerturb && !isPerturbing ? 'pointer' : 'not-allowed',
            }}
          >
            {isPerturbing ? '...' : op.label}
          </button>
        ))}
      </div>
      {!canPerturb && (
        <div className="mt-1 opacity-50" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
          Perturbation not available for {layer.layer_type} layers
        </div>
      )}
      {result && (
        <button
          onClick={reset}
          className="mt-2 rounded-sm w-full"
          style={{
            background: 'rgba(255,100,100,0.1)',
            border: '1px solid rgba(255,100,100,0.3)',
            color: '#ff6464',
            padding: '3px 7px',
            fontSize: 'var(--font-size-xs)',
            fontFamily: 'var(--font-primary)',
            cursor: 'pointer',
          }}
        >
          RESET
        </button>
      )}
    </div>
  );
}
