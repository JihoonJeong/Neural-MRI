import { usePerturbStore } from '../../store/usePerturbStore';
import type { TokenPrediction } from '../../types/perturb';

function PredictionRow({ pred, rank }: { pred: TokenPrediction; rank: number }) {
  return (
    <div className="flex items-center gap-2" style={{ fontSize: 'var(--font-size-xs)' }}>
      <span style={{ color: 'var(--text-secondary)', width: 14 }}>{rank}.</span>
      <span
        className="rounded px-1"
        style={{ background: 'rgba(255,255,255,0.04)', color: 'var(--text-data)' }}
      >
        {pred.token}
      </span>
      <span style={{ color: 'var(--text-secondary)', marginLeft: 'auto' }}>
        {(pred.prob * 100).toFixed(1)}%
      </span>
    </div>
  );
}

export function ComparisonPanel() {
  const result = usePerturbStore((s) => s.result);
  const error = usePerturbStore((s) => s.error);

  if (error) {
    return (
      <div className="p-3" style={{ fontSize: 'var(--font-size-xs)', color: '#ff6464' }}>
        Perturbation error: {error}
      </div>
    );
  }

  if (!result) return null;

  const logitColor = result.logit_diff < 0 ? '#ff6464' : '#64ff96';
  const klColor = result.kl_divergence > 0.5 ? '#ff6464' : result.kl_divergence > 0.1 ? '#ffaa44' : '#64ff96';

  return (
    <div className="p-3" style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-data)' }}>
      <div
        className="mb-2 tracking-wide"
        style={{ color: 'var(--text-primary)', fontSize: 'var(--font-size-sm)' }}
      >
        COMPARISON
      </div>

      {/* Original vs Perturbed top prediction */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)', marginBottom: 2 }}>
            ORIGINAL
          </div>
          <div
            className="rounded px-2 py-1 text-center"
            style={{
              background: 'rgba(0,255,170,0.08)',
              border: '1px solid rgba(0,255,170,0.2)',
              color: 'var(--accent-active)',
              fontSize: 'var(--font-size-md)',
            }}
          >
            {result.original.token}
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
              {(result.original.prob * 100).toFixed(1)}%
            </div>
          </div>
        </div>
        <div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)', marginBottom: 2 }}>
            PERTURBED
          </div>
          <div
            className="rounded px-2 py-1 text-center"
            style={{
              background: 'rgba(255,100,100,0.08)',
              border: '1px solid rgba(255,100,100,0.2)',
              color: '#ff9494',
              fontSize: 'var(--font-size-md)',
            }}
          >
            {result.perturbed.token}
            <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
              {(result.perturbed.prob * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Delta metrics */}
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 mb-3" style={{ fontSize: 'var(--font-size-xs)' }}>
        <span style={{ color: 'var(--text-primary)' }}>Logit diff:</span>
        <span style={{ color: logitColor }}>{result.logit_diff.toFixed(3)}</span>
        <span style={{ color: 'var(--text-primary)' }}>KL divergence:</span>
        <span style={{ color: klColor }}>{result.kl_divergence.toFixed(4)}</span>
        <span style={{ color: 'var(--text-primary)' }}>Component:</span>
        <span style={{ color: 'var(--text-secondary)' }}>{result.component}</span>
        <span style={{ color: 'var(--text-primary)' }}>Type:</span>
        <span style={{ color: 'var(--text-secondary)' }}>{result.perturbation_type}</span>
      </div>

      {/* Top-k predictions side by side */}
      <div className="grid grid-cols-2 gap-2">
        <div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)', marginBottom: 2 }}>
            Top-3 Original
          </div>
          {result.top_k_original.slice(0, 3).map((p, i) => (
            <PredictionRow key={i} pred={p} rank={i + 1} />
          ))}
        </div>
        <div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)', marginBottom: 2 }}>
            Top-3 Perturbed
          </div>
          {result.top_k_perturbed.slice(0, 3).map((p, i) => (
            <PredictionRow key={i} pred={p} rank={i + 1} />
          ))}
        </div>
      </div>
    </div>
  );
}
