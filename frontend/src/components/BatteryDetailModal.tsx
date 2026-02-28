import { useEffect } from 'react';
import { useBatteryStore } from '../store/useBatteryStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';
import type { TestResult, CompareResult, BatterySAESummary } from '../types/battery';

const CATEGORY_COLORS: Record<string, string> = {
  factual_recall: '#44ddaa',
  gender_bias: '#ffaa44',
  syntactic: '#aaccee',
  negation: '#ff6644',
  repetition: '#aa88ff',
  reasoning: '#44aaff',
};

function ProbBar({ prob, maxProb = 1 }: { prob: number; maxProb?: number }) {
  const width = Math.min((prob / maxProb) * 100, 100);
  return (
    <div
      style={{
        width: 80,
        height: 6,
        background: 'rgba(255,255,255,0.06)',
        borderRadius: 3,
        overflow: 'hidden',
        flexShrink: 0,
      }}
    >
      <div
        style={{
          width: `${width}%`,
          height: '100%',
          background: 'var(--accent-active)',
          borderRadius: 3,
        }}
      />
    </div>
  );
}

function CompareSection({ compare, t }: { compare: CompareResult; t: (key: TranslationKey) => string }) {
  return (
    <div className="mt-2 ml-3" style={{ fontSize: 'var(--font-size-xs)' }}>
      <div style={{ color: 'var(--text-secondary)', marginBottom: 2 }}>
        {t('battery.compare' as TranslationKey)}: "{compare.prompt}"
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {compare.top_k.slice(0, 3).map((p, i) => (
          <span key={i} style={{ color: 'var(--text-data)' }}>
            {p.token.trim() || '""'}: {(p.prob * 100).toFixed(1)}%
          </span>
        ))}
      </div>
      {compare.pronoun_probs && (
        <div className="flex gap-3 mt-1">
          {Object.entries(compare.pronoun_probs).map(([pron, prob]) => (
            <span key={pron} style={{ color: '#ffaa44' }}>
              {pron.trim()}: {(prob * 100).toFixed(1)}%
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function SAEFeaturesSection({ result, t }: { result: TestResult; t: (key: TranslationKey) => string }) {
  if (!result.sae_features) return null;

  return (
    <div className="ml-5 mt-2">
      <div style={{ fontSize: 'var(--font-size-xs)', color: '#aa88ff', marginBottom: 2 }}>
        {t('battery.saeFeatures' as TranslationKey)} (L{result.sae_features.layer_idx})
      </div>
      {result.sae_features.top_features.map((f) => (
        <div key={f.feature_idx} className="flex items-center gap-1 py-0.5" style={{ fontSize: 'var(--font-size-xs)' }}>
          <span style={{ color: '#aa88ff', width: 50, flexShrink: 0, textAlign: 'right', fontFamily: 'var(--font-primary)' }}>
            #{f.feature_idx}
          </span>
          <div style={{
            flex: 1, maxWidth: 80, height: 5, background: 'rgba(255,255,255,0.05)',
            borderRadius: 3, overflow: 'hidden',
          }}>
            <div style={{
              width: `${Math.min(f.activation_normalized * 100, 100)}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #7c3aed, #06b6d4)',
              borderRadius: 3,
            }} />
          </div>
          <span style={{ color: 'var(--text-data)', width: 40, flexShrink: 0, textAlign: 'right' }}>
            {f.activation.toFixed(1)}
          </span>
          {f.neuronpedia_url && (
            <a
              href={f.neuronpedia_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              style={{ color: '#06b6d4', textDecoration: 'none', flexShrink: 0, fontSize: 10 }}
              title="Neuronpedia"
            >
              NP
            </a>
          )}
        </div>
      ))}
    </div>
  );
}

function CrossTestSAESection({ summary, totalTests, t }: { summary: BatterySAESummary; totalTests: number; t: (key: TranslationKey) => string }) {
  return (
    <div className="mb-4 pb-3" style={{ borderBottom: '1px solid var(--border)' }}>
      <div style={{ fontSize: 'var(--font-size-sm)', fontWeight: 'bold', color: '#aa88ff', marginBottom: 6 }}>
        {t('battery.crossTestSae' as TranslationKey)} (L{summary.layer_idx})
      </div>
      <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-data)', marginBottom: 6, lineHeight: 1.6 }}>
        {summary.interpretation}
      </div>
      {summary.cross_test_features.length > 0 && (
        <div style={{ fontSize: 'var(--font-size-xs)' }}>
          {summary.cross_test_features.slice(0, 5).map((cf) => (
            <div key={cf.feature_idx} className="flex items-center gap-2 py-0.5">
              <span style={{ color: '#aa88ff', fontFamily: 'var(--font-primary)' }}>#{cf.feature_idx}</span>
              <span style={{ color: 'var(--text-secondary)' }}>
                {cf.count}/{totalTests} tests
              </span>
              <span style={{ color: 'var(--text-data)' }}>
                [{cf.categories.join(', ')}]
              </span>
              <span style={{ color: 'var(--text-data)' }}>
                avg: {cf.avg_activation.toFixed(1)}
              </span>
              {cf.neuronpedia_url && (
                <a
                  href={cf.neuronpedia_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{ color: '#06b6d4', textDecoration: 'none', fontSize: 10 }}
                >
                  NP
                </a>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TestResultCard({ result, t }: { result: TestResult; t: (key: TranslationKey) => string }) {
  const catColor = CATEGORY_COLORS[result.category] || 'var(--text-secondary)';
  const maxProb = Math.max(...result.top_k.map((p) => p.prob), 0.01);

  return (
    <div
      className="mb-4 pb-3"
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-1">
        <span style={{ color: result.passed ? '#44ddaa' : '#ff4466', fontWeight: 'bold' }}>
          {result.passed ? '\u2713' : '\u2717'}
        </span>
        <span style={{ color: catColor, fontSize: 'var(--font-size-sm)', fontWeight: 'bold' }}>
          [{result.category}]
        </span>
        <span style={{ color: 'var(--text-primary)', fontSize: 'var(--font-size-sm)' }}>
          {result.name}
        </span>
      </div>

      {/* Prompt */}
      <div style={{ color: 'var(--accent-active)', fontSize: 'var(--font-size-xs)', marginBottom: 4, paddingLeft: 20 }}>
        "{result.prompt}"
      </div>

      {/* Top-5 predictions */}
      <div className="ml-5 mb-2">
        {result.top_k.map((p, i) => (
          <div key={i} className="flex items-center gap-2 py-0.5" style={{ fontSize: 'var(--font-size-xs)' }}>
            <span style={{ color: 'var(--text-secondary)', width: 16, textAlign: 'right' }}>{i + 1}.</span>
            <span style={{ color: 'var(--text-primary)', width: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {p.token.trim() || '""'}
            </span>
            <ProbBar prob={p.prob} maxProb={maxProb} />
            <span style={{ color: 'var(--text-data)', width: 45, textAlign: 'right' }}>
              {(p.prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>

      {/* Activation Summary */}
      <div className="ml-5 mb-1" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
        {t('battery.peak' as TranslationKey)}: L{result.activation_summary.peak_layers.join(',')} ({result.activation_summary.peak_activation.toFixed(1)}) | {t('battery.activeLayers' as TranslationKey)}: {result.activation_summary.active_layer_count}
      </div>

      {/* Interpretation */}
      <div className="ml-5" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-data)', lineHeight: 1.6 }}>
        {result.interpretation}
      </div>

      {/* Compare results */}
      {result.compare_results?.map((cr, i) => (
        <CompareSection key={i} compare={cr} t={t} />
      ))}

      {/* SAE Features */}
      <SAEFeaturesSection result={result} t={t} />
    </div>
  );
}

export function BatteryDetailModal() {
  const { result, isDetailOpen, closeDetail } = useBatteryStore();
  const t = useLocaleStore((s) => s.t);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isDetailOpen) closeDetail();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isDetailOpen, closeDetail]);

  useEffect(() => {
    if (isDetailOpen) {
      document.body.style.overflow = 'hidden';
      return () => { document.body.style.overflow = ''; };
    }
  }, [isDetailOpen]);

  if (!isDetailOpen || !result) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 100,
        background: 'rgba(0,0,0,0.75)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        animation: 'nmri-fade-in 200ms ease-out',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) closeDetail();
      }}
    >
      <div
        style={{
          background: 'var(--bg-primary)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          width: '90vw',
          maxWidth: 700,
          maxHeight: '85vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center gap-3 px-5 py-3 shrink-0"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <button
            onClick={closeDetail}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: 'var(--font-size-lg)',
              cursor: 'pointer',
              fontFamily: 'var(--font-primary)',
              padding: '0 4px',
            }}
          >
            X
          </button>
          <span
            className="tracking-widest font-bold"
            style={{ color: 'var(--text-primary)', fontSize: 'var(--font-size-lg)' }}
          >
            {t('battery.detailTitle' as TranslationKey)}
          </span>
        </div>

        {/* Summary bar */}
        <div
          className="flex items-center gap-4 px-5 py-2 shrink-0"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)' }}
        >
          <span style={{ color: 'var(--text-data)', fontSize: 'var(--font-size-sm)' }}>
            {t('battery.model' as TranslationKey)}: <span style={{ color: 'var(--text-primary)' }}>{result.model_id}</span>
          </span>
          <span style={{ color: '#44ddaa', fontSize: 'var(--font-size-sm)', fontWeight: 'bold' }}>
            {result.passed} {t('battery.passed' as TranslationKey)}
          </span>
          {result.failed > 0 && (
            <span style={{ color: '#ff4466', fontSize: 'var(--font-size-sm)', fontWeight: 'bold' }}>
              {result.failed} {t('battery.failed' as TranslationKey)}
            </span>
          )}
          <span style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)' }}>
            {result.total_tests} {t('battery.total' as TranslationKey)}
          </span>
          {result.sae_summary && (
            <span style={{ color: '#aa88ff', fontSize: 'var(--font-size-xs)' }}>
              SAE L{result.sae_summary.layer_idx}
            </span>
          )}
        </div>

        {/* Results */}
        <div className="overflow-y-auto px-5 py-4" style={{ flex: 1 }}>
          {result.results.map((r) => (
            <TestResultCard key={r.test_id} result={r} t={t} />
          ))}

          {/* Cross-test SAE summary */}
          {result.sae_summary && (
            <CrossTestSAESection
              summary={result.sae_summary}
              totalTests={result.total_tests}
              t={t}
            />
          )}

          {/* Summary */}
          <div
            className="mt-2"
            style={{
              color: 'var(--text-secondary)',
              fontSize: 'var(--font-size-xs)',
              fontStyle: 'italic',
            }}
          >
            {result.summary}
          </div>
        </div>
      </div>
    </div>
  );
}
