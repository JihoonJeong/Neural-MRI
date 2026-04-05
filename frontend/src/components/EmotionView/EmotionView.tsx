import { useEffect } from 'react';

import { useEmotionStore } from '../../store/useEmotionStore';
import { useModelStore } from '../../store/useModelStore';
import { useScanStore } from '../../store/useScanStore';
import { TranscriptHeatmap } from './TranscriptHeatmap';
import { PCAScatter } from './PCAScatter';
import { SteerControls } from './SteerControls';
import { SweepChart } from './SweepChart';
import { LayerEvolution } from './LayerEvolution';

const ACCENT = '#e879a0';

export function EmotionView() {
  const isLoaded = useModelStore((s) => s.modelInfo !== null);
  const prompt = useScanStore((s) => s.prompt);
  const {
    probeResult,
    isExtracting,
    extractProbes,
    fetchPCA,
    pcaResult,
  } = useEmotionStore();

  const error = useEmotionStore((s) => s.error);
  const hasProbes = probeResult !== null;

  // Auto-extract probes when entering emotion tab with a loaded model
  // Stop retrying if there's an error
  useEffect(() => {
    if (isLoaded && !hasProbes && !isExtracting && !error) {
      extractProbes();
    }
  }, [isLoaded, hasProbes, isExtracting, error, extractProbes]);

  // Fetch PCA after probes are ready
  useEffect(() => {
    if (hasProbes && !pcaResult) {
      fetchPCA();
    }
  }, [hasProbes, pcaResult, fetchPCA]);

  if (!isLoaded) {
    return (
      <div className="flex items-center justify-center h-full" style={{ color: 'var(--text-secondary)' }}>
        Load a model to use Emotion Analysis
      </div>
    );
  }

  if (isExtracting) {
    return (
      <div className="flex items-center justify-center h-full" style={{ color: ACCENT }}>
        <div>
          <div style={{ fontSize: 'var(--font-size-lg)', marginBottom: 8 }}>Extracting emotion probes...</div>
          <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
            21 emotions × 3 passages = 63 forward passes
          </div>
        </div>
      </div>
    );
  }

  if (!hasProbes && error) {
    return (
      <div className="flex items-center justify-center h-full" style={{ color: 'var(--text-secondary)' }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: 'var(--font-size-sm)', color: '#f87171', marginBottom: 8 }}>
            Probe extraction failed: {error}
          </div>
          <button
            onClick={() => { useEmotionStore.getState().reset(); extractProbes(); }}
            style={{
              fontSize: 'var(--font-size-xs)',
              padding: '4px 12px',
              color: ACCENT,
              border: `1px solid ${ACCENT}`,
              borderRadius: 4,
              background: 'transparent',
              cursor: 'pointer',
            }}
          >
            RETRY
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full" style={{ overflow: 'hidden' }}>
      {/* Zone 1: Main canvas area */}
      <div className="flex-1 flex flex-col" style={{ overflow: 'auto', padding: 16 }}>
        {/* A: Transcript Heatmap */}
        <TranscriptHeatmap prompt={prompt} />

        {/* B: PCA Scatter */}
        <div style={{ marginTop: 16 }}>
          <PCAScatter />
        </div>

        {/* D: Sweep Chart */}
        <div style={{ marginTop: 16 }}>
          <SweepChart prompt={prompt} />
        </div>

        {/* F: Layer Evolution */}
        <div style={{ marginTop: 16 }}>
          <LayerEvolution prompt={prompt} />
        </div>
      </div>

      {/* Zone 2: Right controls */}
      <div
        className="flex flex-col shrink-0"
        style={{
          width: 300,
          borderLeft: '1px solid var(--border)',
          background: 'var(--bg-secondary)',
          overflowY: 'auto',
          padding: 12,
        }}
      >
        <SteerControls prompt={prompt} />
      </div>
    </div>
  );
}
