import { useCallback, useEffect, useRef, useState } from 'react';

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
  const modelId = useModelStore((s) => s.modelInfo?.model_id);
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

  // Reset probes when model changes
  useEffect(() => {
    useEmotionStore.getState().reset();
  }, [modelId]);

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
    return <ExtractionProgress />;
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

  const containerRef = useRef<HTMLDivElement>(null);
  const [rightWidth, setRightWidth] = useState(
    () => parseInt(localStorage.getItem('nmri-emo-right-width') ?? '360', 10)
  );
  const isDragging = useRef(false);

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isDragging.current = true;

    const onMouseMove = (ev: MouseEvent) => {
      if (!isDragging.current || !containerRef.current) return;
      const rect = containerRef.current.getBoundingClientRect();
      const newRight = Math.max(260, Math.min(rect.width - 300, rect.right - ev.clientX));
      setRightWidth(newRight);
      localStorage.setItem('nmri-emo-right-width', String(newRight));
    };

    const onMouseUp = () => {
      isDragging.current = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };

    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }, []);

  return (
    <div ref={containerRef} className="flex h-full" style={{ overflow: 'hidden' }}>
      {/* Zone 1: Main canvas area */}
      <div className="flex-1 flex flex-col" style={{ overflow: 'auto', padding: 16, minWidth: 300 }}>
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

      {/* Draggable divider */}
      <div
        onMouseDown={onMouseDown}
        style={{
          width: 5,
          cursor: 'col-resize',
          background: 'transparent',
          flexShrink: 0,
          position: 'relative',
        }}
      >
        <div style={{
          position: 'absolute',
          left: 2,
          top: 0,
          bottom: 0,
          width: 1,
          background: 'var(--border)',
        }} />
      </div>

      {/* Zone 2: Right controls */}
      <div
        className="flex flex-col shrink-0"
        style={{
          width: rightWidth,
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

/* ---------- Extraction Progress ---------- */

function ExtractionProgress() {
  const TOTAL = 63;
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setElapsed((e) => e + 0.1), 100);
    return () => clearInterval(t);
  }, []);

  // Estimate ~50ms per forward pass for GPT-2, scale up for larger models
  const estimatedPerPass = 0.05; // seconds
  const estimatedProgress = Math.min(Math.floor(elapsed / estimatedPerPass), TOTAL);
  const pct = Math.min((estimatedProgress / TOTAL) * 100, 99);

  return (
    <div className="flex items-center justify-center h-full" style={{ color: ACCENT }}>
      <div style={{ textAlign: 'center', width: 320 }}>
        <div style={{ fontSize: 'var(--font-size-lg)', marginBottom: 12 }}>
          Extracting emotion probes...
        </div>
        <div style={{
          fontSize: 'var(--font-size-sm)',
          color: 'var(--text-secondary)',
          marginBottom: 12,
        }}>
          {estimatedProgress}/{TOTAL} forward passes
        </div>
        {/* Progress bar */}
        <div style={{
          width: '100%',
          height: 4,
          background: 'var(--bg-primary)',
          borderRadius: 2,
          overflow: 'hidden',
        }}>
          <div style={{
            width: `${pct}%`,
            height: '100%',
            background: ACCENT,
            borderRadius: 2,
            transition: 'width 0.1s linear',
          }} />
        </div>
        <div style={{
          fontSize: 'var(--font-size-xs)',
          color: 'var(--text-secondary)',
          marginTop: 8,
        }}>
          21 emotions × 3 passages
        </div>
      </div>
    </div>
  );
}
