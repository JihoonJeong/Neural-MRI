import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

import { useEmotionStore } from '../../store/useEmotionStore';
import { useModelStore } from '../../store/useModelStore';
import { useScanStore } from '../../store/useScanStore';
import { useLocaleStore } from '../../store/useLocaleStore';
import type { TranslationKey } from '../../i18n/translations';
import type { EmotionActivation } from '../../types/emotion';

const ACCENT = '#e879a0';
const ACCENT_DIM = '#a0506a';

export function EmotionPanel() {
  const t = useLocaleStore((s) => s.t);
  const isLoaded = useModelStore((s) => s.modelInfo !== null);
  const prompt = useScanStore((s) => s.prompt);

  const {
    emotionList,
    probeResult,
    steerResult,
    isExtracting,
    isSteering,
    error,
    selectedEmotion,
    strength,
    fetchList,
    extractProbes,
    steer,
    setSelectedEmotion,
    setStrength,
  } = useEmotionStore();

  useEffect(() => {
    if (isLoaded) fetchList();
  }, [isLoaded, fetchList]);

  const hasProbes = probeResult !== null;

  return (
    <div className="px-3 py-2">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span
          className="tracking-wide font-bold"
          style={{ fontSize: 'var(--font-size-xs)', color: ACCENT }}
        >
          {t('emotion.title' as TranslationKey)}
        </span>
        {!hasProbes && (
          <button
            onClick={extractProbes}
            disabled={isExtracting || !isLoaded}
            style={{
              fontSize: 'var(--font-size-xs)',
              padding: '1px 6px',
              color: isExtracting ? 'var(--text-secondary)' : ACCENT,
              cursor: isExtracting || !isLoaded ? 'default' : 'pointer',
              opacity: !isLoaded ? 0.4 : 1,
            }}
          >
            {isExtracting
              ? t('emotion.extracting' as TranslationKey)
              : t('emotion.extract' as TranslationKey)}
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            fontSize: 'var(--font-size-xs)',
            color: '#f87171',
            marginBottom: 4,
          }}
        >
          {error.length > 80 ? error.slice(0, 80) + '...' : error}
        </div>
      )}

      {/* Probes extracted — show steering controls */}
      {hasProbes && (
        <>
          <div
            style={{
              fontSize: 'var(--font-size-xs)',
              color: 'var(--text-secondary)',
              marginBottom: 6,
            }}
          >
            {probeResult.n_emotions} emotions @ layer {probeResult.layer_idx}
          </div>

          {/* Emotion selector */}
          <div className="flex items-center gap-2 mb-1">
            <select
              value={selectedEmotion}
              onChange={(e) => setSelectedEmotion(e.target.value)}
              style={{
                fontSize: 'var(--font-size-xs)',
                background: 'var(--bg-primary)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border)',
                borderRadius: 3,
                padding: '1px 4px',
                flex: 1,
              }}
            >
              {(emotionList?.emotions ?? probeResult.emotions).map((e) => (
                <option key={e} value={e}>
                  {e}
                </option>
              ))}
            </select>
          </div>

          {/* Strength slider */}
          <div className="flex items-center gap-2 mb-2">
            <span
              style={{
                fontSize: 'var(--font-size-xs)',
                color: 'var(--text-secondary)',
                minWidth: 48,
              }}
            >
              {strength > 0 ? '+' : ''}
              {strength.toFixed(2)}
            </span>
            <input
              type="range"
              min={-0.2}
              max={0.2}
              step={0.01}
              value={strength}
              onChange={(e) => setStrength(parseFloat(e.target.value))}
              style={{ flex: 1, accentColor: ACCENT }}
            />
            <button
              onClick={() => steer(prompt)}
              disabled={isSteering || !prompt}
              style={{
                fontSize: 'var(--font-size-xs)',
                padding: '1px 6px',
                color: isSteering ? 'var(--text-secondary)' : ACCENT,
                cursor: isSteering || !prompt ? 'default' : 'pointer',
              }}
            >
              {isSteering
                ? t('emotion.steering' as TranslationKey)
                : t('emotion.steer' as TranslationKey)}
            </button>
          </div>
        </>
      )}

      {/* Steer result */}
      {steerResult && (
        <>
          {/* Comparison texts */}
          <ComparisonTexts
            original={steerResult.comparison.original_text}
            steered={steerResult.comparison.steered_text}
            emotion={steerResult.comparison.emotion}
            strength={steerResult.comparison.strength}
          />

          {/* Emotion activation bars */}
          <EmotionBars
            original={steerResult.original_emotions}
            steered={steerResult.steered_emotions}
          />
        </>
      )}
    </div>
  );
}

/* ---------- Comparison Texts ---------- */

function ComparisonTexts({
  original,
  steered,
  emotion,
  strength,
}: {
  original: string;
  steered: string;
  emotion: string;
  strength: number;
}) {
  const label = `${emotion} ${strength > 0 ? '+' : ''}${strength.toFixed(2)}`;
  return (
    <div style={{ fontSize: 'var(--font-size-xs)', marginBottom: 8 }}>
      <div style={{ color: 'var(--text-secondary)', marginBottom: 2 }}>Original:</div>
      <div
        style={{
          color: 'var(--text-primary)',
          background: 'var(--bg-primary)',
          padding: '3px 6px',
          borderRadius: 3,
          marginBottom: 4,
          maxHeight: 60,
          overflow: 'auto',
          lineHeight: 1.4,
        }}
      >
        {original || '(empty)'}
      </div>
      <div style={{ color: ACCENT, marginBottom: 2 }}>Steered ({label}):</div>
      <div
        style={{
          color: 'var(--text-primary)',
          background: 'var(--bg-primary)',
          padding: '3px 6px',
          borderRadius: 3,
          border: `1px solid ${ACCENT_DIM}`,
          maxHeight: 60,
          overflow: 'auto',
          lineHeight: 1.4,
        }}
      >
        {steered || '(empty)'}
      </div>
    </div>
  );
}

/* ---------- Emotion Activation Bars (D3) ---------- */

function EmotionBars({
  original,
  steered,
}: {
  original: EmotionActivation[];
  steered: EmotionActivation[];
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !original.length) return;

    const emotions = original.map((e) => e.emotion);
    const origMap = new Map(original.map((e) => [e.emotion, e.activation]));
    const steerMap = new Map(steered.map((e) => [e.emotion, e.activation]));

    // Sort by absolute difference (most changed first)
    const sorted = [...emotions].sort((a, b) => {
      const diffA = Math.abs((steerMap.get(a) ?? 0) - (origMap.get(a) ?? 0));
      const diffB = Math.abs((steerMap.get(b) ?? 0) - (origMap.get(b) ?? 0));
      return diffB - diffA;
    });

    const top = sorted.slice(0, 12);
    const barH = 8;
    const gap = 2;
    const rowH = barH * 2 + gap + 4;
    const margin = { top: 4, right: 8, bottom: 4, left: 72 };
    const width = 220;
    const height = margin.top + top.length * rowH + margin.bottom;

    const allVals = [...origMap.values(), ...steerMap.values()];
    const maxAbs = Math.max(...allVals.map(Math.abs), 0.01);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const x = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([margin.left, width - margin.right]);

    const g = svg.append('g');

    // Zero line
    g.append('line')
      .attr('x1', x(0))
      .attr('x2', x(0))
      .attr('y1', margin.top)
      .attr('y2', height - margin.bottom)
      .attr('stroke', 'var(--border)')
      .attr('stroke-width', 0.5);

    top.forEach((emotion, i) => {
      const y = margin.top + i * rowH;
      const origVal = origMap.get(emotion) ?? 0;
      const steerVal = steerMap.get(emotion) ?? 0;

      // Label
      g.append('text')
        .attr('x', margin.left - 4)
        .attr('y', y + barH + 2)
        .attr('text-anchor', 'end')
        .attr('fill', 'var(--text-secondary)')
        .attr('font-size', 9)
        .text(emotion);

      // Original bar
      const ox1 = Math.min(x(0), x(origVal));
      const ow = Math.abs(x(origVal) - x(0));
      g.append('rect')
        .attr('x', ox1)
        .attr('y', y)
        .attr('width', ow)
        .attr('height', barH)
        .attr('fill', '#6b7280')
        .attr('rx', 1);

      // Steered bar
      const sx1 = Math.min(x(0), x(steerVal));
      const sw = Math.abs(x(steerVal) - x(0));
      g.append('rect')
        .attr('x', sx1)
        .attr('y', y + barH + gap)
        .attr('width', sw)
        .attr('height', barH)
        .attr('fill', ACCENT)
        .attr('rx', 1);
    });
  }, [original, steered]);

  return (
    <div>
      <div
        style={{
          fontSize: 'var(--font-size-xs)',
          color: 'var(--text-secondary)',
          marginBottom: 2,
        }}
      >
        Activation shift{' '}
        <span style={{ color: '#6b7280' }}>original</span> /{' '}
        <span style={{ color: ACCENT }}>steered</span>
      </div>
      <svg ref={svgRef} style={{ display: 'block' }} />
    </div>
  );
}
