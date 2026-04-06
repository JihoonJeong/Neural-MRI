import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

import { useEmotionStore } from '../../store/useEmotionStore';
import { groupTokensForDisplay } from '../../utils/tokenDisplay';

const ACCENT = '#e879a0';

export function TranscriptHeatmap({ prompt }: { prompt: string }) {
  const { projectResult, isProjecting, project, probeResult } = useEmotionStore();
  const svgRef = useRef<SVGSVGElement>(null);

  const hasProbes = probeResult !== null;
  const hasData = projectResult !== null && projectResult.prompt === prompt;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span style={{ fontSize: 'var(--font-size-sm)', color: ACCENT, fontWeight: 'bold', letterSpacing: '0.5px' }}>
          TRANSCRIPT HEATMAP
        </span>
        <button
          onClick={() => project(prompt)}
          disabled={isProjecting || !hasProbes || !prompt}
          style={{
            fontSize: 'var(--font-size-xs)',
            padding: '2px 8px',
            color: isProjecting ? 'var(--text-secondary)' : ACCENT,
            cursor: isProjecting || !hasProbes ? 'default' : 'pointer',
            border: `1px solid ${ACCENT}40`,
            borderRadius: 3,
            background: 'transparent',
          }}
        >
          {isProjecting ? 'PROJECTING...' : 'PROJECT'}
        </button>
      </div>

      {hasData ? (
        <HeatmapCanvas data={projectResult} svgRef={svgRef} />
      ) : (
        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', padding: '20px 0', textAlign: 'center' }}>
          {!hasProbes
            ? 'Extract probes first'
            : !prompt
              ? 'Enter a prompt above, then click PROJECT'
              : 'Enter a prompt above, then click PROJECT to visualize emotion activations across tokens'}
        </div>
      )}
    </div>
  );
}

function HeatmapCanvas({
  data,
  svgRef,
}: {
  data: NonNullable<ReturnType<typeof useEmotionStore.getState>['projectResult']>;
  svgRef: React.RefObject<SVGSVGElement | null>;
}) {
  const setSelectedEmotion = useEmotionStore((s) => s.setSelectedEmotion);

  useEffect(() => {
    if (!svgRef.current || !data) return;

    const emotions = data.emotions;
    const rawTokens = data.tokens;

    // Group byte-level tokens into displayable unicode segments
    const displayTokens = groupTokensForDisplay(rawTokens.map((t) => t.token_str));

    // Merge activations for grouped tokens (average)
    const mergedTokens = displayTokens.map((dt) => {
      const merged: Record<string, number> = {};
      for (const e of emotions) {
        let sum = 0;
        for (const idx of dt.indices) {
          sum += rawTokens[idx]?.activations[e] ?? 0;
        }
        merged[e] = sum / dt.indices.length;
      }
      return { label: dt.label, activations: merged, indices: dt.indices };
    });

    const nEmotions = emotions.length;
    const nTokens = mergedTokens.length;

    const cellW = Math.max(24, Math.min(50, 700 / nTokens));
    const cellH = 16;
    const margin = { top: 80, right: 20, bottom: 20, left: 100 };
    const width = margin.left + nTokens * cellW + margin.right;
    const height = margin.top + nEmotions * cellH + margin.bottom;

    // Collect all values for color scale
    const allVals: number[] = [];
    mergedTokens.forEach((t) => emotions.forEach((e) => allVals.push(t.activations[e] ?? 0)));
    const maxAbs = Math.max(...allVals.map(Math.abs), 0.01);

    const colorScale = d3.scaleDiverging<string>()
      .domain([-maxAbs, 0, maxAbs])
      .interpolator(d3.interpolateRdBu)
      .clamp(true);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const g = svg.append('g');

    // Token labels (top, rotated)
    mergedTokens.forEach((t, ti) => {
      g.append('text')
        .attr('x', margin.left + ti * cellW + cellW / 2)
        .attr('y', margin.top - 6)
        .attr('text-anchor', 'start')
        .attr('transform', `rotate(-55, ${margin.left + ti * cellW + cellW / 2}, ${margin.top - 6})`)
        .attr('fill', 'var(--text-secondary)')
        .attr('font-size', 9)
        .text(t.label.trim() || '?');
    });

    // Emotion labels (left)
    emotions.forEach((e, ei) => {
      g.append('text')
        .attr('x', margin.left - 6)
        .attr('y', margin.top + ei * cellH + cellH / 2 + 4)
        .attr('text-anchor', 'end')
        .attr('fill', 'var(--text-secondary)')
        .attr('font-size', 10)
        .style('cursor', 'pointer')
        .text(e)
        .on('click', () => setSelectedEmotion(e));
    });

    // Cells
    const tooltip = d3.select('body').append('div')
      .attr('class', 'emotion-tooltip')
      .style('position', 'fixed')
      .style('display', 'none')
      .style('background', '#1a1a2e')
      .style('border', '1px solid #333')
      .style('color', '#ccc')
      .style('padding', '4px 8px')
      .style('font-size', '11px')
      .style('border-radius', '3px')
      .style('pointer-events', 'none')
      .style('z-index', '9999');

    mergedTokens.forEach((t, ti) => {
      emotions.forEach((e, ei) => {
        const val = t.activations[e] ?? 0;
        g.append('rect')
          .attr('x', margin.left + ti * cellW)
          .attr('y', margin.top + ei * cellH)
          .attr('width', cellW - 1)
          .attr('height', cellH - 1)
          .attr('fill', colorScale(val))
          .attr('rx', 1)
          .style('cursor', 'pointer')
          .on('mouseover', (event) => {
            tooltip
              .style('display', 'block')
              .style('left', `${event.clientX + 12}px`)
              .style('top', `${event.clientY - 8}px`)
              .html(`<b>${e}</b> @ "${t.label.trim()}"<br/>activation: ${val.toFixed(2)}`);
          })
          .on('mousemove', (event) => {
            tooltip
              .style('left', `${event.clientX + 12}px`)
              .style('top', `${event.clientY - 8}px`);
          })
          .on('mouseout', () => tooltip.style('display', 'none'))
          .on('click', () => setSelectedEmotion(e));
      });
    });

    return () => { tooltip.remove(); };
  }, [data, svgRef, setSelectedEmotion]);

  return (
    <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 500 }}>
      <svg ref={svgRef} style={{ display: 'block' }} />
    </div>
  );
}
