import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

import { useEmotionStore } from '../../store/useEmotionStore';

const ACCENT = '#e879a0';
const COLORS = [
  '#e879a0', '#4ade80', '#60a5fa', '#facc15', '#c084fc',
  '#fb923c', '#22d3ee', '#f87171', '#a3e635', '#e879f0',
  '#94a3b8', '#fbbf24', '#34d399', '#818cf8', '#f472b6',
  '#a78bfa', '#38bdf8', '#fb7185', '#86efac', '#fde68a', '#cbd5e1',
];

export function LayerEvolution({ prompt }: { prompt: string }) {
  const { layerEvoResult, layerEvolution, probeResult, selectedEmotion } = useEmotionStore();
  const svgRef = useRef<SVGSVGElement>(null);

  const hasProbes = probeResult !== null;
  const hasData = layerEvoResult !== null;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span style={{ fontSize: 'var(--font-size-sm)', color: ACCENT, fontWeight: 'bold', letterSpacing: '0.5px' }}>
          LAYER EVOLUTION
        </span>
        <button
          onClick={() => layerEvolution(prompt)}
          disabled={!hasProbes || !prompt}
          style={{
            fontSize: 'var(--font-size-xs)',
            padding: '2px 8px',
            color: hasProbes ? ACCENT : 'var(--text-secondary)',
            cursor: !hasProbes ? 'default' : 'pointer',
            border: `1px solid ${ACCENT}40`,
            borderRadius: 3,
            background: 'transparent',
          }}
        >
          ANALYZE
        </button>
      </div>

      {hasData ? (
        <LayerCanvas data={layerEvoResult} svgRef={svgRef} selectedEmotion={selectedEmotion} />
      ) : (
        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', padding: '20px 0', textAlign: 'center' }}>
          Shows how emotion activations change from early (sensory) to late (action) layers — click ANALYZE
        </div>
      )}
    </div>
  );
}

function LayerCanvas({
  data,
  svgRef,
  selectedEmotion,
}: {
  data: NonNullable<ReturnType<typeof useEmotionStore.getState>['layerEvoResult']>;
  svgRef: React.RefObject<SVGSVGElement | null>;
  selectedEmotion: string;
}) {
  useEffect(() => {
    if (!svgRef.current || !data) return;

    const { emotions, layers } = data;
    const width = 500;
    const height = 260;
    const margin = { top: 20, right: 120, bottom: 40, left: 50 };

    const x = d3.scaleLinear()
      .domain([0, layers.length - 1])
      .range([margin.left, width - margin.right]);

    const allVals = layers.flatMap((l) => emotions.map((e) => l.activations[e] ?? 0));
    const yExt = d3.extent(allVals) as [number, number];
    const yPad = (yExt[1] - yExt[0]) * 0.1 || 1;
    const y = d3.scaleLinear()
      .domain([yExt[0] - yPad, yExt[1] + yPad])
      .range([height - margin.bottom, margin.top]);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    // Axes
    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(d3.axisBottom(x).ticks(layers.length > 20 ? 10 : layers.length).tickFormat((d) => `${d}`))
      .attr('color', 'var(--text-secondary)').attr('font-size', 9);
    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(d3.axisLeft(y).ticks(5))
      .attr('color', 'var(--text-secondary)').attr('font-size', 9);

    svg.append('text')
      .attr('x', (margin.left + width - margin.right) / 2).attr('y', height - 2)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-secondary)').attr('font-size', 10)
      .text('Layer');

    // Zero line
    svg.append('line')
      .attr('x1', margin.left).attr('x2', width - margin.right)
      .attr('y1', y(0)).attr('y2', y(0))
      .attr('stroke', 'var(--border)').attr('stroke-dasharray', '3,3');

    // Lines per emotion
    emotions.forEach((emotion, ei) => {
      const isSelected = emotion === selectedEmotion;
      const color = COLORS[ei % COLORS.length];
      const lineData = layers.map((l) => ({ layer: l.layer_idx, val: l.activations[emotion] ?? 0 }));

      const line = d3.line<typeof lineData[0]>()
        .x((d) => x(d.layer))
        .y((d) => y(d.val));

      svg.append('path')
        .datum(lineData)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', color)
        .attr('stroke-width', isSelected ? 2.5 : 1)
        .attr('opacity', isSelected ? 1 : 0.4);

      // Legend label at right edge
      const lastVal = lineData[lineData.length - 1];
      svg.append('text')
        .attr('x', width - margin.right + 6)
        .attr('y', y(lastVal.val) + 3)
        .attr('fill', color)
        .attr('font-size', isSelected ? 10 : 8)
        .attr('font-weight', isSelected ? 'bold' : 'normal')
        .attr('opacity', isSelected ? 1 : 0.6)
        .text(emotion);
    });

    // Token info
    svg.append('text')
      .attr('x', margin.left).attr('y', margin.top - 6)
      .attr('fill', 'var(--text-secondary)').attr('font-size', 9)
      .text(`Token: "${data.token_str.trim()}" (idx ${data.token_idx})`);
  }, [data, svgRef, selectedEmotion]);

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg ref={svgRef} style={{ display: 'block' }} />
    </div>
  );
}
