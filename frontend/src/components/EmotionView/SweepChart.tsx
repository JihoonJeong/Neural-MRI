import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

import { useEmotionStore } from '../../store/useEmotionStore';

const ACCENT = '#e879a0';

export function SweepChart({ prompt }: { prompt: string }) {
  const { sweepResult, isSweeping, sweep, selectedEmotion, probeResult } = useEmotionStore();
  const svgRef = useRef<SVGSVGElement>(null);

  const hasProbes = probeResult !== null;
  const hasData = sweepResult !== null;

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span style={{ fontSize: 'var(--font-size-sm)', color: ACCENT, fontWeight: 'bold', letterSpacing: '0.5px' }}>
          STRENGTH SWEEP — {selectedEmotion}
        </span>
        <button
          onClick={() => sweep(prompt)}
          disabled={isSweeping || !hasProbes || !prompt}
          style={{
            fontSize: 'var(--font-size-xs)',
            padding: '2px 8px',
            color: isSweeping ? 'var(--text-secondary)' : ACCENT,
            cursor: isSweeping || !hasProbes ? 'default' : 'pointer',
            border: `1px solid ${ACCENT}40`,
            borderRadius: 3,
            background: 'transparent',
          }}
        >
          {isSweeping ? 'SWEEPING...' : 'SWEEP'}
        </button>
      </div>

      {hasData ? <SweepCanvas data={sweepResult} svgRef={svgRef} /> : (
        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', padding: '20px 0', textAlign: 'center' }}>
          {isSweeping ? `Running ${selectedEmotion} sweep...` : 'Click SWEEP to generate dose-response curve'}
        </div>
      )}
    </div>
  );
}

function SweepCanvas({
  data,
  svgRef,
}: {
  data: NonNullable<ReturnType<typeof useEmotionStore.getState>['sweepResult']>;
  svgRef: React.RefObject<SVGSVGElement | null>;
}) {
  useEffect(() => {
    if (!svgRef.current || !data) return;

    const { points } = data;
    const width = 500;
    const height = 240;
    const margin = { top: 20, right: 30, bottom: 40, left: 60 };

    const x = d3.scaleLinear()
      .domain(d3.extent(points, (p) => p.strength) as [number, number])
      .range([margin.left, width - margin.right]);

    const yExt = d3.extent(points, (p) => p.target_emotion_activation) as [number, number];
    const yPad = (yExt[1] - yExt[0]) * 0.1 || 1;
    const y = d3.scaleLinear()
      .domain([yExt[0] - yPad, yExt[1] + yPad])
      .range([height - margin.bottom, margin.top]);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    // Zero line
    svg.append('line')
      .attr('x1', x(0)).attr('x2', x(0))
      .attr('y1', margin.top).attr('y2', height - margin.bottom)
      .attr('stroke', 'var(--border)').attr('stroke-dasharray', '3,3');

    // Axes
    const xAxis = d3.axisBottom(x).ticks(7).tickFormat((d) => `${+d > 0 ? '+' : ''}${d}`);
    const yAxis = d3.axisLeft(y).ticks(5);

    svg.append('g')
      .attr('transform', `translate(0,${height - margin.bottom})`)
      .call(xAxis)
      .attr('color', 'var(--text-secondary)')
      .attr('font-size', 9);

    svg.append('g')
      .attr('transform', `translate(${margin.left},0)`)
      .call(yAxis)
      .attr('color', 'var(--text-secondary)')
      .attr('font-size', 9);

    // Axis labels
    svg.append('text')
      .attr('x', width / 2).attr('y', height - 2)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-secondary)').attr('font-size', 10)
      .text('Steering Strength');

    svg.append('text')
      .attr('x', 14).attr('y', height / 2)
      .attr('transform', `rotate(-90, 14, ${height / 2})`)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-secondary)').attr('font-size', 10)
      .text(`${data.emotion} activation`);

    // Line
    const line = d3.line<typeof points[0]>()
      .x((d) => x(d.strength))
      .y((d) => y(d.target_emotion_activation));

    svg.append('path')
      .datum(points)
      .attr('d', line)
      .attr('fill', 'none')
      .attr('stroke', ACCENT)
      .attr('stroke-width', 2);

    // Points with tooltips
    const tooltip = d3.select('body').append('div')
      .style('position', 'fixed')
      .style('display', 'none')
      .style('background', '#1a1a2e')
      .style('border', '1px solid #333')
      .style('color', '#ccc')
      .style('padding', '4px 8px')
      .style('font-size', '10px')
      .style('border-radius', '3px')
      .style('pointer-events', 'none')
      .style('z-index', '9999')
      .style('max-width', '250px');

    points.forEach((p) => {
      svg.append('circle')
        .attr('cx', x(p.strength))
        .attr('cy', y(p.target_emotion_activation))
        .attr('r', 4)
        .attr('fill', ACCENT)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .style('cursor', 'pointer')
        .on('mouseover', (event) => {
          tooltip
            .style('display', 'block')
            .style('left', `${event.clientX + 12}px`)
            .style('top', `${event.clientY - 8}px`)
            .html(`<b>strength: ${p.strength > 0 ? '+' : ''}${p.strength}</b><br/>activation: ${p.target_emotion_activation.toFixed(1)}<br/>"${p.generated_text.slice(0, 80)}..."`);
        })
        .on('mouseout', () => tooltip.style('display', 'none'));
    });

    return () => { tooltip.remove(); };
  }, [data, svgRef]);

  return (
    <div style={{ overflowX: 'auto' }}>
      <svg ref={svgRef} style={{ display: 'block' }} />
    </div>
  );
}
