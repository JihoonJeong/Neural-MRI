import { useEffect, useRef } from 'react';
import * as d3 from 'd3';

import { useEmotionStore } from '../../store/useEmotionStore';

const ACCENT = '#e879a0';

export function PCAScatter() {
  const { pcaResult } = useEmotionStore();
  const svgRef = useRef<SVGSVGElement>(null);
  const setSelectedEmotion = useEmotionStore((s) => s.setSelectedEmotion);
  const selectedEmotion = useEmotionStore((s) => s.selectedEmotion);

  useEffect(() => {
    if (!svgRef.current || !pcaResult) return;

    const { emotions, pc1, pc2, variance_explained } = pcaResult;

    const width = 500;
    const height = 340;
    const margin = { top: 30, right: 30, bottom: 40, left: 50 };

    const xExt = d3.extent(pc1) as [number, number];
    const yExt = d3.extent(pc2) as [number, number];
    const xPad = (xExt[1] - xExt[0]) * 0.15 || 1;
    const yPad = (yExt[1] - yExt[0]) * 0.15 || 1;

    const x = d3.scaleLinear()
      .domain([xExt[0] - xPad, xExt[1] + xPad])
      .range([margin.left, width - margin.right]);
    const y = d3.scaleLinear()
      .domain([yExt[0] - yPad, yExt[1] + yPad])
      .range([height - margin.bottom, margin.top]);

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    // Zero lines
    svg.append('line')
      .attr('x1', x(0)).attr('x2', x(0))
      .attr('y1', margin.top).attr('y2', height - margin.bottom)
      .attr('stroke', 'var(--border)').attr('stroke-dasharray', '3,3');
    svg.append('line')
      .attr('x1', margin.left).attr('x2', width - margin.right)
      .attr('y1', y(0)).attr('y2', y(0))
      .attr('stroke', 'var(--border)').attr('stroke-dasharray', '3,3');

    // Axis labels
    svg.append('text')
      .attr('x', width / 2).attr('y', height - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-secondary)').attr('font-size', 10)
      .text(`PC1 — Valence (${(variance_explained[0] * 100).toFixed(0)}%)`);
    svg.append('text')
      .attr('x', 14).attr('y', height / 2)
      .attr('text-anchor', 'middle')
      .attr('transform', `rotate(-90, 14, ${height / 2})`)
      .attr('fill', 'var(--text-secondary)').attr('font-size', 10)
      .text(`PC2 — Arousal (${(variance_explained[1] * 100).toFixed(0)}%)`);

    // Points + labels
    emotions.forEach((e, i) => {
      const isSelected = e === selectedEmotion;
      const cx = x(pc1[i]);
      const cy = y(pc2[i]);

      svg.append('circle')
        .attr('cx', cx).attr('cy', cy).attr('r', isSelected ? 6 : 4)
        .attr('fill', isSelected ? ACCENT : '#6b7280')
        .attr('stroke', isSelected ? '#fff' : 'none')
        .attr('stroke-width', isSelected ? 1.5 : 0)
        .style('cursor', 'pointer')
        .on('click', () => setSelectedEmotion(e));

      svg.append('text')
        .attr('x', cx + 7).attr('y', cy + 3)
        .attr('fill', isSelected ? ACCENT : 'var(--text-secondary)')
        .attr('font-size', isSelected ? 10 : 8)
        .attr('font-weight', isSelected ? 'bold' : 'normal')
        .style('cursor', 'pointer')
        .text(e)
        .on('click', () => setSelectedEmotion(e));
    });
  }, [pcaResult, selectedEmotion, setSelectedEmotion]);

  if (!pcaResult) return null;

  return (
    <div>
      <span style={{ fontSize: 'var(--font-size-sm)', color: ACCENT, fontWeight: 'bold', letterSpacing: '0.5px' }}>
        EMOTION SPACE (PCA)
      </span>
      <div style={{ overflowX: 'auto', marginTop: 8 }}>
        <svg ref={svgRef} style={{ display: 'block' }} />
      </div>
    </div>
  );
}
