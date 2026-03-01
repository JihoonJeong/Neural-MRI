import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import { useScanStore } from '../../store/useScanStore';
import { useLocaleStore } from '../../store/useLocaleStore';
import type { TranslationKey } from '../../i18n/translations';

export function AttentionPanel() {
  const circuitData = useScanStore((s) => s.circuitData);
  const t = useLocaleStore((s) => s.t);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);

  const heads = useMemo(() => {
    if (!circuitData) return [];
    return circuitData.attention_heads;
  }, [circuitData]);

  const layers = useMemo(() => {
    const set = new Set(heads.map((h) => h.layer_idx));
    return Array.from(set).sort((a, b) => a - b);
  }, [heads]);

  const headsInLayer = useMemo(() => {
    return heads
      .filter((h) => h.layer_idx === selectedLayer)
      .sort((a, b) => a.head_idx - b.head_idx);
  }, [heads, selectedLayer]);

  const currentHead = useMemo(() => {
    return headsInLayer.find((h) => h.head_idx === selectedHead) ?? null;
  }, [headsInLayer, selectedHead]);

  // Reset head when layer changes
  useEffect(() => {
    setSelectedHead(0);
  }, [selectedLayer]);

  if (!circuitData || heads.length === 0) {
    return (
      <div className="px-3 py-2" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
        <div className="tracking-wide mb-1" style={{ color: 'var(--accent-active)' }}>
          {t('attention.title' as TranslationKey)}
        </div>
        {t('attention.needDTI' as TranslationKey)}
      </div>
    );
  }

  const tokens = circuitData.tokens;

  return (
    <div className="px-3 py-2" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-primary)' }}>
      <div className="tracking-wide mb-2" style={{ color: 'var(--accent-active)' }}>
        {t('attention.title' as TranslationKey)}
      </div>

      {/* Selectors */}
      <div className="flex gap-2 mb-2">
        <label className="flex items-center gap-1">
          <span style={{ color: 'var(--text-secondary)' }}>L:</span>
          <select
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(Number(e.target.value))}
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              padding: '1px 4px',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
            }}
          >
            {layers.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </label>
        <label className="flex items-center gap-1">
          <span style={{ color: 'var(--text-secondary)' }}>H:</span>
          <select
            value={selectedHead}
            onChange={(e) => setSelectedHead(Number(e.target.value))}
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              padding: '1px 4px',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
            }}
          >
            {headsInLayer.map((h) => (
              <option key={h.head_idx} value={h.head_idx}>{h.head_idx}</option>
            ))}
          </select>
        </label>
      </div>

      {/* Heatmap */}
      {currentHead && <AttentionHeatmap pattern={currentHead.pattern} tokens={tokens} />}
    </div>
  );
}

function AttentionHeatmap({ pattern, tokens }: { pattern: number[][]; tokens: string[] }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || pattern.length === 0) return;

    const n = Math.min(pattern.length, tokens.length);
    const cellSize = Math.min(14, Math.floor(200 / n));
    const labelW = 35;
    const labelH = 35;
    const margin = { top: labelH, left: labelW, right: 4, bottom: 4 };
    const w = margin.left + n * cellSize + margin.right;
    const h = margin.top + n * cellSize + margin.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', w).attr('height', h);

    const color = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    // Column labels (destination tokens)
    for (let j = 0; j < n; j++) {
      const tok = tokens[j].replace(/\s/g, '\u2423');
      svg.append('text')
        .attr('x', margin.left + j * cellSize + cellSize / 2)
        .attr('y', margin.top - 4)
        .attr('text-anchor', 'end')
        .attr('transform', `rotate(-45, ${margin.left + j * cellSize + cellSize / 2}, ${margin.top - 4})`)
        .attr('fill', 'var(--text-secondary)')
        .attr('font-size', 7)
        .text(tok.length > 5 ? tok.slice(0, 5) : tok);
    }

    // Row labels (source tokens) and cells
    for (let i = 0; i < n; i++) {
      const tok = tokens[i].replace(/\s/g, '\u2423');
      svg.append('text')
        .attr('x', margin.left - 3)
        .attr('y', margin.top + i * cellSize + cellSize / 2 + 3)
        .attr('text-anchor', 'end')
        .attr('fill', 'var(--text-secondary)')
        .attr('font-size', 7)
        .text(tok.length > 5 ? tok.slice(0, 5) : tok);

      for (let j = 0; j < n; j++) {
        const val = pattern[i]?.[j] ?? 0;
        svg.append('rect')
          .attr('x', margin.left + j * cellSize)
          .attr('y', margin.top + i * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', color(val))
          .attr('rx', 1)
          .append('title')
          .text(`${tokens[i]}→${tokens[j]}: ${val.toFixed(3)}`);
      }
    }
  }, [pattern, tokens]);

  return <svg ref={svgRef} style={{ maxWidth: '100%', overflow: 'visible' }} />;
}
