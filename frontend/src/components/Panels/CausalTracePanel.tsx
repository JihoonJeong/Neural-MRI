import { useEffect, useRef } from 'react';
import * as d3 from 'd3';
import { useCausalTraceStore } from '../../store/useCausalTraceStore';
import { useScanStore } from '../../store/useScanStore';
import { useLocaleStore } from '../../store/useLocaleStore';
import type { TranslationKey } from '../../i18n/translations';

export function CausalTracePanel() {
  const { result, isTracing, corruptPrompt, setCorruptPrompt, runTrace, clear } = useCausalTraceStore();
  const prompt = useScanStore((s) => s.prompt);
  const t = useLocaleStore((s) => s.t);

  return (
    <div className="px-3 py-2" style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-primary)' }}>
      <div className="mb-2 tracking-wide" style={{ color: 'var(--accent-active)' }}>
        {t('causalTrace.title' as TranslationKey)}
      </div>

      {/* Inputs */}
      <div className="mb-2">
        <div style={{ color: 'var(--text-secondary)', marginBottom: 2 }}>
          {t('causalTrace.clean' as TranslationKey)}: {prompt.length > 25 ? prompt.slice(0, 25) + '...' : prompt || '—'}
        </div>
        <input
          type="text"
          value={corruptPrompt}
          onChange={(e) => setCorruptPrompt(e.target.value)}
          placeholder={t('causalTrace.corruptPlaceholder' as TranslationKey)}
          style={{
            width: '100%',
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            color: 'var(--text-primary)',
            padding: '3px 6px',
            fontSize: 'var(--font-size-xs)',
            fontFamily: 'var(--font-primary)',
          }}
        />
      </div>

      <div className="flex gap-2 mb-2">
        <button
          onClick={runTrace}
          disabled={isTracing || !corruptPrompt || !prompt}
          style={{
            background: isTracing ? 'rgba(255,255,255,0.04)' : 'rgba(0,255,170,0.1)',
            border: '1px solid rgba(0,255,170,0.2)',
            color: isTracing ? 'var(--text-secondary)' : 'var(--accent-active)',
            padding: '2px 8px',
            fontSize: 'var(--font-size-xs)',
            fontFamily: 'var(--font-primary)',
            cursor: isTracing || !corruptPrompt || !prompt ? 'not-allowed' : 'pointer',
            letterSpacing: '1px',
          }}
        >
          {isTracing ? t('causalTrace.tracing' as TranslationKey) : t('causalTrace.trace' as TranslationKey)}
        </button>
        {result && (
          <button
            onClick={clear}
            style={{
              background: 'none',
              border: '1px solid var(--border)',
              color: 'var(--text-secondary)',
              padding: '2px 8px',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
            }}
          >
            Clear
          </button>
        )}
      </div>

      {/* Predictions summary */}
      {result && (
        <div className="mb-2" style={{ color: 'var(--text-secondary)' }}>
          <span style={{ color: 'var(--accent-active)' }}>Clean:</span> "{result.clean_prediction}"
          {' '}
          <span style={{ color: '#ff6464' }}>Corrupt:</span> "{result.corrupt_prediction}"
        </div>
      )}

      {/* Heatmap */}
      {result && <CausalTraceHeatmap />}
    </div>
  );
}

function CausalTraceHeatmap() {
  const svgRef = useRef<SVGSVGElement>(null);
  const result = useCausalTraceStore((s) => s.result);

  useEffect(() => {
    if (!svgRef.current || !result) return;

    const { cells, n_layers } = result;
    const attnCells = cells.filter((c) => c.component_type === 'attn');
    const mlpCells = cells.filter((c) => c.component_type === 'mlp');
    const embedCell = cells.find((c) => c.component_type === 'embed');

    const cellSize = 12;
    const labelW = 30;
    const colLabelH = 14;
    const margin = { top: colLabelH + 4, left: labelW + 4, right: 8, bottom: 4 };
    const cols = 2; // attn, mlp
    const rows = n_layers;
    const w = margin.left + cols * cellSize + margin.right;
    const h = margin.top + (rows + 1) * cellSize + margin.bottom; // +1 for embed

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', w).attr('height', h);

    const color = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    // Column labels
    svg.append('text')
      .attr('x', margin.left + 0.5 * cellSize)
      .attr('y', margin.top - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-secondary)')
      .attr('font-size', 8)
      .text('Attn');
    svg.append('text')
      .attr('x', margin.left + 1.5 * cellSize)
      .attr('y', margin.top - 4)
      .attr('text-anchor', 'middle')
      .attr('fill', 'var(--text-secondary)')
      .attr('font-size', 8)
      .text('MLP');

    // Embed row
    if (embedCell) {
      svg.append('text')
        .attr('x', margin.left - 4)
        .attr('y', margin.top + 0.5 * cellSize + 3)
        .attr('text-anchor', 'end')
        .attr('fill', 'var(--text-secondary)')
        .attr('font-size', 7)
        .text('emb');
      svg.append('rect')
        .attr('x', margin.left)
        .attr('y', margin.top)
        .attr('width', cellSize * 2)
        .attr('height', cellSize)
        .attr('fill', color(embedCell.recovery_score))
        .attr('rx', 1)
        .append('title')
        .text(`embed: ${embedCell.recovery_score.toFixed(3)}`);
    }

    // Layer rows
    for (let layer = 0; layer < n_layers; layer++) {
      const y = margin.top + (layer + 1) * cellSize;

      // Row label
      svg.append('text')
        .attr('x', margin.left - 4)
        .attr('y', y + cellSize * 0.5 + 3)
        .attr('text-anchor', 'end')
        .attr('fill', 'var(--text-secondary)')
        .attr('font-size', 7)
        .text(`L${layer}`);

      // Attn cell
      const attn = attnCells.find((c) => c.layer_idx === layer);
      if (attn) {
        svg.append('rect')
          .attr('x', margin.left)
          .attr('y', y)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', color(attn.recovery_score))
          .attr('rx', 1)
          .append('title')
          .text(`${attn.component}: ${attn.recovery_score.toFixed(3)}`);
      }

      // MLP cell
      const mlp = mlpCells.find((c) => c.layer_idx === layer);
      if (mlp) {
        svg.append('rect')
          .attr('x', margin.left + cellSize)
          .attr('y', y)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', color(mlp.recovery_score))
          .attr('rx', 1)
          .append('title')
          .text(`${mlp.component}: ${mlp.recovery_score.toFixed(3)}`);
      }
    }

    // Find max recovery component
    const maxCell = cells.reduce((a, b) => a.recovery_score > b.recovery_score ? a : b, cells[0]);
    if (maxCell) {
      const maxY = maxCell.component_type === 'embed'
        ? margin.top
        : margin.top + (maxCell.layer_idx + 1) * cellSize;
      const maxX = maxCell.component_type === 'mlp'
        ? margin.left + cellSize
        : margin.left;

      svg.append('rect')
        .attr('x', maxX - 1)
        .attr('y', maxY - 1)
        .attr('width', (maxCell.component_type === 'embed' ? cellSize * 2 : cellSize) + 1)
        .attr('height', cellSize + 1)
        .attr('fill', 'none')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1.5)
        .attr('rx', 2);
    }
  }, [result]);

  return <svg ref={svgRef} />;
}
