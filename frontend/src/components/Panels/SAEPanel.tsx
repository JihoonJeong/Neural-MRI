import { useCallback, useEffect, useRef } from 'react';
import { useSAEStore } from '../../store/useSAEStore';
import { useScanStore } from '../../store/useScanStore';
import { useModelStore } from '../../store/useModelStore';
import { useLocaleStore } from '../../store/useLocaleStore';
import type { TranslationKey } from '../../i18n/translations';
import * as d3 from 'd3';

export function SAEPanel() {
  const t = useLocaleStore((s) => s.t);
  const isModelLoaded = useModelStore((s) => s.modelInfo !== null);
  const prompt = useScanStore((s) => s.prompt);
  const selectedTokenIdx = useScanStore((s) => s.selectedTokenIdx);

  const {
    saeInfo,
    saeData,
    isScanning,
    error,
    selectedLayer,
    selectedFeatureIdx,
    runScan,
    setSelectedLayer,
    setSelectedFeatureIdx,
  } = useSAEStore();

  const available = saeInfo?.available ?? false;

  const handleDecode = useCallback(() => {
    if (prompt) runScan(prompt);
  }, [prompt, runScan]);

  // Current token's top features
  const currentTokenFeatures =
    saeData && selectedTokenIdx < saeData.token_features.length
      ? saeData.token_features[selectedTokenIdx]
      : null;

  return (
    <div className="px-3 py-2">
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <span
          className="tracking-wide font-bold"
          style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
        >
          {t('sae.title' as TranslationKey)}
        </span>
        {available && (
          <button
            onClick={handleDecode}
            disabled={isScanning || !isModelLoaded || !prompt}
            style={{
              background: isScanning ? 'rgba(255,255,255,0.04)' : 'rgba(170,136,255,0.1)',
              border: '1px solid rgba(170,136,255,0.25)',
              color: isScanning ? 'var(--text-secondary)' : '#aa88ff',
              padding: '2px 8px',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              cursor: isScanning || !isModelLoaded || !prompt ? 'not-allowed' : 'pointer',
              borderRadius: 3,
            }}
          >
            {isScanning
              ? t('sae.scanning' as TranslationKey)
              : t('sae.scan' as TranslationKey)}
          </button>
        )}
      </div>

      {/* Not available */}
      {!available && (
        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
          {t('sae.noSae' as TranslationKey)}
        </div>
      )}

      {/* Layer selector */}
      {available && saeInfo && (
        <div className="flex items-center gap-2 mb-2">
          <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
            {t('sae.layer' as TranslationKey)}
          </span>
          <select
            value={selectedLayer}
            onChange={(e) => setSelectedLayer(Number(e.target.value))}
            style={{
              background: 'var(--bg-primary)',
              color: 'var(--text-data)',
              border: '1px solid var(--border)',
              borderRadius: 3,
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              padding: '1px 4px',
              flex: 1,
            }}
          >
            {saeInfo.layers.map((l) => (
              <option key={l} value={l}>
                {l}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ fontSize: 'var(--font-size-xs)', color: '#ff4466', marginBottom: 4 }}>
          {error.length > 60 ? error.slice(0, 60) + '...' : error}
        </div>
      )}

      {/* Summary stats */}
      {saeData && (
        <div
          className="mb-2"
          style={{
            fontSize: 'var(--font-size-xs)',
            color: 'var(--text-secondary)',
            display: 'flex',
            gap: 8,
          }}
        >
          <span>
            {t('sae.sparsity' as TranslationKey)}:{' '}
            <span style={{ color: '#aa88ff' }}>{(saeData.sparsity * 100).toFixed(1)}%</span>
          </span>
          <span>
            {t('sae.reconstruction' as TranslationKey)}:{' '}
            <span style={{ color: '#aa88ff' }}>{saeData.reconstruction_loss.toFixed(4)}</span>
          </span>
        </div>
      )}

      {/* Feature list for current token */}
      {saeData && currentTokenFeatures && (
        <>
          <div
            className="mb-1"
            style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
          >
            {t('sae.features' as TranslationKey)}
            <span style={{ color: 'var(--text-data)', marginLeft: 4 }}>
              [{currentTokenFeatures.token_str.trim() || ' '}]
            </span>
          </div>
          <div style={{ maxHeight: 140, overflowY: 'auto', marginBottom: 4 }}>
            {currentTokenFeatures.top_features.map((f) => (
              <div
                key={f.feature_idx}
                className="flex items-center gap-1 py-0.5"
                style={{
                  fontSize: 'var(--font-size-xs)',
                  cursor: 'pointer',
                  background:
                    selectedFeatureIdx === f.feature_idx
                      ? 'rgba(170,136,255,0.1)'
                      : 'transparent',
                  borderRadius: 2,
                  padding: '1px 2px',
                }}
                onClick={() => setSelectedFeatureIdx(f.feature_idx)}
              >
                {/* Feature index */}
                <span
                  style={{
                    color: '#aa88ff',
                    width: 44,
                    flexShrink: 0,
                    textAlign: 'right',
                  }}
                >
                  #{f.feature_idx}
                </span>
                {/* Activation bar */}
                <div
                  style={{
                    flex: 1,
                    height: 6,
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: 3,
                    overflow: 'hidden',
                  }}
                >
                  <div
                    style={{
                      width: `${Math.min(f.activation_normalized * 100, 100)}%`,
                      height: '100%',
                      background: `linear-gradient(90deg, #7c3aed, #06b6d4)`,
                      borderRadius: 3,
                    }}
                  />
                </div>
                {/* Value */}
                <span
                  style={{
                    color: 'var(--text-data)',
                    width: 36,
                    flexShrink: 0,
                    textAlign: 'right',
                  }}
                >
                  {f.activation.toFixed(1)}
                </span>
                {/* Neuronpedia link */}
                {f.neuronpedia_url && (
                  <a
                    href={f.neuronpedia_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                    style={{
                      color: '#06b6d4',
                      textDecoration: 'none',
                      flexShrink: 0,
                      fontSize: 10,
                    }}
                    title={t('sae.neuronpedia' as TranslationKey)}
                  >
                    NP
                  </a>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* Heatmap */}
      {saeData && saeData.heatmap_feature_indices.length > 0 && (
        <>
          <div
            className="mb-1"
            style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
          >
            {t('sae.heatmap' as TranslationKey)}
          </div>
          <SAEHeatmap
            tokens={saeData.tokens}
            featureIndices={saeData.heatmap_feature_indices}
            values={saeData.heatmap_values}
            selectedTokenIdx={selectedTokenIdx}
            selectedFeatureIdx={selectedFeatureIdx}
            onSelectFeature={setSelectedFeatureIdx}
          />
        </>
      )}
    </div>
  );
}

// ─── SAE Heatmap (D3) ──────────────────────────────────────────

interface HeatmapProps {
  tokens: string[];
  featureIndices: number[];
  values: number[][]; // [n_tokens][n_features]
  selectedTokenIdx: number;
  selectedFeatureIdx: number | null;
  onSelectFeature: (idx: number | null) => void;
}

function SAEHeatmap({
  tokens,
  featureIndices,
  values,
  selectedTokenIdx,
  selectedFeatureIdx,
  onSelectFeature,
}: HeatmapProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || featureIndices.length === 0) return;

    const nTokens = tokens.length;
    const nFeatures = Math.min(featureIndices.length, 30); // cap columns for readability
    const displayIndices = featureIndices.slice(0, nFeatures);

    const cellW = 6;
    const cellH = 10;
    const marginLeft = 40;
    const marginTop = 24;
    const width = marginLeft + nFeatures * cellW;
    const height = marginTop + nTokens * cellH;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    // Color scale: viridis-like (purple → cyan → yellow)
    const colorScale = d3
      .scaleSequential(d3.interpolateViridis)
      .domain([0, 1]);

    // Token labels (Y axis)
    svg
      .selectAll('.tok-label')
      .data(tokens)
      .enter()
      .append('text')
      .attr('x', marginLeft - 2)
      .attr('y', (_, i) => marginTop + i * cellH + cellH - 1)
      .attr('text-anchor', 'end')
      .attr('font-size', 7)
      .attr('font-family', 'var(--font-primary)')
      .attr('fill', (_, i) =>
        i === selectedTokenIdx ? '#aa88ff' : 'var(--text-secondary)'
      )
      .text((t) => {
        const s = t.trim() || ' ';
        return s.length > 5 ? s.slice(0, 5) : s;
      });

    // Feature index labels (X axis, rotated)
    svg
      .selectAll('.feat-label')
      .data(displayIndices)
      .enter()
      .append('text')
      .attr(
        'transform',
        (_, i) =>
          `translate(${marginLeft + i * cellW + cellW / 2}, ${marginTop - 2}) rotate(-60)`
      )
      .attr('text-anchor', 'end')
      .attr('font-size', 6)
      .attr('font-family', 'var(--font-primary)')
      .attr('fill', (d) =>
        d === selectedFeatureIdx ? '#06b6d4' : 'var(--text-secondary)'
      )
      .text((d) => d);

    // Cells
    for (let ti = 0; ti < nTokens; ti++) {
      for (let fi = 0; fi < nFeatures; fi++) {
        const val = values[ti]?.[fi] ?? 0;
        svg
          .append('rect')
          .attr('x', marginLeft + fi * cellW)
          .attr('y', marginTop + ti * cellH)
          .attr('width', cellW - 0.5)
          .attr('height', cellH - 0.5)
          .attr('fill', val > 0 ? colorScale(val) : 'rgba(255,255,255,0.02)')
          .attr('rx', 0.5)
          .attr('stroke', ti === selectedTokenIdx ? 'rgba(170,136,255,0.4)' : 'none')
          .attr('stroke-width', ti === selectedTokenIdx ? 0.5 : 0)
          .style('cursor', 'pointer')
          .on('click', () => onSelectFeature(displayIndices[fi]))
          .append('title')
          .text(
            `${tokens[ti].trim() || ' '} | #${displayIndices[fi]} | ${val.toFixed(3)}`
          );
      }
    }
  }, [tokens, featureIndices, values, selectedTokenIdx, selectedFeatureIdx, onSelectFeature]);

  return (
    <div style={{ overflowX: 'auto', overflowY: 'auto', maxHeight: 160 }}>
      <svg ref={svgRef} />
    </div>
  );
}
