import { useEffect, useRef, useMemo, useCallback, useState } from 'react';
import * as d3 from 'd3';
import { useScanStore } from '../../store/useScanStore';
import { useModelStore } from '../../store/useModelStore';
import { usePerturbStore } from '../../store/usePerturbStore';
import { useAnimationFrame } from '../../hooks/useAnimationFrame';
import { getNodeColor, getEdgeStyle } from './colorMaps';
import { buildBrainLayout } from './brainLayout';
import { buildNetworkLayout } from './networkLayout';
import { buildRadialLayout } from './radialLayout';
import type { ActivationData, AnomalyData, CircuitData, LayerStructure, ConnectionInfo, LayerWeightStats, StructuralData, WeightData } from '../../types/scan';
import type { ScanMode } from '../../types/model';

export interface CanvasNode {
  id: string;
  x: number;
  y: number;
  radius: number;
  layerType: string;
  layerIndex: number | null;
  paramCount: number;
  ratio: number; // normalized value for coloring (0-1)
}

export interface CanvasEdge {
  fromId: string;
  toId: string;
  strength: number;
  isPathway?: boolean; // DTI: true for important circuit paths
}

const DEFAULT_WIDTH = 560;
const DEFAULT_HEIGHT = 620;
const PADDING_Y = 40;
const PADDING_X = 60;

export interface ScanCanvasProps {
  width?: number;
  height?: number;
  dataOverride?: {
    mode: ScanMode;
    structuralData: StructuralData | null;
    weightData: WeightData | null;
    activationData: ActivationData | null;
    circuitData: CircuitData | null;
    anomalyData: AnomalyData | null;
    selectedTokenIdx: number;
    selectedLayerId?: string | null;
  };
  label?: string;
  onLayerSelect?: (id: string | null) => void;
  hideEmptyStates?: boolean;
}

function buildLayout(
  layers: LayerStructure[],
  connections: ConnectionInfo[],
  canvasWidth: number,
  canvasHeight: number,
  weightLayers?: LayerWeightStats[],
): { nodes: CanvasNode[]; edges: CanvasEdge[] } {
  const maxParam = Math.max(...layers.map((l) => l.param_count), 1);

  // Build weight lookup: group by layer_id, take average l2_norm
  const weightMap = new Map<string, number>();
  if (weightLayers) {
    const grouped = new Map<string, number[]>();
    for (const w of weightLayers) {
      const arr = grouped.get(w.layer_id) ?? [];
      arr.push(w.l2_norm);
      grouped.set(w.layer_id, arr);
    }
    const allNorms = weightLayers.map((w) => w.l2_norm);
    const maxNorm = Math.max(...allNorms, 1);
    for (const [id, norms] of grouped) {
      const avg = norms.reduce((a, b) => a + b, 0) / norms.length;
      weightMap.set(id, avg / maxNorm);
    }
  }

  const yStep = (canvasHeight - PADDING_Y * 2) / Math.max(layers.length - 1, 1);

  const nodes: CanvasNode[] = layers.map((layer, i) => {
    const paramRatio = layer.param_count / maxParam;
    const baseRadius = 4 + paramRatio * 8;

    // For T2 mode, look up weight ratio
    const wRatio = weightMap.get(layer.layer_id) ?? paramRatio;

    return {
      id: layer.layer_id,
      x: canvasWidth / 2,
      y: PADDING_Y + i * yStep,
      radius: baseRadius,
      layerType: layer.layer_type,
      layerIndex: layer.layer_index,
      paramCount: layer.param_count,
      ratio: wRatio,
    };
  });

  // Spread nodes horizontally by type within the same block index
  const spread = Math.min(50, canvasWidth * 0.09);
  for (const node of nodes) {
    if (node.layerType === 'attention') {
      node.x = canvasWidth / 2 - spread;
    } else if (node.layerType === 'mlp') {
      node.x = canvasWidth / 2 + spread;
    }
  }

  const nodeMap = new Map(nodes.map((n) => [n.id, n]));

  const edges: CanvasEdge[] = connections
    .filter((c) => nodeMap.has(c.from_id) && nodeMap.has(c.to_id))
    .map((c) => {
      const from = nodeMap.get(c.from_id)!;
      const to = nodeMap.get(c.to_id)!;
      return {
        fromId: c.from_id,
        toId: c.to_id,
        strength: (from.ratio + to.ratio) / 2,
      };
    });

  return { nodes, edges };
}

/** Format node label for tooltip */
function nodeLabel(node: CanvasNode): string {
  if (node.layerType === 'embedding') return 'Embed';
  if (node.layerType === 'output') return 'Output';
  if (node.layerType === 'attention') return `Attn ${node.layerIndex}`;
  if (node.layerType === 'mlp') return `MLP ${node.layerIndex}`;
  return node.id;
}

export function ScanCanvas(props: ScanCanvasProps = {}) {
  const canvasWidth = props.width ?? DEFAULT_WIDTH;
  const canvasHeight = props.height ?? DEFAULT_HEIGHT;

  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const nodesRef = useRef<d3.Selection<SVGGElement, CanvasNode, SVGGElement, unknown> | null>(null);
  const edgesRef = useRef<d3.Selection<SVGPathElement, CanvasEdge, SVGGElement, unknown> | null>(null);
  const glowsRef = useRef<d3.Selection<SVGCircleElement, CanvasNode, SVGGElement, unknown> | null>(null);
  const zoomRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  // Read from store (used when no dataOverride)
  const storeData = useScanStore();
  const modelInfo = useModelStore((s) => s.modelInfo);
  const isModelLoading = useModelStore((s) => s.isLoading);
  const isRescanAnimating = usePerturbStore((s) => s.isRescanAnimating);

  // Merge: dataOverride wins if provided
  const mode = props.dataOverride?.mode ?? storeData.mode;
  const structuralData = props.dataOverride?.structuralData ?? storeData.structuralData;
  const weightData = props.dataOverride?.weightData ?? storeData.weightData;
  const activationData = props.dataOverride?.activationData ?? storeData.activationData;
  const circuitData = props.dataOverride?.circuitData ?? storeData.circuitData;
  const anomalyData = props.dataOverride?.anomalyData ?? storeData.anomalyData;
  const selectedTokenIdx = props.dataOverride?.selectedTokenIdx ?? storeData.selectedTokenIdx;
  const selectedLayerId = props.dataOverride?.selectedLayerId ?? storeData.selectedLayerId;
  const handleSelectLayer = props.onLayerSelect ?? storeData.selectLayer;
  const layoutMode = storeData.layoutMode;
  const prevLayoutRef = useRef(layoutMode);

  // --- Crossfade state ---
  const prevModeRef = useRef<ScanMode>(mode);
  const [fadeOpacity, setFadeOpacity] = useState(1);

  useEffect(() => {
    if (prevModeRef.current !== mode) {
      prevModeRef.current = mode;
      setFadeOpacity(0);
      const timer = setTimeout(() => setFadeOpacity(1), 30);
      return () => clearTimeout(timer);
    }
  }, [mode]);

  // --- Tooltip state ---
  const [tooltip, setTooltip] = useState<{ x: number; y: number; node: CanvasNode } | null>(null);

  // Build base layout from structural data
  const { nodes: baseNodes, edges: baseEdges, brainPath } = useMemo(() => {
    if (!structuralData) return { nodes: [], edges: [], brainPath: '' };
    if (layoutMode === 'brain') {
      return buildBrainLayout(
        structuralData.layers,
        structuralData.connections,
        canvasWidth,
        canvasHeight,
        weightData?.layers,
      );
    }
    if (layoutMode === 'network') {
      const result = buildNetworkLayout(
        structuralData.layers,
        structuralData.connections,
        canvasWidth,
        canvasHeight,
        weightData?.layers,
      );
      return { ...result, brainPath: '' };
    }
    if (layoutMode === 'radial') {
      const result = buildRadialLayout(
        structuralData.layers,
        structuralData.connections,
        canvasWidth,
        canvasHeight,
        weightData?.layers,
      );
      return { ...result, brainPath: '' };
    }
    const result = buildLayout(
      structuralData.layers,
      structuralData.connections,
      canvasWidth,
      canvasHeight,
      weightData?.layers,
    );
    return { ...result, brainPath: '' };
  }, [structuralData, weightData, canvasWidth, canvasHeight, layoutMode]);

  // Compute final nodes/edges with mode-specific overlays
  const { nodes, edges } = useMemo(() => {
    if (baseNodes.length === 0) return { nodes: baseNodes, edges: baseEdges };

    const overlayNodes = baseNodes.map((n) => ({ ...n }));

    if (mode === 'fMRI' && activationData) {
      const actMap = new Map(activationData.layers.map((l) => [l.layer_id, l]));
      for (const node of overlayNodes) {
        const layerAct = actMap.get(node.id);
        if (layerAct && selectedTokenIdx < layerAct.activations.length) {
          node.ratio = layerAct.activations[selectedTokenIdx];
        }
      }
    } else if (mode === 'DTI' && circuitData) {
      const compMap = new Map(circuitData.components.map((c) => [c.layer_id, c]));
      for (const node of overlayNodes) {
        const comp = compMap.get(node.id);
        if (comp) {
          node.ratio = comp.importance;
        }
      }
    } else if (mode === 'FLAIR' && anomalyData) {
      // FLAIR layers use "blocks.{i}" IDs; map to both attn and mlp nodes
      const anomMap = new Map(anomalyData.layers.map((l) => [l.layer_id, l]));
      for (const node of overlayNodes) {
        const blockMatch = node.id.match(/^blocks\.(\d+)/);
        if (blockMatch) {
          const blockId = `blocks.${blockMatch[1]}`;
          const anom = anomMap.get(blockId);
          if (anom && selectedTokenIdx < anom.anomaly_scores.length) {
            node.ratio = anom.anomaly_scores[selectedTokenIdx];
          }
        } else {
          node.ratio = 0; // embed/unembed: no anomaly data
        }
      }
    }

    // Build edges: for DTI, use circuit connections with pathway info
    let overlayEdges = baseEdges;
    if (mode === 'DTI' && circuitData) {
      const nodeMap = new Map(overlayNodes.map((n) => [n.id, n]));
      overlayEdges = circuitData.connections
        .filter((c) => nodeMap.has(c.from_id) && nodeMap.has(c.to_id))
        .map((c) => ({
          fromId: c.from_id,
          toId: c.to_id,
          strength: c.strength,
          isPathway: c.is_pathway,
        }));
    } else if (mode === 'FLAIR') {
      // Recompute edge strength from anomaly-overlaid node ratios
      const nodeMap = new Map(overlayNodes.map((n) => [n.id, n]));
      overlayEdges = baseEdges.map((e) => ({
        ...e,
        strength: ((nodeMap.get(e.fromId)?.ratio ?? 0) + (nodeMap.get(e.toId)?.ratio ?? 0)) / 2,
      }));
    }

    return { nodes: overlayNodes, edges: overlayEdges };
  }, [baseNodes, baseEdges, mode, activationData, circuitData, anomalyData, selectedTokenIdx]);

  // D3 structure rendering (only when nodes/edges change shape)
  useEffect(() => {
    if (!svgRef.current || nodes.length === 0) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Reset zoom on layout mode change
    if (prevLayoutRef.current !== layoutMode) {
      prevLayoutRef.current = layoutMode;
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (svgRef.current as any).__zoom = d3.zoomIdentity;
    }

    const defs = svg.append('defs');

    // Vignette gradient
    const radGrad = defs
      .append('radialGradient')
      .attr('id', 'vignette')
      .attr('cx', '50%')
      .attr('cy', '50%')
      .attr('r', '70%');
    radGrad.append('stop').attr('offset', '60%').attr('stop-color', 'transparent');
    radGrad.append('stop').attr('offset', '100%').attr('stop-color', 'rgba(0,0,0,0.6)');

    // Glow filter
    const glowFilter = defs.append('filter').attr('id', 'glow');
    glowFilter.append('feGaussianBlur').attr('stdDeviation', '4').attr('result', 'blur');
    glowFilter.append('feMerge').selectAll('feMergeNode')
      .data(['blur', 'SourceGraphic'])
      .join('feMergeNode')
      .attr('in', (d) => d);

    // fMRI glow filter (larger, more intense)
    const fmriGlow = defs.append('filter').attr('id', 'fmri-glow');
    fmriGlow.append('feGaussianBlur').attr('stdDeviation', '8').attr('result', 'blur');
    fmriGlow.append('feMerge').selectAll('feMergeNode')
      .data(['blur', 'SourceGraphic'])
      .join('feMergeNode')
      .attr('in', (d) => d);

    // --- Zoom group: all zoomable content goes here ---
    const g = svg.append('g').attr('class', 'zoom-group');

    // Set up zoom behavior (create once, reuse across re-renders)
    if (!zoomRef.current) {
      zoomRef.current = d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.3, 4])
        .on('zoom', (event) => {
          d3.select(svgRef.current!)
            .select<SVGGElement>('g.zoom-group')
            .attr('transform', event.transform.toString());
        });
    }
    svg.call(zoomRef.current);
    svg.on('dblclick.zoom', null); // replace default double-click zoom-in

    // Restore current zoom transform (persists across re-renders)
    const currentTransform = d3.zoomTransform(svgRef.current);
    g.attr('transform', currentTransform.toString());

    // Double-click to reset zoom
    svg.on('dblclick', () => {
      if (!zoomRef.current) return;
      svg.transition().duration(300).call(zoomRef.current.transform, d3.zoomIdentity);
    });

    // Brain silhouette (only in brain layout mode)
    if (brainPath) {
      g.append('path')
        .attr('d', brainPath)
        .attr('fill', 'rgba(100, 170, 136, 0.04)')
        .attr('stroke', 'rgba(100, 170, 136, 0.10)')
        .attr('stroke-width', 1)
        .attr('pointer-events', 'none');
    }

    const nodeMap = new Map(nodes.map((n) => [n.id, n]));

    // Edges — use <path> for DTI (curved), <path> with straight line for others
    edgesRef.current = g
      .selectAll<SVGPathElement, CanvasEdge>('path.edge')
      .data(edges)
      .join('path')
      .attr('class', 'edge')
      .attr('d', (d) => {
        const from = nodeMap.get(d.fromId)!;
        const to = nodeMap.get(d.toId)!;
        if (mode === 'DTI' && d.isPathway) {
          // Quadratic bezier curve for pathway edges
          const midX = (from.x + to.x) / 2 + (from.x === to.x ? 30 : 0);
          const midY = (from.y + to.y) / 2;
          return `M ${from.x} ${from.y} Q ${midX} ${midY} ${to.x} ${to.y}`;
        }
        return `M ${from.x} ${from.y} L ${to.x} ${to.y}`;
      })
      .attr('fill', 'none')
      .attr('stroke', (d) => getEdgeStyle(mode, d.strength).stroke)
      .attr('stroke-width', (d) => {
        const base = getEdgeStyle(mode, d.strength).strokeWidth;
        return d.isPathway ? base * 1.5 : base;
      })
      .attr('opacity', (d) => getEdgeStyle(mode, d.strength).opacity);

    // DTI: add stroke-dasharray for pathway flow animation
    if (mode === 'DTI') {
      edgesRef.current
        .filter((d) => !!d.isPathway)
        .attr('stroke-dasharray', '8 4');
    }

    // fMRI glow circles (behind nodes, for high-activation nodes)
    glowsRef.current = g
      .selectAll<SVGCircleElement, CanvasNode>('circle.glow')
      .data(nodes)
      .join('circle')
      .attr('class', 'glow')
      .attr('cx', (d) => d.x)
      .attr('cy', (d) => d.y)
      .attr('r', (d) => d.radius + 10)
      .attr('fill', 'none')
      .attr('stroke', (d) => getNodeColor(mode, d.ratio))
      .attr('stroke-width', 0)
      .attr('opacity', 0)
      .attr('filter', 'url(#fmri-glow)');

    // Node groups
    nodesRef.current = g
      .selectAll<SVGGElement, CanvasNode>('g.node')
      .data(nodes)
      .join('g')
      .attr('class', 'node')
      .attr('transform', (d) => `translate(${d.x},${d.y})`)
      .style('cursor', 'pointer')
      .on('click', (_event, d) => {
        handleSelectLayer(selectedLayerId === d.id ? null : d.id);
      })
      .on('mouseenter', (_event, d) => {
        const svgEl = svgRef.current;
        const containerEl = containerRef.current;
        if (!svgEl || !containerEl) return;
        const containerRect = containerEl.getBoundingClientRect();
        const svgRect = svgEl.getBoundingClientRect();
        // Account for zoom transform
        const transform = d3.zoomTransform(svgEl);
        const screenX = d.x * transform.k + transform.x;
        const screenY = d.y * transform.k + transform.y;
        const x = svgRect.left - containerRect.left + screenX + d.radius * transform.k + 12;
        const y = svgRect.top - containerRect.top + screenY - 10;
        setTooltip({ x, y, node: d });
      })
      .on('mouseleave', () => {
        setTooltip(null);
      });

    // Node circles
    nodesRef.current
      .append('circle')
      .attr('class', 'node-circle')
      .attr('r', (d) => d.radius)
      .attr('fill', (d) => getNodeColor(mode, d.ratio))
      .attr('stroke', (d) => (d.id === selectedLayerId ? '#00ffaa' : 'none'))
      .attr('stroke-width', (d) => (d.id === selectedLayerId ? 2 : 0));

    // Selected node glow rings
    nodesRef.current
      .filter((d) => d.id === selectedLayerId)
      .append('circle')
      .attr('r', (d) => d.radius + 8)
      .attr('fill', 'none')
      .attr('stroke', '#00ffaa')
      .attr('stroke-width', 1)
      .attr('opacity', 0.4);

    // Left-side labels (only in vertical stack layout)
    if (canvasWidth >= 400 && layoutMode === 'vertical') {
      g
        .selectAll<SVGTextElement, CanvasNode>('text.label')
        .data(nodes)
        .join('text')
        .attr('class', 'label')
        .attr('x', PADDING_X - 10)
        .attr('y', (d) => d.y + 4)
        .attr('text-anchor', 'end')
        .attr('fill', 'rgba(100,180,140,0.5)')
        .attr('font-size', 'var(--font-size-xs)')
        .attr('font-family', 'var(--font-primary)')
        .text((d) => nodeLabel(d));
    }

    // Vignette overlay (outside zoom group — stays fixed)
    svg
      .append('rect')
      .attr('width', canvasWidth)
      .attr('height', canvasHeight)
      .attr('fill', 'url(#vignette)')
      .attr('pointer-events', 'none');
  }, [nodes, edges, mode, selectedLayerId, handleSelectLayer, canvasWidth, canvasHeight, brainPath, layoutMode]);

  // Animation frame: update node colors, radii, glow, edge dash offset
  const isAnimated = mode === 'fMRI' || mode === 'DTI' || mode === 'FLAIR';
  const rescanStartRef = useRef<number | null>(null);
  const skipRescan = !!props.dataOverride; // no rescan animation for compare canvases

  const animate = useCallback((time: number) => {
    if (!nodesRef.current || !edgesRef.current) return;

    // Rescan flash: 300ms opacity dip when perturbation completes
    if (!skipRescan && isRescanAnimating) {
      if (rescanStartRef.current === null) rescanStartRef.current = time;
      const elapsed = time - rescanStartRef.current;
      const flash = elapsed < 150 ? 1 - elapsed / 150 : (elapsed - 150) / 150;
      const opacity = 0.3 + Math.min(flash, 1) * 0.7;
      nodesRef.current.style('opacity', opacity);
      edgesRef.current.style('opacity', opacity);
      glowsRef.current?.style('opacity', opacity);
      return; // skip mode-specific animation during flash
    } else {
      rescanStartRef.current = null;
      nodesRef.current.style('opacity', 1);
      edgesRef.current.style('opacity', null);
      glowsRef.current?.style('opacity', null);
    }

    if (mode === 'fMRI') {
      // Pulse animation: sin-based radius and color modulation
      nodesRef.current.select<SVGCircleElement>('circle.node-circle')
        .attr('r', (d) => {
          const pulse = Math.sin(time * 0.003 + d.x * 0.01 + d.y * 0.02) * 0.15 + 0.85;
          return d.radius * (0.9 + d.ratio * 0.3 * pulse);
        })
        .attr('fill', (d) => {
          const pulse = Math.sin(time * 0.003 + d.x * 0.01 + d.y * 0.02) * 0.15 + 0.85;
          return getNodeColor('fMRI', Math.min(1, d.ratio * pulse));
        });

      // Glow for high-activation nodes
      glowsRef.current
        ?.attr('stroke-width', (d) => d.ratio > 0.5 ? 3 + Math.sin(time * 0.004) * 2 : 0)
        .attr('opacity', (d) => d.ratio > 0.5 ? d.ratio * 0.4 : 0)
        .attr('stroke', (d) => getNodeColor('fMRI', d.ratio))
        .attr('r', (d) => {
          const pulse = Math.sin(time * 0.003 + d.y * 0.02) * 3;
          return d.radius + 8 + pulse;
        });

      // fMRI edge pulse
      edgesRef.current
        .attr('opacity', (d) => {
          const base = getEdgeStyle('fMRI', d.strength).opacity;
          const pulse = Math.sin(time * 0.002 + d.strength * 5) * 0.1;
          return base + pulse;
        });
    }

    if (mode === 'DTI') {
      // Flow animation: stroke-dashoffset moves along pathway edges
      edgesRef.current
        .filter((d) => !!d.isPathway)
        .attr('stroke-dashoffset', -time * 0.03);

      // Pulse non-pathway edges subtly
      edgesRef.current
        .filter((d) => !d.isPathway)
        .attr('opacity', (d) => {
          return getEdgeStyle('DTI', d.strength).opacity * 0.3;
        });
    }

    if (mode === 'FLAIR') {
      // Anomalous nodes: red pulse + glow
      nodesRef.current.select<SVGCircleElement>('circle.node-circle')
        .attr('r', (d) => {
          if (d.ratio > 0.3) {
            const pulse = Math.sin(time * 0.004 + d.y * 0.02) * 0.2 + 0.8;
            return d.radius * (1 + d.ratio * 0.3 * pulse);
          }
          return d.radius * 0.9;
        })
        .attr('fill', (d) => getNodeColor('FLAIR', d.ratio));

      // Glow for anomalous nodes (ratio > 0.3)
      glowsRef.current
        ?.attr('stroke-width', (d) => d.ratio > 0.3 ? 2 + Math.sin(time * 0.005) * 2 : 0)
        .attr('opacity', (d) => d.ratio > 0.3 ? d.ratio * 0.5 : 0)
        .attr('stroke', (d) => getNodeColor('FLAIR', d.ratio))
        .attr('r', (d) => {
          const pulse = Math.sin(time * 0.004 + d.y * 0.02) * 3;
          return d.radius + 8 + pulse;
        });

      // Edge opacity follows anomaly strength
      edgesRef.current
        .attr('stroke', (d) => getEdgeStyle('FLAIR', d.strength).stroke)
        .attr('opacity', (d) => {
          const base = getEdgeStyle('FLAIR', d.strength).opacity;
          const pulse = Math.sin(time * 0.003 + d.strength * 3) * 0.08;
          return base + pulse;
        });
    }
  }, [mode, isRescanAnimating, skipRescan]);

  useAnimationFrame(animate, isAnimated || (!skipRescan && isRescanAnimating));

  // --- Empty states (skip when used as compare sub-canvas) ---
  if (!props.hideEmptyStates) {
    if (isModelLoading) {
      return (
        <div
          className="flex items-center justify-center"
          style={{ width: canvasWidth, height: canvasHeight, color: 'var(--text-secondary)' }}
        >
          <div className="text-center">
            <div
              style={{
                fontSize: 'var(--font-size-lg)',
                color: 'var(--accent-active)',
                marginBottom: 8,
                animation: 'scan-pulse 1.5s ease-in-out infinite',
              }}
            >
              LOADING MODEL...
            </div>
            <div style={{ fontSize: 'var(--font-size-sm)' }}>Initializing neural pathways</div>
          </div>
        </div>
      );
    }

    if (!modelInfo) {
      return (
        <div
          className="flex items-center justify-center"
          style={{ width: canvasWidth, height: canvasHeight, color: 'var(--text-secondary)' }}
        >
          <div className="text-center">
            <div style={{ fontSize: 'var(--font-size-lg)', color: 'var(--text-primary)', marginBottom: 8 }}>
              NO MODEL LOADED
            </div>
            <div style={{ fontSize: 'var(--font-size-sm)' }}>Select a model to begin scanning</div>
          </div>
        </div>
      );
    }

    if (!structuralData) {
      return (
        <div
          className="flex items-center justify-center"
          style={{ width: canvasWidth, height: canvasHeight, color: 'var(--text-secondary)' }}
        >
          <div className="text-center">
            <div style={{ fontSize: 'var(--font-size-lg)', color: 'var(--text-primary)', marginBottom: 8 }}>
              READY TO SCAN
            </div>
            <div style={{ fontSize: 'var(--font-size-sm)' }}>Press SCAN to begin</div>
          </div>
        </div>
      );
    }
  }

  /** Mode-specific value label for tooltip */
  const modeValueLabel = (node: CanvasNode): string => {
    if (mode === 'fMRI') return `Activation: ${(node.ratio * 100).toFixed(1)}%`;
    if (mode === 'DTI') return `Importance: ${(node.ratio * 100).toFixed(1)}%`;
    if (mode === 'FLAIR') return `Anomaly: ${(node.ratio * 100).toFixed(1)}%`;
    if (mode === 'T2') return `Weight norm: ${(node.ratio * 100).toFixed(1)}%`;
    return `Params: ${(node.paramCount / 1e6).toFixed(1)}M`;
  };

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      {props.label && (
        <div
          style={{
            position: 'absolute',
            top: 8,
            right: 12,
            fontSize: 'var(--font-size-sm)',
            color: 'var(--text-secondary)',
            fontWeight: 'bold',
            letterSpacing: '2px',
            opacity: 0.6,
            zIndex: 5,
          }}
        >
          {props.label}
        </div>
      )}
      <svg
        ref={svgRef}
        width={canvasWidth}
        height={canvasHeight}
        data-nmri-canvas={props.label || 'main'}
        style={{
          display: 'block',
          opacity: fadeOpacity,
          transition: 'opacity 0.3s ease-in-out',
        }}
      />

      {/* Hover tooltip (positioned outside SVG for flexible styling) */}
      {tooltip && (
        <div
          style={{
            position: 'absolute',
            left: tooltip.x,
            top: tooltip.y,
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            padding: '6px 10px',
            fontSize: 'var(--font-size-sm)',
            fontFamily: 'var(--font-primary)',
            color: 'var(--text-data)',
            pointerEvents: 'none',
            zIndex: 10,
            whiteSpace: 'nowrap',
            animation: 'nmri-fade-in 0.15s ease-out',
            boxShadow: '0 2px 8px rgba(0,0,0,0.4)',
          }}
        >
          <div style={{ color: 'var(--text-primary)', marginBottom: 2 }}>
            {nodeLabel(tooltip.node)}
          </div>
          <div style={{ color: 'var(--text-secondary)', fontSize: 'var(--font-size-xs)' }}>
            {tooltip.node.layerType} | {(tooltip.node.paramCount / 1e6).toFixed(1)}M params
          </div>
          <div style={{ color: 'var(--accent-active)', fontSize: 'var(--font-size-xs)', marginTop: 2 }}>
            {modeValueLabel(tooltip.node)}
          </div>
        </div>
      )}
    </div>
  );
}
