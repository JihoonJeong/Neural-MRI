import type { CanvasNode, CanvasEdge } from './ScanCanvas';
import type { LayerStructure, ConnectionInfo, LayerWeightStats } from '../../types/scan';

/**
 * Build a brain-shaped (sagittal view) layout for model layers.
 *
 * Mapping (SVG coords: right = anterior/front, left = posterior/back):
 *   - Embedding → brain stem (bottom center)
 *   - Transformer blocks → distributed along cortex arc
 *     - Early blocks → occipital (back/left)
 *     - Middle blocks → parietal (top)
 *     - Late blocks → frontal (front/right)
 *   - Attention layers → outer ring (cortex)
 *   - MLP layers → inner ring (white matter)
 *   - Output → frontal pole (top-right)
 */

export function buildBrainLayout(
  layers: LayerStructure[],
  connections: ConnectionInfo[],
  canvasWidth: number,
  canvasHeight: number,
  weightLayers?: LayerWeightStats[],
): { nodes: CanvasNode[]; edges: CanvasEdge[]; brainPath: string } {
  const maxParam = Math.max(...layers.map((l) => l.param_count), 1);

  // Weight lookup
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

  // Brain geometry — center in upper portion, leave room for brain stem below
  const cx = canvasWidth * 0.50;
  const cy = canvasHeight * 0.38;
  const rx = canvasWidth * 0.28;
  const ry = canvasHeight * 0.25;
  const innerScale = 0.62; // MLP ring as fraction of outer ring

  // Separate layers by type
  const embedding = layers.filter((l) => l.layer_type === 'embedding');
  const output = layers.filter((l) => l.layer_type === 'output');
  const attention = layers.filter((l) => l.layer_type === 'attention');
  const mlp = layers.filter((l) => l.layer_type === 'mlp');

  const nodes: CanvasNode[] = [];

  const makeNode = (layer: LayerStructure, x: number, y: number): CanvasNode => {
    const paramRatio = layer.param_count / maxParam;
    const baseRadius = 4 + paramRatio * 8;
    const wRatio = weightMap.get(layer.layer_id) ?? paramRatio;
    return {
      id: layer.layer_id,
      x, y,
      radius: baseRadius,
      layerType: layer.layer_type,
      layerIndex: layer.layer_index,
      paramCount: layer.param_count,
      ratio: wRatio,
    };
  };

  // Arc from occipital (back/left, 160°) through parietal (top, 90°) to frontal (front/right, 20°)
  // In standard math angles with SVG y-inversion: positive sin → upward
  // arcSpan is negative → clockwise traversal through the top
  const arcStart = (160 / 180) * Math.PI;
  const arcEnd = (20 / 180) * Math.PI;
  const arcSpan = arcEnd - arcStart; // negative

  // Embedding → brain stem (bottom center)
  for (const layer of embedding) {
    nodes.push(makeNode(layer, cx, cy + ry + canvasHeight * 0.12));
  }

  // Attention → outer cortex ring
  for (let i = 0; i < attention.length; i++) {
    const t = attention.length > 1 ? i / (attention.length - 1) : 0.5;
    const angle = arcStart + t * arcSpan;
    const x = cx + rx * Math.cos(angle);
    const y = cy - ry * Math.sin(angle); // SVG y inverted
    nodes.push(makeNode(attention[i], x, y));
  }

  // MLP → inner ring (white matter)
  for (let i = 0; i < mlp.length; i++) {
    const t = mlp.length > 1 ? i / (mlp.length - 1) : 0.5;
    const angle = arcStart + t * arcSpan;
    const x = cx + rx * innerScale * Math.cos(angle);
    const y = cy - ry * innerScale * Math.sin(angle);
    nodes.push(makeNode(mlp[i], x, y));
  }

  // Output → frontal pole (top-right)
  for (const layer of output) {
    nodes.push(makeNode(layer, cx + rx * 0.55, cy - ry * 0.70));
  }

  // Build edges
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

  const brainPath = generateBrainPath(cx, cy, rx, ry, canvasHeight);

  return { nodes, edges, brainPath };
}

/**
 * Generate a stylized brain silhouette path (sagittal view).
 * The path wraps around the node positions with ~20% margin,
 * forming a recognizable brain shape with a brain stem.
 *
 * Shape: occipital (back/left) → parietal (top) → frontal (front/right)
 *        → temporal (front-bottom) → brain stem → back to occipital
 */
function generateBrainPath(
  cx: number, cy: number,
  rx: number, ry: number,
  canvasHeight: number,
): string {
  // Silhouette extends ~22% beyond the node ellipse
  const mx = rx * 1.22;
  const my = ry * 1.22;

  // Brain stem bottom (slightly below embedding position)
  const stemY = cy + ry + canvasHeight * 0.15;

  // Trace clockwise starting from brain stem bottom
  return [
    // Brain stem bottom
    `M ${cx} ${stemY}`,

    // Left/back: stem up through cerebellum area to lower occipital
    `C ${cx - mx * 0.12} ${stemY - my * 0.15},
       ${cx - mx * 0.42} ${cy + my * 0.65},
       ${cx - mx * 0.55} ${cy + my * 0.35}`,

    // Occipital (back of brain): continue upward
    `C ${cx - mx * 0.72} ${cy + my * 0.05},
       ${cx - mx * 0.88} ${cy - my * 0.15},
       ${cx - mx * 0.88} ${cy - my * 0.38}`,

    // Occipital to parietal (back-top to top)
    `C ${cx - mx * 0.88} ${cy - my * 0.62},
       ${cx - mx * 0.60} ${cy - my * 0.92},
       ${cx - mx * 0.18} ${cy - my * 0.98}`,

    // Parietal to frontal (top to front-top)
    `C ${cx + mx * 0.18} ${cy - my * 1.02},
       ${cx + mx * 0.58} ${cy - my * 0.95},
       ${cx + mx * 0.82} ${cy - my * 0.68}`,

    // Frontal (front of brain): curve downward
    `C ${cx + mx * 0.96} ${cy - my * 0.42},
       ${cx + mx * 1.00} ${cy - my * 0.08},
       ${cx + mx * 0.92} ${cy + my * 0.18}`,

    // Temporal (front-bottom): continue down toward stem
    `C ${cx + mx * 0.82} ${cy + my * 0.42},
       ${cx + mx * 0.55} ${cy + my * 0.60},
       ${cx + mx * 0.35} ${cy + my * 0.55}`,

    // Temporal to brain stem
    `C ${cx + mx * 0.18} ${cy + my * 0.50},
       ${cx + mx * 0.10} ${stemY - my * 0.10},
       ${cx} ${stemY}`,

    'Z',
  ].join(' ');
}
