import type { CanvasNode, CanvasEdge } from './ScanCanvas';
import type { LayerStructure, ConnectionInfo, LayerWeightStats } from '../../types/scan';

/**
 * Build a radial (concentric ring) layout.
 *
 * - Center: Embedding layer
 * - Rings expand outward: each transformer block is a ring
 *   - Attention nodes on the left arc, MLP nodes on the right arc
 * - Outermost: Output (unembed) layer
 *
 * Like a cross-section of the brain — inner layers are deep, outer are surface.
 */
export function buildRadialLayout(
  layers: LayerStructure[],
  connections: ConnectionInfo[],
  canvasWidth: number,
  canvasHeight: number,
  weightLayers?: LayerWeightStats[],
): { nodes: CanvasNode[]; edges: CanvasEdge[] } {
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
    const maxNorm = Math.max(...weightLayers.map((w) => w.l2_norm), 1);
    for (const [id, norms] of grouped) {
      const avg = norms.reduce((a, b) => a + b, 0) / norms.length;
      weightMap.set(id, avg / maxNorm);
    }
  }

  const cx = canvasWidth / 2;
  const cy = canvasHeight / 2;

  // Separate layers by type
  const embedding = layers.filter((l) => l.layer_type === 'embedding');
  const output = layers.filter((l) => l.layer_type === 'output');
  const attention = layers.filter((l) => l.layer_type === 'attention');
  const mlp = layers.filter((l) => l.layer_type === 'mlp');

  // Number of unique block indices (rings)
  const blockIndices = new Set([
    ...attention.map((l) => l.layer_index ?? 0),
    ...mlp.map((l) => l.layer_index ?? 0),
  ]);
  const numRings = blockIndices.size;
  const maxRadius = Math.min(canvasWidth, canvasHeight) * 0.42;
  const minRadius = 40;

  const nodes: CanvasNode[] = [];

  const makeNode = (layer: LayerStructure, x: number, y: number): CanvasNode => {
    const paramRatio = layer.param_count / maxParam;
    const baseRadius = 4 + paramRatio * 8;
    const wRatio = weightMap.get(layer.layer_id) ?? paramRatio;
    return {
      id: layer.layer_id,
      x,
      y,
      radius: baseRadius,
      layerType: layer.layer_type,
      layerIndex: layer.layer_index,
      paramCount: layer.param_count,
      ratio: wRatio,
    };
  };

  // Embedding → center
  for (const layer of embedding) {
    nodes.push(makeNode(layer, cx, cy));
  }

  // Transformer blocks → concentric rings
  // Attention on the left arc (-π/4 to -3π/4), MLP on the right arc (π/4 to 3π/4)
  const ringStep = numRings > 0 ? (maxRadius - minRadius) / numRings : 0;

  for (let i = 0; i < attention.length; i++) {
    const blockIdx = attention[i].layer_index ?? i;
    const ring = minRadius + (blockIdx + 1) * ringStep;
    // Spread attention nodes along left arc
    const arcSpread = Math.min(Math.PI * 0.6, Math.PI * 0.3 + attention.length * 0.02);
    const baseAngle = Math.PI; // left side
    const offset = attention.length > 1
      ? (i / (attention.length - 1) - 0.5) * arcSpread
      : 0;
    const angle = baseAngle + offset;
    nodes.push(makeNode(attention[i], cx + ring * Math.cos(angle), cy + ring * Math.sin(angle)));
  }

  for (let i = 0; i < mlp.length; i++) {
    const blockIdx = mlp[i].layer_index ?? i;
    const ring = minRadius + (blockIdx + 1) * ringStep;
    // Spread MLP nodes along right arc
    const arcSpread = Math.min(Math.PI * 0.6, Math.PI * 0.3 + mlp.length * 0.02);
    const baseAngle = 0; // right side
    const offset = mlp.length > 1
      ? (i / (mlp.length - 1) - 0.5) * arcSpread
      : 0;
    const angle = baseAngle + offset;
    nodes.push(makeNode(mlp[i], cx + ring * Math.cos(angle), cy + ring * Math.sin(angle)));
  }

  // Output → outermost position (bottom)
  for (const layer of output) {
    nodes.push(makeNode(layer, cx, cy + maxRadius + 20));
  }

  // Build edges
  const nodeMap = new Map(nodes.map((nd) => [nd.id, nd]));
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
