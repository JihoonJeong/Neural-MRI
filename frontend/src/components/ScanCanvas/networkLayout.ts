import type { CanvasNode, CanvasEdge } from './ScanCanvas';
import type { LayerStructure, ConnectionInfo, LayerWeightStats } from '../../types/scan';

/**
 * Build a force-directed network layout using a simple iterative simulation.
 *
 * Nodes repel each other while edges act as springs, producing an organic
 * graph layout. We run a synchronous simulation (no D3 force dependency)
 * to keep it deterministic and fast.
 */
export function buildNetworkLayout(
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
  const n = layers.length;

  // Initialize positions in a circle
  const positions: { x: number; y: number }[] = layers.map((_, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    const r = Math.min(canvasWidth, canvasHeight) * 0.3;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  });

  // Build adjacency set for spring forces
  const connSet = new Set<string>();
  for (const c of connections) {
    connSet.add(`${c.from_id}|${c.to_id}`);
  }
  const idToIdx = new Map(layers.map((l, i) => [l.layer_id, i]));

  // Run simple force simulation
  const iterations = 80;
  const repulsion = 3000;
  const springK = 0.02;
  const springLen = Math.min(canvasWidth, canvasHeight) * 0.12;
  const damping = 0.85;
  const padding = 50;

  const vx = new Float64Array(n);
  const vy = new Float64Array(n);

  for (let iter = 0; iter < iterations; iter++) {
    const temp = 1 - iter / iterations; // cooling

    // Repulsion between all pairs
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = positions[j].x - positions[i].x;
        let dy = positions[j].y - positions[i].y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (repulsion * temp) / (dist * dist);
        dx = (dx / dist) * force;
        dy = (dy / dist) * force;
        vx[i] -= dx;
        vy[i] -= dy;
        vx[j] += dx;
        vy[j] += dy;
      }
    }

    // Spring forces along edges
    for (const c of connections) {
      const i = idToIdx.get(c.from_id);
      const j = idToIdx.get(c.to_id);
      if (i === undefined || j === undefined) continue;
      let dx = positions[j].x - positions[i].x;
      let dy = positions[j].y - positions[i].y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = springK * (dist - springLen);
      dx = (dx / dist) * force;
      dy = (dy / dist) * force;
      vx[i] += dx;
      vy[i] += dy;
      vx[j] -= dx;
      vy[j] -= dy;
    }

    // Apply velocities with damping and clamp to canvas
    for (let i = 0; i < n; i++) {
      vx[i] *= damping;
      vy[i] *= damping;
      positions[i].x += vx[i];
      positions[i].y += vy[i];
      positions[i].x = Math.max(padding, Math.min(canvasWidth - padding, positions[i].x));
      positions[i].y = Math.max(padding, Math.min(canvasHeight - padding, positions[i].y));
    }
  }

  // Build nodes
  const nodes: CanvasNode[] = layers.map((layer, i) => {
    const paramRatio = layer.param_count / maxParam;
    const baseRadius = 4 + paramRatio * 8;
    const wRatio = weightMap.get(layer.layer_id) ?? paramRatio;
    return {
      id: layer.layer_id,
      x: positions[i].x,
      y: positions[i].y,
      radius: baseRadius,
      layerType: layer.layer_type,
      layerIndex: layer.layer_index,
      paramCount: layer.param_count,
      ratio: wRatio,
    };
  });

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
