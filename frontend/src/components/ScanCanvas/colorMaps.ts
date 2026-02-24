import type { ScanMode } from '../../types/model';

/** T1: Grayscale based on param count ratio (0-1) */
export function t1Color(ratio: number): string {
  const v = Math.floor(160 + ratio * 60);
  return `rgb(${v},${v},${v + 10})`;
}

/** T2: Blue scale based on weight magnitude ratio (0-1) */
export function t2Color(ratio: number): string {
  const r = Math.floor(60 + ratio * 40);
  const g = Math.floor(80 + ratio * 80);
  const b = Math.floor(100 + ratio * 155);
  return `rgb(${r},${g},${b})`;
}

/** fMRI placeholder: cool-to-hot */
export function fmriColor(activation: number): string {
  if (activation > 0.6) {
    return `rgb(${Math.floor(200 + activation * 55)},${Math.floor(activation * 120)},${Math.floor(activation * 30)})`;
  }
  if (activation > 0.3) {
    return `rgb(${Math.floor(activation * 200)},${Math.floor(activation * 160)},${Math.floor(40 + activation * 60)})`;
  }
  return `rgb(${Math.floor(30 + activation * 80)},${Math.floor(30 + activation * 100)},${Math.floor(80 + activation * 120)})`;
}

/** Get node color for a given mode and ratio value */
export function getNodeColor(mode: ScanMode, ratio: number): string {
  switch (mode) {
    case 'T1':
      return t1Color(ratio);
    case 'T2':
      return t2Color(ratio);
    case 'fMRI':
      return fmriColor(ratio);
    case 'DTI': {
      const hue = ratio * 240;
      return `hsl(${hue}, 70%, 55%)`;
    }
    case 'FLAIR':
      return ratio > 0.3
        ? `rgb(255,${Math.floor(50 + ratio * 60)},${Math.floor(80 + ratio * 40)})`
        : 'rgb(60,65,75)';
    default:
      return '#888';
  }
}

/** Get edge color/opacity for a given mode */
export function getEdgeStyle(
  mode: ScanMode,
  strength: number,
): { stroke: string; strokeWidth: number; opacity: number } {
  switch (mode) {
    case 'T1':
      return { stroke: 'rgb(120,130,140)', strokeWidth: 0.5, opacity: 0.15 };
    case 'T2':
      return { stroke: 'rgb(100,150,200)', strokeWidth: 0.5 + strength, opacity: strength * 0.4 };
    case 'fMRI':
      return strength > 0.5
        ? { stroke: `rgb(255,${Math.floor(100 + strength * 80)},50)`, strokeWidth: strength * 2, opacity: strength * 0.5 }
        : { stroke: 'rgb(50,70,120)', strokeWidth: 0.3, opacity: strength * 0.15 };
    case 'DTI':
      return { stroke: `hsl(${strength * 200 + 100},80%,60%)`, strokeWidth: 1 + strength * 2, opacity: 0.4 + strength * 0.4 };
    case 'FLAIR':
      return strength > 0.3
        ? { stroke: 'rgb(255,80,100)', strokeWidth: 1.5, opacity: 0.5 }
        : { stroke: 'rgb(50,55,65)', strokeWidth: 0.3, opacity: 0.08 };
    default:
      return { stroke: '#555', strokeWidth: 0.5, opacity: 0.1 };
  }
}
