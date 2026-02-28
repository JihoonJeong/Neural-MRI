import { prepareSvgClone } from './exportUtils';

/**
 * Convert an SVG element to a Canvas element.
 * Returns a Promise that resolves once the image is drawn on the canvas.
 */
export function svgToCanvas(
  svgEl: SVGSVGElement,
  scale = 1,
): Promise<HTMLCanvasElement> {
  const w = parseInt(svgEl.getAttribute('width') || '560');
  const h = parseInt(svgEl.getAttribute('height') || '620');
  const svgStr = prepareSvgClone(svgEl);

  const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = w * scale;
      canvas.height = h * scale;
      const ctx = canvas.getContext('2d')!;
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve(canvas);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error('Failed to load SVG as image'));
    };
    img.src = url;
  });
}

/**
 * Capture the current NMRI canvas SVG(s) from the DOM and return as a Canvas.
 */
export async function captureCanvasFrame(scale = 1): Promise<HTMLCanvasElement> {
  const svgEl = document.querySelector<SVGSVGElement>('[data-nmri-canvas]');
  if (!svgEl) throw new Error('No NMRI canvas found in DOM');
  return svgToCanvas(svgEl, scale);
}
