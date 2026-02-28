import type { DiagnosticReport } from '../types/report';

// CSS variable → resolved value map (from theme/variables.css)
export const CSS_VAR_MAP: Record<string, string> = {
  'var(--bg-primary)': '#0a0c10',
  'var(--bg-secondary)': '#0c0e14',
  'var(--bg-surface)': '#12151c',
  'var(--border)': 'rgba(100, 170, 136, 0.15)',
  'var(--text-primary)': '#66aa88',
  'var(--text-secondary)': '#556677',
  'var(--text-data)': '#aabbcc',
  'var(--accent-active)': '#00ffaa',
  'var(--font-primary)': "'JetBrains Mono', 'Fira Code', 'Courier New', monospace",
  'var(--font-size-xs)': '9px',
  'var(--font-size-sm)': '10px',
  'var(--font-size-md)': '11px',
  'var(--font-size-lg)': '14px',
};

export function downloadFile(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export function resolveCssVars(svgStr: string): string {
  let resolved = svgStr;
  for (const [cssVar, value] of Object.entries(CSS_VAR_MAP)) {
    resolved = resolved.replaceAll(cssVar, value);
  }
  return resolved;
}

export function prepareSvgClone(svgEl: SVGSVGElement): string {
  const clone = svgEl.cloneNode(true) as SVGSVGElement;
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');

  const w = svgEl.getAttribute('width') || '560';
  const h = svgEl.getAttribute('height') || '620';

  // Insert background rect as first child
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  bg.setAttribute('width', w);
  bg.setAttribute('height', h);
  bg.setAttribute('fill', '#0a0c10');
  clone.insertBefore(bg, clone.firstChild);

  const raw = new XMLSerializer().serializeToString(clone);
  return resolveCssVars(raw);
}

/** Get all canvas SVGs from DOM */
function getCanvasSvgs(): SVGSVGElement[] {
  const els = document.querySelectorAll<SVGSVGElement>('[data-nmri-canvas]');
  return Array.from(els);
}

export function exportSvg(filename = 'neural-mri-scan.svg') {
  const svgs = getCanvasSvgs();
  if (svgs.length === 0) return;

  if (svgs.length === 1) {
    const svgStr = prepareSvgClone(svgs[0]);
    downloadFile(new Blob([svgStr], { type: 'image/svg+xml' }), filename);
    return;
  }

  // Compare mode: merge two SVGs side by side
  const w1 = parseInt(svgs[0].getAttribute('width') || '270');
  const w2 = parseInt(svgs[1].getAttribute('width') || '270');
  const h = parseInt(svgs[0].getAttribute('height') || '620');
  const gap = 4;
  const totalW = w1 + w2 + gap;

  const svg1 = prepareSvgClone(svgs[0]);
  const svg2 = prepareSvgClone(svgs[1]);

  const combined = `<svg xmlns="http://www.w3.org/2000/svg" width="${totalW}" height="${h}">
<rect width="${totalW}" height="${h}" fill="#0a0c10"/>
<g transform="translate(0,0)">${stripSvgWrapper(svg1)}</g>
<g transform="translate(${w1 + gap},0)">${stripSvgWrapper(svg2)}</g>
</svg>`;

  downloadFile(new Blob([combined], { type: 'image/svg+xml' }), filename);
}

function stripSvgWrapper(svgStr: string): string {
  return svgStr.replace(/<svg[^>]*>/, '').replace(/<\/svg>$/, '');
}

export function exportPng(filename = 'neural-mri-scan.png', scale = 2) {
  const svgs = getCanvasSvgs();
  if (svgs.length === 0) return;

  let svgStr: string;
  let w: number;
  let h: number;

  if (svgs.length === 1) {
    svgStr = prepareSvgClone(svgs[0]);
    w = parseInt(svgs[0].getAttribute('width') || '560');
    h = parseInt(svgs[0].getAttribute('height') || '620');
  } else {
    const w1 = parseInt(svgs[0].getAttribute('width') || '270');
    const w2 = parseInt(svgs[1].getAttribute('width') || '270');
    const gap = 4;
    w = w1 + w2 + gap;
    h = parseInt(svgs[0].getAttribute('height') || '620');

    const svg1 = prepareSvgClone(svgs[0]);
    const svg2 = prepareSvgClone(svgs[1]);
    svgStr = `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">
<rect width="${w}" height="${h}" fill="#0a0c10"/>
<g transform="translate(0,0)">${stripSvgWrapper(svg1)}</g>
<g transform="translate(${w1 + gap},0)">${stripSvgWrapper(svg2)}</g>
</svg>`;
  }

  const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const img = new Image();

  img.onload = () => {
    const canvas = document.createElement('canvas');
    canvas.width = w * scale;
    canvas.height = h * scale;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    URL.revokeObjectURL(url);

    canvas.toBlob((pngBlob) => {
      if (pngBlob) downloadFile(pngBlob, filename);
    }, 'image/png');
  };

  img.src = url;
}

export function exportJson(data: unknown, filename = 'neural-mri-data.json') {
  const json = JSON.stringify(data, null, 2);
  downloadFile(new Blob([json], { type: 'application/json' }), filename);
}

export function exportReport(report: DiagnosticReport, filename = 'neural-mri-report.md') {
  const paramStr = `${(report.total_params / 1e6).toFixed(0)}M`;
  const lines: string[] = [
    '# NEURAL MRI DIAGNOSTIC REPORT',
    '',
    `**Patient:** ${report.model_name} (${paramStr})`,
    `**Date:** ${report.date}`,
    `**Prompt:** "${report.prompt}"`,
    '',
    '## TECHNIQUE',
    report.technique.join(', '),
    '',
    '## FINDINGS',
  ];

  for (const f of report.findings) {
    lines.push(`### [${f.scan_mode}] ${f.title} — ${f.severity.toUpperCase()}`);
    for (const d of f.details) {
      lines.push(`- ${d}`);
    }
    if (f.explanation) {
      lines.push('', `> ${f.explanation}`);
    }
    lines.push('');
  }

  lines.push('## IMPRESSION');
  for (const imp of report.impressions) {
    lines.push(`${imp.index}. ${imp.text} _(${imp.severity})_`);
  }

  lines.push('', '## RECOMMENDATION');
  for (const rec of report.recommendations) {
    lines.push(`- ${rec}`);
  }

  lines.push('', '---', '_Generated by Neural MRI_');

  const md = lines.join('\n');
  downloadFile(new Blob([md], { type: 'text/markdown' }), filename);
}
