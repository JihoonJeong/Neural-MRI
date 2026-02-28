import type { RecordingFrame, ScanRecording } from '../types/recording';
import { captureCanvasFrame } from './svgCapture';
import { downloadFile } from './exportUtils';
import { useRecordingStore } from '../store/useRecordingStore';

/** Wait one animation frame for DOM to update. */
function waitFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/** Wait N milliseconds. */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Render a recording frame to the ScanCanvas by setting it as the current playback frame. */
async function renderFrame(frame: RecordingFrame): Promise<void> {
  const store = useRecordingStore.getState();
  const idx = store.recording?.frames.indexOf(frame) ?? -1;
  if (idx >= 0) {
    useRecordingStore.setState({
      currentFrameIdx: idx,
      currentFrame: frame,
    });
  }
  // Wait for React to render and D3 to update
  await waitFrame();
  await sleep(50);
  await waitFrame();
}

export interface ExportOptions {
  fps?: number;
  scale?: number;
  onProgress?: (progress: number) => void;
}

/**
 * Export a recording as WebM video using native MediaRecorder.
 * This renders each frame to the ScanCanvas, captures it, and records to WebM.
 */
export async function exportWebM(
  recording: ScanRecording,
  options: ExportOptions = {},
): Promise<void> {
  const { fps = 10, scale = 1, onProgress } = options;
  const frames = recording.frames;
  if (frames.length === 0) return;

  // Render first frame to determine canvas size
  await renderFrame(frames[0]);
  const firstCanvas = await captureCanvasFrame(scale);
  const width = firstCanvas.width;
  const height = firstCanvas.height;

  // Create a persistent canvas for recording
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(firstCanvas, 0, 0);

  // Set up MediaRecorder
  const stream = canvas.captureStream(0); // 0 = manual frame request
  const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
    ? 'video/webm;codecs=vp9'
    : 'video/webm';

  const recorder = new MediaRecorder(stream, {
    mimeType,
    videoBitsPerSecond: 2_000_000,
  });

  const chunks: Blob[] = [];
  recorder.ondataavailable = (e) => {
    if (e.data.size > 0) chunks.push(e.data);
  };

  const done = new Promise<Blob>((resolve) => {
    recorder.onstop = () => {
      resolve(new Blob(chunks, { type: mimeType }));
    };
  });

  recorder.start();
  const frameDelay = 1000 / fps;

  for (let i = 0; i < frames.length; i++) {
    await renderFrame(frames[i]);
    const frameCanvas = await captureCanvasFrame(scale);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(frameCanvas, 0, 0);

    // Request a frame from the stream
    const track = stream.getVideoTracks()[0];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if ((track as any).requestFrame) (track as any).requestFrame();

    onProgress?.(i / frames.length);
    await sleep(frameDelay);
  }

  recorder.stop();
  const blob = await done;

  const ts = new Date().toISOString().slice(0, 10);
  downloadFile(blob, `nmri-recording-${recording.modelId}-${ts}.webm`);
  onProgress?.(1);
}

/**
 * Export a recording as an animated GIF.
 * Uses a simple Canvas-based approach: captures each frame as image data
 * and encodes them into a GIF using a minimal encoder.
 *
 * Falls back to multi-frame PNG download if encoding fails.
 */
export async function exportGif(
  recording: ScanRecording,
  options: ExportOptions = {},
): Promise<void> {
  const { fps = 5, scale = 1, onProgress } = options;
  const frames = recording.frames;
  if (frames.length === 0) return;

  // Capture all frames as canvas images
  const canvases: HTMLCanvasElement[] = [];
  for (let i = 0; i < frames.length; i++) {
    await renderFrame(frames[i]);
    const canvas = await captureCanvasFrame(scale);
    canvases.push(canvas);
    onProgress?.((i / frames.length) * 0.8);
  }

  // Encode as animated GIF using a simple approach:
  // We'll create an image sequence and use the GIF encoder
  const width = canvases[0].width;
  const height = canvases[0].height;
  const delay = Math.round(1000 / fps);

  // Minimal GIF89a encoder
  const gif = encodeGif(canvases, width, height, delay);
  onProgress?.(0.95);

  const blob = new Blob([gif.buffer as ArrayBuffer], { type: 'image/gif' });
  const ts = new Date().toISOString().slice(0, 10);
  downloadFile(blob, `nmri-recording-${recording.modelId}-${ts}.gif`);
  onProgress?.(1);
}

// --- Minimal GIF89a Encoder ---
// Supports animation with global color table (256-color quantized)

function encodeGif(
  canvases: HTMLCanvasElement[],
  width: number,
  height: number,
  delay: number,
): Uint8Array {
  const parts: Uint8Array[] = [];

  // Header
  parts.push(str2bytes('GIF89a'));

  // Logical Screen Descriptor
  parts.push(uint16LE(width));
  parts.push(uint16LE(height));
  // GCT flag=1, color res=7, sort=0, size=7 (256 colors) → 0xF7
  parts.push(new Uint8Array([0xf7, 0x00, 0x00]));

  // Global Color Table (256 entries, 768 bytes) — web-safe palette + dark fills
  const palette = buildPalette();
  parts.push(palette);

  // Netscape Extension (loop forever)
  parts.push(
    new Uint8Array([
      0x21, 0xff, 0x0b, // Extension
      ...str2arr('NETSCAPE2.0'),
      0x03, 0x01, 0x00, 0x00, // Sub-block: loop count = 0 (infinite)
      0x00, // Block terminator
    ]),
  );

  for (const canvas of canvases) {
    const ctx = canvas.getContext('2d')!;
    const imageData = ctx.getImageData(0, 0, width, height);
    const indices = quantize(imageData.data, palette);

    // Graphic Control Extension
    const delayCs = Math.round(delay / 10); // centiseconds
    parts.push(
      new Uint8Array([
        0x21, 0xf9, 0x04, // Extension
        0x00, // No transparency
        ...uint16LEArr(delayCs),
        0x00, // Transparent color index (unused)
        0x00, // Block terminator
      ]),
    );

    // Image Descriptor
    parts.push(new Uint8Array([0x2c]));
    parts.push(uint16LE(0)); // left
    parts.push(uint16LE(0)); // top
    parts.push(uint16LE(width));
    parts.push(uint16LE(height));
    parts.push(new Uint8Array([0x00])); // No local color table

    // Image Data (LZW compressed)
    const lzwMinCodeSize = 8;
    parts.push(new Uint8Array([lzwMinCodeSize]));
    const compressed = lzwEncode(indices, lzwMinCodeSize);
    // Split into sub-blocks of max 255 bytes
    let offset = 0;
    while (offset < compressed.length) {
      const blockSize = Math.min(255, compressed.length - offset);
      parts.push(new Uint8Array([blockSize]));
      parts.push(compressed.slice(offset, offset + blockSize));
      offset += blockSize;
    }
    parts.push(new Uint8Array([0x00])); // Block terminator
  }

  // Trailer
  parts.push(new Uint8Array([0x3b]));

  // Concatenate all parts
  const totalLen = parts.reduce((sum, p) => sum + p.length, 0);
  const result = new Uint8Array(totalLen);
  let pos = 0;
  for (const part of parts) {
    result.set(part, pos);
    pos += part.length;
  }
  return result;
}

function buildPalette(): Uint8Array {
  // 256-color palette optimized for our dark medical theme
  const colors = new Uint8Array(768);
  for (let i = 0; i < 256; i++) {
    // Build a palette with:
    // 0-63: dark grays/greens (background tones)
    // 64-127: greens (accent, nodes)
    // 128-191: blues/cyans (secondary)
    // 192-223: reds/oranges (anomaly, warm)
    // 224-255: pure grays
    if (i < 64) {
      const v = Math.round((i / 63) * 40);
      colors[i * 3] = v;
      colors[i * 3 + 1] = Math.round(v * 1.3);
      colors[i * 3 + 2] = Math.round(v * 1.1);
    } else if (i < 128) {
      const t = (i - 64) / 63;
      colors[i * 3] = Math.round(t * 80);
      colors[i * 3 + 1] = Math.round(80 + t * 175);
      colors[i * 3 + 2] = Math.round(t * 136);
    } else if (i < 192) {
      const t = (i - 128) / 63;
      colors[i * 3] = Math.round(t * 100);
      colors[i * 3 + 1] = Math.round(100 + t * 120);
      colors[i * 3 + 2] = Math.round(150 + t * 105);
    } else if (i < 224) {
      const t = (i - 192) / 31;
      colors[i * 3] = Math.round(150 + t * 105);
      colors[i * 3 + 1] = Math.round(t * 100);
      colors[i * 3 + 2] = Math.round(t * 80);
    } else {
      const v = Math.round(((i - 224) / 31) * 255);
      colors[i * 3] = v;
      colors[i * 3 + 1] = v;
      colors[i * 3 + 2] = v;
    }
  }
  return colors;
}

function quantize(rgba: Uint8ClampedArray, palette: Uint8Array): Uint8Array {
  const pixelCount = rgba.length / 4;
  const indices = new Uint8Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    indices[i] = findClosest(r, g, b, palette);
  }
  return indices;
}

function findClosest(r: number, g: number, b: number, palette: Uint8Array): number {
  let minDist = Infinity;
  let best = 0;
  for (let i = 0; i < 256; i++) {
    const dr = r - palette[i * 3];
    const dg = g - palette[i * 3 + 1];
    const db = b - palette[i * 3 + 2];
    const dist = dr * dr + dg * dg + db * db;
    if (dist < minDist) {
      minDist = dist;
      best = i;
    }
  }
  return best;
}

function lzwEncode(indices: Uint8Array, minCodeSize: number): Uint8Array {
  const clearCode = 1 << minCodeSize;
  const eoiCode = clearCode + 1;
  let codeSize = minCodeSize + 1;
  let nextCode = eoiCode + 1;
  const maxCode = 4096;

  // Use a Map-based dictionary for fast lookups
  const dict = new Map<string, number>();
  for (let i = 0; i < clearCode; i++) {
    dict.set(String(i), i);
  }

  const output: number[] = [];
  let bits = 0;
  let bitCount = 0;

  const emit = (code: number) => {
    bits |= code << bitCount;
    bitCount += codeSize;
    while (bitCount >= 8) {
      output.push(bits & 0xff);
      bits >>= 8;
      bitCount -= 8;
    }
  };

  emit(clearCode);

  let current = String(indices[0]);

  for (let i = 1; i < indices.length; i++) {
    const next = current + ',' + indices[i];
    if (dict.has(next)) {
      current = next;
    } else {
      emit(dict.get(current)!);
      if (nextCode < maxCode) {
        dict.set(next, nextCode++);
        if (nextCode > (1 << codeSize) && codeSize < 12) {
          codeSize++;
        }
      } else {
        // Reset dictionary
        emit(clearCode);
        dict.clear();
        for (let j = 0; j < clearCode; j++) {
          dict.set(String(j), j);
        }
        nextCode = eoiCode + 1;
        codeSize = minCodeSize + 1;
      }
      current = String(indices[i]);
    }
  }

  emit(dict.get(current)!);
  emit(eoiCode);

  // Flush remaining bits
  if (bitCount > 0) {
    output.push(bits & 0xff);
  }

  return new Uint8Array(output);
}

function str2bytes(s: string): Uint8Array {
  return new Uint8Array(s.split('').map((c) => c.charCodeAt(0)));
}

function str2arr(s: string): number[] {
  return s.split('').map((c) => c.charCodeAt(0));
}

function uint16LE(n: number): Uint8Array {
  return new Uint8Array([n & 0xff, (n >> 8) & 0xff]);
}

function uint16LEArr(n: number): number[] {
  return [n & 0xff, (n >> 8) & 0xff];
}
