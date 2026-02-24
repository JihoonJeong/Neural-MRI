import { useModelStore } from '../store/useModelStore';
import { useScanStore } from '../store/useScanStore';
import { SCAN_MODES } from '../types/model';

export function DicomHeader() {
  const modelInfo = useModelStore((s) => s.modelInfo);
  const { mode, prompt } = useScanStore();

  const now = new Date();
  const ts = `${now.getFullYear()}.${String(now.getMonth() + 1).padStart(2, '0')}.${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;

  return (
    <div
      className="flex justify-between px-3 py-2 opacity-80"
      style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-primary)', letterSpacing: '0.5px' }}
    >
      <div>
        <div>MODEL: {modelInfo?.model_id ?? '---'}</div>
        <div>SEQ: {SCAN_MODES[mode].label.toUpperCase()}</div>
        <div>PROMPT: {prompt.slice(0, 40)}{prompt.length > 40 ? '...' : ''}</div>
      </div>
      <div className="text-right">
        <div>Model Resonance Imaging v0.1</div>
        <div>SCAN: {ts}</div>
        <div>FOV: All Layers</div>
      </div>
    </div>
  );
}
