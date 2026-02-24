import { useScanStore } from '../store/useScanStore';
import { useLocaleStore } from '../store/useLocaleStore';
import { SCAN_MODES, type ScanMode } from '../types/model';
import { Tooltip } from './Tooltip';
import type { TranslationKey } from '../i18n/translations';

const MODE_KEYS: ScanMode[] = ['T1', 'T2', 'fMRI', 'DTI', 'FLAIR'];

export function ModeTabs() {
  const { mode, setMode } = useScanStore();
  const t = useLocaleStore((s) => s.t);

  return (
    <div className="flex" style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
      {MODE_KEYS.map((key) => {
        const m = SCAN_MODES[key];
        const isActive = mode === key;
        return (
          <div key={key} style={{ position: 'relative', flex: 1 }}>
            <Tooltip text={t(`tooltip.${key}` as TranslationKey)} position="bottom">
              <button
                onClick={() => setMode(key)}
                className="w-full py-2.5 px-2 transition-all duration-300"
                style={{
                  background: isActive ? 'rgba(100,170,136,0.08)' : 'transparent',
                  border: 'none',
                  borderBottom: isActive ? `2px solid ${m.color}` : '2px solid transparent',
                  color: isActive ? m.color : 'var(--text-secondary)',
                  fontSize: 'var(--font-size-sm)',
                  fontFamily: 'var(--font-primary)',
                  cursor: 'pointer',
                  letterSpacing: '0.5px',
                }}
              >
                <div style={{ fontWeight: isActive ? 'bold' : 'normal' }}>{m.label}</div>
                {isActive && (
                  <div style={{ fontSize: 'var(--font-size-xs)', marginTop: '2px', opacity: 0.7 }}>
                    {m.desc}
                  </div>
                )}
              </button>
            </Tooltip>
          </div>
        );
      })}
    </div>
  );
}
