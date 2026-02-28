import { useScanStore } from '../store/useScanStore';
import { useLocaleStore } from '../store/useLocaleStore';
import { SCAN_MODES, type ScanMode } from '../types/model';
import { Tooltip } from './Tooltip';
import type { TranslationKey } from '../i18n/translations';

const MODE_KEYS: ScanMode[] = ['T1', 'T2', 'fMRI', 'DTI', 'FLAIR'];

export function ModeTabs() {
  const { mode, setMode, layoutMode, setLayoutMode } = useScanStore();
  const t = useLocaleStore((s) => s.t);

  const isBrain = layoutMode === 'brain';

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

      {/* Layout toggle */}
      <div className="flex items-center px-2 shrink-0">
        <Tooltip
          text={isBrain ? t('layout.vertical' as TranslationKey) : t('layout.brain' as TranslationKey)}
          position="bottom"
        >
          <button
            onClick={() => setLayoutMode(isBrain ? 'vertical' : 'brain')}
            style={{
              background: isBrain ? 'rgba(0,255,170,0.12)' : 'transparent',
              border: isBrain ? '1px solid rgba(0,255,170,0.3)' : '1px solid var(--border)',
              color: isBrain ? 'var(--accent-active)' : 'var(--text-secondary)',
              padding: '4px 8px',
              fontSize: 'var(--font-size-sm)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
              borderRadius: 4,
              transition: 'all 0.2s ease',
            }}
          >
            {isBrain ? '\u25ce' : '\u2261'}
          </button>
        </Tooltip>
      </div>
    </div>
  );
}
