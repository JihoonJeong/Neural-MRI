import { useScanStore } from '../store/useScanStore';
import type { LayoutMode } from '../store/useScanStore';
import { useEmotionStore } from '../store/useEmotionStore';
import { useLocaleStore } from '../store/useLocaleStore';
import { SCAN_MODES, type ScanMode } from '../types/model';
import { Tooltip } from './Tooltip';
import type { TranslationKey } from '../i18n/translations';

const MODE_KEYS: ScanMode[] = ['T1', 'T2', 'fMRI', 'DTI', 'FLAIR'];
const EMO_COLOR = '#e879a0';

const LAYOUT_ORDER: LayoutMode[] = ['vertical', 'brain', 'network', 'radial'];
const LAYOUT_ICONS: Record<LayoutMode, string> = {
  vertical: '\u2261',   // ≡ stack
  brain: '\u25ce',      // ◎ brain
  network: '\u2b2f',    // ⬯ hexagon (network)
  radial: '\u25c9',     // ◉ bullseye (radial)
};

export function ModeTabs() {
  const { mode, setMode, layoutMode, setLayoutMode } = useScanStore();
  const { tabActive: emoActive, setTabActive: setEmoActive } = useEmotionStore();
  const t = useLocaleStore((s) => s.t);

  const isNonDefault = layoutMode !== 'vertical';

  const cycleLayout = () => {
    const idx = LAYOUT_ORDER.indexOf(layoutMode);
    const next = LAYOUT_ORDER[(idx + 1) % LAYOUT_ORDER.length];
    setLayoutMode(next);
  };

  const layoutKey = `layout.${layoutMode}` as TranslationKey;

  return (
    <div className="flex" style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-secondary)' }}>
      {MODE_KEYS.map((key) => {
        const m = SCAN_MODES[key];
        const isActive = mode === key;
        return (
          <div key={key} style={{ position: 'relative', flex: 1 }}>
            <Tooltip text={t(`tooltip.${key}` as TranslationKey)} position="bottom">
              <button
                onClick={() => { setMode(key); setEmoActive(false); }}
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

      {/* EMO tab */}
      <div style={{ position: 'relative', flex: 0.8 }}>
        <button
          onClick={() => setEmoActive(!emoActive)}
          className="w-full py-2.5 px-2 transition-all duration-300"
          style={{
            background: emoActive ? 'rgba(232,121,160,0.08)' : 'transparent',
            border: 'none',
            borderBottom: emoActive ? `2px solid ${EMO_COLOR}` : '2px solid transparent',
            color: emoActive ? EMO_COLOR : 'var(--text-secondary)',
            fontSize: 'var(--font-size-sm)',
            fontFamily: 'var(--font-primary)',
            cursor: 'pointer',
            letterSpacing: '0.5px',
          }}
        >
          <div style={{ fontWeight: emoActive ? 'bold' : 'normal' }}>EMO</div>
          {emoActive && (
            <div style={{ fontSize: 'var(--font-size-xs)', marginTop: '2px', opacity: 0.7 }}>
              Emotion Vector Analysis
            </div>
          )}
        </button>
      </div>

      {/* Layout cycle button */}
      <div className="flex items-center px-2 shrink-0">
        <Tooltip text={t(layoutKey)} position="bottom">
          <button
            onClick={cycleLayout}
            style={{
              background: isNonDefault ? 'rgba(0,255,170,0.12)' : 'transparent',
              border: isNonDefault ? '1px solid rgba(0,255,170,0.3)' : '1px solid var(--border)',
              color: isNonDefault ? 'var(--accent-active)' : 'var(--text-secondary)',
              padding: '4px 8px',
              fontSize: 'var(--font-size-sm)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
              borderRadius: 4,
              transition: 'all 0.2s ease',
            }}
          >
            {LAYOUT_ICONS[layoutMode]}
          </button>
        </Tooltip>
      </div>
    </div>
  );
}
