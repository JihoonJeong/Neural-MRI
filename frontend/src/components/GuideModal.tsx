import { useEffect } from 'react';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';

const SECTIONS: { key: string; color: string }[] = [
  { key: 'overview', color: 'var(--accent-active)' },
  { key: 'T1', color: '#e0e0e0' },
  { key: 'T2', color: '#aaccee' },
  { key: 'fMRI', color: '#ff6644' },
  { key: 'DTI', color: '#44ddaa' },
  { key: 'FLAIR', color: '#ff4466' },
  { key: 'perturbation', color: 'var(--accent-active)' },
  { key: 'shortcuts', color: 'var(--text-primary)' },
];

export function GuideModal() {
  const { isGuideOpen, closeGuide, toggleLocale, locale, t } = useLocaleStore();

  // ESC and H key handlers
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isGuideOpen) {
        closeGuide();
      }
      if (e.key === 'h' || e.key === 'H') {
        // Don't toggle when typing in an input
        const tag = (e.target as HTMLElement).tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        if (isGuideOpen) closeGuide();
        else useLocaleStore.getState().openGuide();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isGuideOpen, closeGuide]);

  // Prevent body scroll when open
  useEffect(() => {
    if (isGuideOpen) {
      document.body.style.overflow = 'hidden';
      return () => { document.body.style.overflow = ''; };
    }
  }, [isGuideOpen]);

  if (!isGuideOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 100,
        background: 'rgba(0,0,0,0.75)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        animation: 'nmri-fade-in 200ms ease-out',
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) closeGuide();
      }}
    >
      <div
        style={{
          background: 'var(--bg-primary)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          width: '90vw',
          maxWidth: 700,
          maxHeight: '80vh',
          display: 'flex',
          flexDirection: 'column',
          boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center gap-3 px-5 py-3 shrink-0"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <button
            onClick={closeGuide}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: 'var(--font-size-lg)',
              cursor: 'pointer',
              fontFamily: 'var(--font-primary)',
              padding: '0 4px',
            }}
          >
            X
          </button>
          <span
            className="tracking-widest font-bold"
            style={{ color: 'var(--text-primary)', fontSize: 'var(--font-size-lg)' }}
          >
            {t('guide.title')}
          </span>
          <button
            onClick={toggleLocale}
            className="ml-auto rounded"
            style={{
              background: 'rgba(0,255,170,0.08)',
              border: '1px solid rgba(0,255,170,0.2)',
              color: 'var(--accent-active)',
              padding: '2px 8px',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
              letterSpacing: '1px',
            }}
          >
            {locale === 'en' ? 'KO' : 'EN'}
          </button>
        </div>

        {/* Content */}
        <div
          className="overflow-y-auto px-5 py-4"
          style={{ flex: 1 }}
        >
          {SECTIONS.map(({ key, color }) => (
            <div key={key} className="mb-6">
              <div
                className="tracking-wide font-bold mb-2"
                style={{
                  color,
                  fontSize: 'var(--font-size-md)',
                  borderBottom: `1px solid ${color}33`,
                  paddingBottom: 4,
                }}
              >
                {t(`guide.${key}.title` as TranslationKey)}
              </div>
              <div
                style={{
                  color: 'var(--text-data)',
                  fontSize: 'var(--font-size-sm)',
                  lineHeight: 1.7,
                  whiteSpace: 'pre-line',
                }}
              >
                {t(`guide.${key}.body` as TranslationKey)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
