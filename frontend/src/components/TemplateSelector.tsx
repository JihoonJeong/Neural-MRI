import { useRef, useState, useEffect } from 'react';
import { PROMPT_TEMPLATES, TEMPLATE_CATEGORIES } from '../data/promptTemplates';
import { useScanStore } from '../store/useScanStore';
import { useCausalTraceStore } from '../store/useCausalTraceStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';

export function TemplateSelector() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const setPrompt = useScanStore((s) => s.setPrompt);
  const setCorruptPrompt = useCausalTraceStore((s) => s.setCorruptPrompt);
  const locale = useLocaleStore((s) => s.locale);
  const t = useLocaleStore((s) => s.t);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  const handleSelect = (id: string) => {
    const tmpl = PROMPT_TEMPLATES.find((t) => t.id === id);
    if (!tmpl) return;
    setPrompt(tmpl.prompt);
    if (tmpl.corruptPrompt) {
      setCorruptPrompt(tmpl.corruptPrompt);
    }
    setOpen(false);
  };

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(!open)}
        title={t('templates.title' as TranslationKey)}
        style={{
          background: open ? 'rgba(0,255,170,0.12)' : 'none',
          border: '1px solid var(--border)',
          color: 'var(--text-secondary)',
          padding: '4px 8px',
          fontSize: 'var(--font-size-sm)',
          fontFamily: 'var(--font-primary)',
          cursor: 'pointer',
          borderRadius: 4,
        }}
      >
        {t('templates.button' as TranslationKey)}
      </button>

      {open && (
        <div
          style={{
            position: 'absolute',
            bottom: '100%',
            left: 0,
            marginBottom: 4,
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            minWidth: 280,
            maxHeight: 320,
            overflowY: 'auto',
            zIndex: 50,
            boxShadow: '0 -4px 16px rgba(0,0,0,0.5)',
            animation: 'nmri-fade-in 0.15s ease-out',
          }}
        >
          {TEMPLATE_CATEGORIES.map((cat) => {
            const templates = PROMPT_TEMPLATES.filter((t) => t.category === cat.id);
            if (templates.length === 0) return null;
            return (
              <div key={cat.id}>
                <div
                  style={{
                    padding: '4px 12px',
                    fontSize: 'var(--font-size-xs)',
                    color: 'var(--accent-active)',
                    letterSpacing: '1px',
                    borderBottom: '1px solid var(--border)',
                    background: 'rgba(0,255,170,0.03)',
                  }}
                >
                  {locale === 'ko' ? cat.nameKo : cat.name}
                </div>
                {templates.map((tmpl) => (
                  <button
                    key={tmpl.id}
                    onClick={() => handleSelect(tmpl.id)}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      background: 'none',
                      border: 'none',
                      color: 'var(--text-data)',
                      padding: '6px 12px',
                      fontSize: 'var(--font-size-xs)',
                      fontFamily: 'var(--font-primary)',
                      cursor: 'pointer',
                    }}
                    onMouseEnter={(e) => { (e.target as HTMLElement).style.background = 'rgba(255,255,255,0.05)'; }}
                    onMouseLeave={(e) => { (e.target as HTMLElement).style.background = 'none'; }}
                  >
                    <div>{locale === 'ko' ? tmpl.nameKo : tmpl.name}</div>
                    <div
                      style={{
                        fontSize: 'var(--font-size-xs)',
                        color: 'var(--text-secondary)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                      }}
                    >
                      {tmpl.prompt}
                    </div>
                  </button>
                ))}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
