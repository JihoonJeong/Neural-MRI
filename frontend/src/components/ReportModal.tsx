import { useEffect, useState } from 'react';
import { useReportStore } from '../store/useReportStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { ReportFinding } from '../types/report';
import type { TranslationKey } from '../i18n/translations';

const SEVERITY_COLORS: Record<string, string> = {
  normal: 'var(--text-data)',
  notable: '#ffaa44',
  warning: '#ff6464',
};

const MODE_COLORS: Record<string, string> = {
  T1: '#e0e0e0',
  T2: '#aaccee',
  fMRI: '#ff6644',
  DTI: '#44ddaa',
  FLAIR: '#ff4466',
  battery: '#aa88ff',
};

function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span
      style={{
        color: SEVERITY_COLORS[severity] || 'var(--text-secondary)',
        fontSize: 'var(--font-size-xs)',
        fontWeight: 'bold',
        textTransform: 'uppercase',
        letterSpacing: '1px',
      }}
    >
      {severity}
    </span>
  );
}

function FindingSection({ finding }: { finding: ReportFinding }) {
  const [expanded, setExpanded] = useState(false);
  const t = useLocaleStore((s) => s.t);
  const modeColor = MODE_COLORS[finding.scan_mode] || 'var(--text-secondary)';

  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1">
        <span style={{ color: modeColor, fontSize: 'var(--font-size-sm)', fontWeight: 'bold' }}>
          [{finding.scan_mode}] {finding.title}
        </span>
        <SeverityBadge severity={finding.severity} />
      </div>
      {finding.details.map((detail, i) => (
        <div
          key={i}
          style={{
            color: 'var(--text-data)',
            fontSize: 'var(--font-size-xs)',
            paddingLeft: 12,
            lineHeight: 1.6,
          }}
        >
          · {detail}
        </div>
      ))}
      {finding.explanation && (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              cursor: 'pointer',
              padding: '4px 0 0 12px',
              opacity: 0.8,
            }}
          >
            {expanded ? '\u25be' : '\u25b8'} {t('report.explain' as TranslationKey)}
          </button>
          {expanded && (
            <div
              style={{
                margin: '4px 0 0 12px',
                padding: '8px 12px',
                background: 'rgba(255,255,255,0.02)',
                borderLeft: '2px solid rgba(255,255,255,0.08)',
                borderRadius: '0 4px 4px 0',
                color: 'var(--text-data)',
                fontSize: 'var(--font-size-xs)',
                lineHeight: 1.7,
              }}
            >
              {finding.explanation}
            </div>
          )}
        </>
      )}
    </div>
  );
}

export function ReportModal() {
  const { report, isOpen, closeReport } = useReportStore();
  const t = useLocaleStore((s) => s.t);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) closeReport();
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [isOpen, closeReport]);

  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
      return () => { document.body.style.overflow = ''; };
    }
  }, [isOpen]);

  if (!isOpen || !report) return null;

  const paramStr = `${(report.total_params / 1e6).toFixed(0)}M`;

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
        if (e.target === e.currentTarget) closeReport();
      }}
    >
      <div
        style={{
          background: 'var(--bg-primary)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          width: '90vw',
          maxWidth: 700,
          maxHeight: '85vh',
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
            onClick={closeReport}
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
            {t('report.title' as TranslationKey)}
          </span>
        </div>

        {/* Content */}
        <div className="overflow-y-auto px-5 py-4" style={{ flex: 1 }}>
          {/* Patient Info */}
          <div className="mb-4" style={{ color: 'var(--text-data)', fontSize: 'var(--font-size-sm)', lineHeight: 1.8 }}>
            <div>{t('report.patient' as TranslationKey)}: <span style={{ color: 'var(--text-primary)' }}>{report.model_name} ({paramStr})</span></div>
            <div>{t('report.date' as TranslationKey)}: <span style={{ color: 'var(--text-primary)' }}>{report.date}</span></div>
            <div>{t('report.prompt' as TranslationKey)}: <span style={{ color: 'var(--accent-active)' }}>"{report.prompt}"</span></div>
          </div>

          {/* Technique */}
          <div className="mb-4">
            <div
              className="tracking-wide font-bold mb-2"
              style={{
                color: 'var(--text-secondary)',
                fontSize: 'var(--font-size-sm)',
                borderBottom: '1px solid var(--border)',
                paddingBottom: 4,
              }}
            >
              {t('report.technique' as TranslationKey)}
            </div>
            <div style={{ color: 'var(--text-data)', fontSize: 'var(--font-size-sm)' }}>
              {report.technique.join(', ')}
            </div>
          </div>

          {/* Findings */}
          <div className="mb-4">
            <div
              className="tracking-wide font-bold mb-2"
              style={{
                color: 'var(--text-secondary)',
                fontSize: 'var(--font-size-sm)',
                borderBottom: '1px solid var(--border)',
                paddingBottom: 4,
              }}
            >
              {t('report.findings' as TranslationKey)}
            </div>
            {report.findings.map((f, i) => (
              <FindingSection key={i} finding={f} />
            ))}
          </div>

          {/* Impression */}
          <div className="mb-4">
            <div
              className="tracking-wide font-bold mb-2"
              style={{
                color: 'var(--text-secondary)',
                fontSize: 'var(--font-size-sm)',
                borderBottom: '1px solid var(--border)',
                paddingBottom: 4,
              }}
            >
              {t('report.impression' as TranslationKey)}
            </div>
            {report.impressions.map((imp) => (
              <div
                key={imp.index}
                style={{
                  color: SEVERITY_COLORS[imp.severity] || 'var(--text-data)',
                  fontSize: 'var(--font-size-sm)',
                  lineHeight: 1.7,
                  marginBottom: 4,
                }}
              >
                {imp.index}. {imp.text}
              </div>
            ))}
          </div>

          {/* Recommendation */}
          <div className="mb-2">
            <div
              className="tracking-wide font-bold mb-2"
              style={{
                color: 'var(--text-secondary)',
                fontSize: 'var(--font-size-sm)',
                borderBottom: '1px solid var(--border)',
                paddingBottom: 4,
              }}
            >
              {t('report.recommendation' as TranslationKey)}
            </div>
            {report.recommendations.map((rec, i) => (
              <div
                key={i}
                style={{
                  color: 'var(--text-data)',
                  fontSize: 'var(--font-size-sm)',
                  lineHeight: 1.7,
                  paddingLeft: 8,
                }}
              >
                · {rec}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
