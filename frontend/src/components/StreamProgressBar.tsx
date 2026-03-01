import { useStreamStore } from '../store/useStreamStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';

export function StreamProgressBar() {
  const status = useStreamStore((s) => s.status);
  const currentToken = useStreamStore((s) => s.currentToken);
  const totalTokens = useStreamStore((s) => s.totalTokens);
  const tokens = useStreamStore((s) => s.tokens);
  const t = useLocaleStore((s) => s.t);

  if (status !== 'streaming' && status !== 'connecting') return null;

  const progress = totalTokens > 0 ? currentToken / totalTokens : 0;
  const currentTokenText = tokens[currentToken - 1]
    ? tokens[currentToken - 1].replace(/^▁/, ' ').replace(/^Ġ/, ' ')
    : '';

  const label =
    status === 'connecting'
      ? t('stream.connecting' as TranslationKey)
      : `${t('stream.streaming' as TranslationKey).replace('...', '')} ${currentToken}/${totalTokens}`;

  return (
    <div
      className="flex items-center gap-2 px-3"
      style={{
        height: 20,
        borderTop: '1px solid var(--border)',
        fontSize: 'var(--font-size-xs)',
        fontFamily: 'var(--font-primary)',
      }}
    >
      {/* Left: label */}
      <span
        className="shrink-0"
        style={{ color: '#00aaff', animation: 'scan-pulse 1.5s ease-in-out infinite' }}
      >
        {label}
      </span>

      {/* Center: progress bar */}
      <div
        className="flex-1"
        style={{
          height: 3,
          background: 'rgba(255,255,255,0.06)',
          borderRadius: 2,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${progress * 100}%`,
            height: '100%',
            background: 'linear-gradient(90deg, #00aaff, #00ffaa)',
            borderRadius: 2,
            transition: 'width 0.15s ease-out',
          }}
        />
      </div>

      {/* Right: current token */}
      {currentTokenText && (
        <span className="shrink-0" style={{ color: 'var(--text-secondary)' }}>
          {currentTokenText}
        </span>
      )}
    </div>
  );
}
