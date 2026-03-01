import { useState, useEffect } from 'react';
import { useSettingsStore } from '../store/useSettingsStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';

const DEVICES = ['auto', 'cpu', 'cuda', 'mps'] as const;

export function SettingsModal() {
  const {
    isOpen,
    closeSettings,
    tokenStatus,
    cacheStatus,
    devicePreference,
    setDevicePreference,
    isValidating,
    error,
    updateToken,
    clearToken,
    clearCache,
  } = useSettingsStore();
  const t = useLocaleStore((s) => s.t);

  const [tokenInput, setTokenInput] = useState('');

  // ESC to close
  useEffect(() => {
    if (!isOpen) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') closeSettings();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [isOpen, closeSettings]);

  if (!isOpen) return null;

  const handleSetToken = async () => {
    if (!tokenInput.trim()) return;
    await updateToken(tokenInput.trim());
    setTokenInput('');
  };

  const statusDot = (() => {
    if (!tokenStatus || !tokenStatus.is_set) return '#555';
    if (tokenStatus.is_valid === true) return '#00ffaa';
    if (tokenStatus.is_valid === false) return '#ff4466';
    return '#888';
  })();

  const statusLabel = (() => {
    if (!tokenStatus || !tokenStatus.is_set)
      return t('settings.tokenNotSet' as TranslationKey);
    if (isValidating) return '...';
    const validity =
      tokenStatus.is_valid === true
        ? t('settings.tokenValid' as TranslationKey)
        : tokenStatus.is_valid === false
          ? t('settings.tokenInvalid' as TranslationKey)
          : '';
    const src =
      tokenStatus.source === 'env'
        ? t('settings.tokenFromEnv' as TranslationKey)
        : tokenStatus.source === 'runtime'
          ? t('settings.tokenFromSession' as TranslationKey)
          : '';
    return [validity, src].filter(Boolean).join(' · ');
  })();

  const sectionStyle = {
    padding: '12px 16px',
    borderBottom: '1px solid var(--border)',
  };

  const labelStyle = {
    fontSize: 'var(--font-size-xs)',
    color: 'var(--accent-active)',
    letterSpacing: '1px',
    marginBottom: 8,
  };

  return (
    <div
      onClick={closeSettings}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0,0,0,0.7)',
        zIndex: 100,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        animation: 'nmri-fade-in 0.15s ease-out',
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: 'var(--bg-surface)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          width: 420,
          maxHeight: '80vh',
          overflowY: 'auto',
          boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        }}
      >
        {/* Header */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            padding: '12px 16px',
            borderBottom: '1px solid var(--border)',
          }}
        >
          <span
            style={{
              fontSize: 'var(--font-size-md)',
              color: 'var(--text-primary)',
              fontFamily: 'var(--font-primary)',
              letterSpacing: '2px',
            }}
          >
            {t('settings.title' as TranslationKey)}
          </span>
          <button
            onClick={closeSettings}
            style={{
              background: 'none',
              border: 'none',
              color: 'var(--text-secondary)',
              fontSize: 'var(--font-size-md)',
              cursor: 'pointer',
            }}
          >
            ✕
          </button>
        </div>

        {/* HF Token Section */}
        <div style={sectionStyle}>
          <div style={labelStyle}>
            {t('settings.token' as TranslationKey)}
          </div>
          <div className="flex items-center gap-2">
            <input
              type="password"
              value={tokenInput}
              onChange={(e) => setTokenInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSetToken();
              }}
              placeholder={t('settings.tokenPlaceholder' as TranslationKey)}
              style={{
                flex: 1,
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid var(--border)',
                color: 'var(--text-data)',
                padding: '5px 8px',
                fontSize: 'var(--font-size-sm)',
                fontFamily: 'var(--font-primary)',
                outline: 'none',
                borderRadius: 3,
              }}
            />
            <button
              onClick={handleSetToken}
              disabled={isValidating || !tokenInput.trim()}
              style={{
                background: 'rgba(0,255,170,0.12)',
                border: '1px solid rgba(0,255,170,0.3)',
                color: 'var(--accent-active)',
                padding: '4px 10px',
                fontSize: 'var(--font-size-xs)',
                fontFamily: 'var(--font-primary)',
                cursor:
                  isValidating || !tokenInput.trim() ? 'default' : 'pointer',
                borderRadius: 3,
                opacity: isValidating || !tokenInput.trim() ? 0.5 : 1,
              }}
            >
              {t('settings.tokenSet' as TranslationKey)}
            </button>
            {tokenStatus?.is_set && (
              <button
                onClick={clearToken}
                style={{
                  background: 'rgba(255,68,102,0.1)',
                  border: '1px solid rgba(255,68,102,0.3)',
                  color: '#ff4466',
                  padding: '4px 10px',
                  fontSize: 'var(--font-size-xs)',
                  fontFamily: 'var(--font-primary)',
                  cursor: 'pointer',
                  borderRadius: 3,
                }}
              >
                {t('settings.tokenClear' as TranslationKey)}
              </button>
            )}
          </div>
          <div
            className="flex items-center gap-2"
            style={{ marginTop: 6 }}
          >
            <div
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: statusDot,
                flexShrink: 0,
              }}
            />
            <span
              style={{
                fontSize: 'var(--font-size-xs)',
                color: 'var(--text-secondary)',
              }}
            >
              {statusLabel}
            </span>
          </div>
        </div>

        {/* Device Section */}
        <div style={sectionStyle}>
          <div style={labelStyle}>
            {t('settings.device' as TranslationKey)}
          </div>
          <div className="flex gap-2">
            {DEVICES.map((d) => (
              <button
                key={d}
                onClick={() => setDevicePreference(d)}
                style={{
                  background:
                    devicePreference === d
                      ? 'rgba(0,255,170,0.15)'
                      : 'rgba(255,255,255,0.04)',
                  border:
                    devicePreference === d
                      ? '1px solid rgba(0,255,170,0.4)'
                      : '1px solid var(--border)',
                  color:
                    devicePreference === d
                      ? 'var(--accent-active)'
                      : 'var(--text-data)',
                  padding: '4px 12px',
                  fontSize: 'var(--font-size-xs)',
                  fontFamily: 'var(--font-primary)',
                  cursor: 'pointer',
                  borderRadius: 3,
                  textTransform: 'uppercase',
                }}
              >
                {d === 'auto'
                  ? t('settings.deviceAuto' as TranslationKey)
                  : d.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        {/* Cache Section */}
        <div style={{ ...sectionStyle, borderBottom: 'none' }}>
          <div style={labelStyle}>
            {t('settings.cache' as TranslationKey)}
          </div>
          <div className="flex items-center gap-3">
            <span
              style={{
                fontSize: 'var(--font-size-sm)',
                color: 'var(--text-data)',
              }}
            >
              {cacheStatus
                ? `${cacheStatus.entry_count} / ${cacheStatus.max_entries}`
                : '— / —'}{' '}
              {t('settings.cacheEntries' as TranslationKey)}
            </span>
            <button
              onClick={clearCache}
              disabled={!cacheStatus || cacheStatus.entry_count === 0}
              style={{
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid var(--border)',
                color: 'var(--text-secondary)',
                padding: '4px 10px',
                fontSize: 'var(--font-size-xs)',
                fontFamily: 'var(--font-primary)',
                cursor:
                  !cacheStatus || cacheStatus.entry_count === 0
                    ? 'default'
                    : 'pointer',
                borderRadius: 3,
                opacity:
                  !cacheStatus || cacheStatus.entry_count === 0 ? 0.4 : 1,
              }}
            >
              {t('settings.cacheClear' as TranslationKey)}
            </button>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div
            style={{
              padding: '8px 16px',
              fontSize: 'var(--font-size-xs)',
              color: '#ff4466',
            }}
          >
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
