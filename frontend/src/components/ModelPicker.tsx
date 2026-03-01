import { useRef, useState, useEffect, useCallback } from 'react';
import { useModelStore } from '../store/useModelStore';
import { useModelSearchStore } from '../store/useModelSearchStore';
import { useScanStore } from '../store/useScanStore';
import { useLocaleStore } from '../store/useLocaleStore';
import { useDebounce } from '../hooks/useDebounce';
import type { TranslationKey } from '../i18n/translations';
import type { HubSearchResult } from '../types/settings';
import type { ModelListEntry } from '../api/client';

function formatDownloads(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function TlBadge({ compat }: { compat: boolean | null }) {
  const t = useLocaleStore((s) => s.t);
  if (compat === true) {
    return (
      <span
        style={{
          fontSize: 'var(--font-size-xs)',
          color: '#00ffaa',
          border: '1px solid rgba(0,255,170,0.3)',
          padding: '0 4px',
          borderRadius: 2,
        }}
      >
        {t('modelPicker.tlCompat' as TranslationKey)}
      </span>
    );
  }
  if (compat === false) {
    return (
      <span
        style={{
          fontSize: 'var(--font-size-xs)',
          color: '#ff4466',
          border: '1px solid rgba(255,68,102,0.3)',
          padding: '0 4px',
          borderRadius: 2,
        }}
      >
        {t('modelPicker.tlIncompat' as TranslationKey)}
      </span>
    );
  }
  return (
    <span
      style={{
        fontSize: 'var(--font-size-xs)',
        color: '#ffaa44',
        border: '1px solid rgba(255,170,68,0.3)',
        padding: '0 4px',
        borderRadius: 2,
      }}
    >
      {t('modelPicker.tlUnknown' as TranslationKey)}
    </span>
  );
}

export function ModelPicker() {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { modelInfo, isLoading, availableModels, loadModel } = useModelStore();
  const { query, setQuery, results, isSearching, search, recentModels, clearResults, tlOnly, setTlOnly } =
    useModelSearchStore();
  const addLog = useScanStore((s) => s.addLog);
  const t = useLocaleStore((s) => s.t);

  // Close on outside click
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

  // Debounced search
  const debouncedSearch = useDebounce(
    useCallback((q: string) => search(q), [search]),
    300,
  );

  const handleQueryChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    setQuery(val);
    debouncedSearch(val);
  };

  const handleSelect = async (modelId: string) => {
    setOpen(false);
    clearResults();
    addLog(`Loading model: ${modelId}...`);
    await loadModel(modelId);
    const info = useModelStore.getState().modelInfo;
    if (info) {
      addLog(`Model loaded: ${modelId}`);
    }
  };

  // Registry models from available list
  const registryModels = availableModels.filter(
    (m) => !m.source || m.source === 'registry',
  );

  // Recent models that aren't in the registry
  const recentIds = recentModels.filter(
    (id) => !registryModels.some((m) => m.model_id === id),
  );

  const currentName =
    modelInfo?.model_name || modelInfo?.model_id || 'gpt2';

  const rowStyle = {
    display: 'block' as const,
    width: '100%',
    textAlign: 'left' as const,
    background: 'none',
    border: 'none',
    padding: '6px 12px',
    cursor: 'pointer',
    fontFamily: 'var(--font-primary)',
  };

  const renderRegistryRow = (m: ModelListEntry) => (
    <button
      key={m.model_id}
      onClick={() => handleSelect(m.model_id)}
      style={rowStyle}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.background =
          'rgba(255,255,255,0.05)';
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.background = 'none';
      }}
    >
      <div className="flex items-center gap-2">
        <span
          style={{
            fontSize: 'var(--font-size-sm)',
            color: m.is_loaded ? 'var(--accent-active)' : 'var(--text-data)',
          }}
        >
          {m.gated ? '\u{1F512} ' : ''}
          {m.display_name}
        </span>
        <span
          style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
        >
          {m.params}
        </span>
        {m.is_loaded && (
          <span style={{ fontSize: 'var(--font-size-xs)', color: 'var(--accent-active)' }}>
            ●
          </span>
        )}
      </div>
    </button>
  );

  const renderSearchRow = (r: HubSearchResult) => (
    <button
      key={r.model_id}
      onClick={() => handleSelect(r.model_id)}
      style={rowStyle}
      onMouseEnter={(e) => {
        (e.currentTarget as HTMLElement).style.background =
          'rgba(255,255,255,0.05)';
      }}
      onMouseLeave={(e) => {
        (e.currentTarget as HTMLElement).style.background = 'none';
      }}
    >
      <div className="flex items-center gap-2 flex-wrap">
        <span
          style={{ fontSize: 'var(--font-size-sm)', color: 'var(--text-data)' }}
        >
          {r.gated ? '\u{1F512} ' : ''}
          {r.model_id}
        </span>
        <span
          style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}
        >
          ↓{formatDownloads(r.downloads)}
        </span>
        <TlBadge compat={r.tl_compat} />
      </div>
    </button>
  );

  const sectionLabel = (text: string) => (
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
      {text}
    </div>
  );

  return (
    <div ref={ref} style={{ position: 'relative' }}>
      <button
        onClick={() => setOpen(!open)}
        disabled={isLoading}
        className="rounded"
        style={{
          background: open ? 'rgba(0,255,170,0.08)' : 'var(--bg-surface)',
          border: '1px solid var(--border)',
          color: 'var(--text-primary)',
          padding: '4px 10px',
          fontSize: 'var(--font-size-sm)',
          fontFamily: 'var(--font-primary)',
          cursor: isLoading ? 'default' : 'pointer',
          maxWidth: 200,
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}
      >
        {currentName} ▾
      </button>

      {open && (
        <div
          style={{
            position: 'absolute',
            top: '100%',
            left: 0,
            marginTop: 4,
            background: 'var(--bg-surface)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            width: 340,
            maxHeight: 420,
            overflowY: 'auto',
            zIndex: 40,
            boxShadow: '0 4px 24px rgba(0,0,0,0.6)',
            animation: 'nmri-fade-in 0.15s ease-out',
          }}
        >
          {/* Search Input */}
          <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--border)' }}>
            <input
              value={query}
              onChange={handleQueryChange}
              placeholder={t('modelPicker.searchPlaceholder' as TranslationKey)}
              autoFocus
              style={{
                width: '100%',
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
            <div className="flex items-center gap-2" style={{ marginTop: 4 }}>
              <label
                className="flex items-center gap-1"
                style={{
                  fontSize: 'var(--font-size-xs)',
                  color: 'var(--text-secondary)',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={tlOnly}
                  onChange={(e) => {
                    setTlOnly(e.target.checked);
                    if (query.trim()) search(query);
                  }}
                  style={{ accentColor: '#00ffaa' }}
                />
                {t('modelPicker.tlOnly' as TranslationKey)}
              </label>
              {isSearching && (
                <span
                  className="loading-dots"
                  style={{ fontSize: 'var(--font-size-xs)', color: 'var(--accent-active)' }}
                >
                  ...
                </span>
              )}
            </div>
          </div>

          {/* Recommended (registry) models */}
          {registryModels.length > 0 && (
            <div>
              {sectionLabel(t('modelPicker.recommended' as TranslationKey))}
              {registryModels.map(renderRegistryRow)}
            </div>
          )}

          {/* Recent models */}
          {recentIds.length > 0 && (
            <div>
              {sectionLabel(t('modelPicker.recent' as TranslationKey))}
              {recentIds.map((id) => (
                <button
                  key={id}
                  onClick={() => handleSelect(id)}
                  style={rowStyle}
                  onMouseEnter={(e) => {
                    (e.currentTarget as HTMLElement).style.background =
                      'rgba(255,255,255,0.05)';
                  }}
                  onMouseLeave={(e) => {
                    (e.currentTarget as HTMLElement).style.background = 'none';
                  }}
                >
                  <span
                    style={{
                      fontSize: 'var(--font-size-sm)',
                      color: 'var(--text-data)',
                    }}
                  >
                    {id}
                  </span>
                </button>
              ))}
            </div>
          )}

          {/* Search Results */}
          {results.length > 0 && (
            <div>
              {sectionLabel(t('modelPicker.results' as TranslationKey))}
              {results.map(renderSearchRow)}
            </div>
          )}

          {/* No results */}
          {query.trim() && !isSearching && results.length === 0 && (
            <div
              style={{
                padding: '12px',
                fontSize: 'var(--font-size-xs)',
                color: 'var(--text-secondary)',
                textAlign: 'center',
              }}
            >
              {t('modelPicker.noResults' as TranslationKey)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
