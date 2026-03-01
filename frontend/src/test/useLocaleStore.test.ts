import { describe, it, expect, beforeEach } from 'vitest';
import { useLocaleStore } from '../store/useLocaleStore';

describe('useLocaleStore', () => {
  beforeEach(() => {
    localStorage.clear();
    useLocaleStore.setState({ locale: 'en', isGuideOpen: false });
  });

  it('defaults to English', () => {
    expect(useLocaleStore.getState().locale).toBe('en');
  });

  it('toggleLocale switches between en and ko', () => {
    useLocaleStore.getState().toggleLocale();
    expect(useLocaleStore.getState().locale).toBe('ko');
    expect(localStorage.getItem('nmri-locale')).toBe('ko');

    useLocaleStore.getState().toggleLocale();
    expect(useLocaleStore.getState().locale).toBe('en');
  });

  it('t() returns translation for current locale', () => {
    const t = useLocaleStore.getState().t;
    const en = t('guide.title');
    expect(en).toBeTruthy();
    expect(typeof en).toBe('string');

    useLocaleStore.getState().toggleLocale();
    const ko = useLocaleStore.getState().t('guide.title');
    expect(ko).toBeTruthy();
    // Should be different from English
    expect(ko).not.toBe(en);
  });

  it('t() returns key as fallback for unknown keys', () => {
    const result = useLocaleStore.getState().t('nonexistent.key' as never);
    expect(result).toBe('nonexistent.key');
  });

  it('guide modal controls work', () => {
    expect(useLocaleStore.getState().isGuideOpen).toBe(false);

    useLocaleStore.getState().openGuide();
    expect(useLocaleStore.getState().isGuideOpen).toBe(true);

    useLocaleStore.getState().closeGuide();
    expect(useLocaleStore.getState().isGuideOpen).toBe(false);
  });
});
