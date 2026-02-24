import { create } from 'zustand';
import { translations, type Locale, type TranslationKey } from '../i18n/translations';

interface LocaleState {
  locale: Locale;
  isGuideOpen: boolean;
  toggleLocale: () => void;
  openGuide: () => void;
  closeGuide: () => void;
  t: (key: TranslationKey) => string;
}

export const useLocaleStore = create<LocaleState>((set, get) => ({
  locale: (localStorage.getItem('nmri-locale') as Locale) ?? 'en',
  isGuideOpen: false,

  toggleLocale: () => {
    const next = get().locale === 'en' ? 'ko' : 'en';
    localStorage.setItem('nmri-locale', next);
    set({ locale: next });
  },

  openGuide: () => set({ isGuideOpen: true }),
  closeGuide: () => set({ isGuideOpen: false }),

  t: (key) => translations[get().locale][key] ?? key,
}));
