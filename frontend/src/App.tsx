import { useEffect } from 'react';
import { TopBar } from './components/TopBar';
import { ModeTabs } from './components/ModeTabs';
import { DicomHeader } from './components/DicomHeader';
import { ScanCanvas } from './components/ScanCanvas/ScanCanvas';
import { ScanLineOverlay } from './components/ScanCanvas/ScanLineOverlay';
import { CompareView } from './components/CompareView';
import { TokenStepper } from './components/TokenStepper';
import { PromptInput } from './components/PromptInput';
import { LayerSummary } from './components/Panels/LayerSummary';
import { StimPanel } from './components/Panels/StimPanel';
import { ComparisonPanel } from './components/Panels/ComparisonPanel';
import { DiffPanel } from './components/Panels/DiffPanel';
import { LogPanel } from './components/Panels/LogPanel';
import { GuideModal } from './components/GuideModal';
import { ReportModal } from './components/ReportModal';
import { BatteryDetailModal } from './components/BatteryDetailModal';
import { BatteryPanel } from './components/Panels/BatteryPanel';
import { useModelStore } from './store/useModelStore';
import { useScanStore } from './store/useScanStore';
import { useCompareStore } from './store/useCompareStore';

export default function App() {
  const fetchModelInfo = useModelStore((s) => s.fetchModelInfo);
  const fetchModels = useModelStore((s) => s.fetchModels);
  const addLog = useScanStore((s) => s.addLog);
  const isCompareMode = useCompareStore((s) => s.isCompareMode);

  // Fetch model info and available models on mount
  useEffect(() => {
    fetchModels();
    fetchModelInfo().then(() => {
      const info = useModelStore.getState().modelInfo;
      if (info) {
        addLog(`Model ready: ${info.model_id} (${(info.n_params / 1e6).toFixed(0)}M params, ${info.device})`);
      }
    });
  }, [fetchModelInfo, fetchModels, addLog]);

  return (
    <div
      className="flex flex-col h-screen font-mono"
      style={{ background: 'var(--bg-primary)' }}
    >
      <TopBar />
      <ModeTabs />

      <div className="flex flex-1 overflow-hidden">
        {/* Left: Main canvas area */}
        <main className="flex-1 flex flex-col">
          <DicomHeader />
          <div className="flex-1 flex justify-center items-start relative overflow-hidden">
            <ScanLineOverlay />
            {isCompareMode ? <CompareView /> : <ScanCanvas />}
          </div>
          <TokenStepper />
          <PromptInput />
          <LogPanel />
        </main>

        {/* Right: Panels */}
        <aside
          className="flex flex-col shrink-0"
          style={{
            width: 240,
            borderLeft: '1px solid var(--border)',
            background: 'var(--bg-secondary)',
          }}
        >
          {isCompareMode ? <DiffPanel /> : <LayerSummary />}
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <StimPanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <ComparisonPanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <BatteryPanel />
          </div>
        </aside>
      </div>
      <GuideModal />
      <ReportModal />
      <BatteryDetailModal />
    </div>
  );
}
