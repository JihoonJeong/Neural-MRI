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
import { SAEPanel } from './components/Panels/SAEPanel';
import { CollabPanel } from './components/Panels/CollabPanel';
import { CausalTracePanel } from './components/Panels/CausalTracePanel';
import { AttentionPanel } from './components/Panels/AttentionPanel';
import { LogitLensPanel } from './components/Panels/LogitLensPanel';
import { PeerCursors } from './components/PeerCursors';
import { RecordingBar } from './components/RecordingBar';
import { SettingsModal } from './components/SettingsModal';
import { useCrossModelStore } from './store/useCrossModelStore';
import { useSettingsStore } from './store/useSettingsStore';
import { useModelStore } from './store/useModelStore';
import { useScanStore } from './store/useScanStore';
import { useCompareStore } from './store/useCompareStore';
import { useCollabStore } from './store/useCollabStore';
import { useRecordingStore } from './store/useRecordingStore';

export default function App() {
  const fetchModelInfo = useModelStore((s) => s.fetchModelInfo);
  const fetchModels = useModelStore((s) => s.fetchModels);
  const addLog = useScanStore((s) => s.addLog);
  const isCompareMode = useCompareStore((s) => s.isCompareMode);
  const isCrossModelMode = useCrossModelStore((s) => s.isCrossModelMode);
  const collabRole = useCollabStore((s) => s.role);
  const remoteScanState = useCollabStore((s) => s.remoteScanState);
  const joinSession = useCollabStore((s) => s.joinSession);
  const sendCursor = useCollabStore((s) => s.sendCursor);
  const playbackFrame = useRecordingStore((s) => s.currentFrame);

  // Fetch model info, available models, and settings on mount
  useEffect(() => {
    fetchModels();
    useSettingsStore.getState().fetchTokenStatus();
    useSettingsStore.getState().fetchCacheStatus();
    fetchModelInfo().then(() => {
      const info = useModelStore.getState().modelInfo;
      if (info) {
        addLog(`Model ready: ${info.model_id} (${(info.n_params / 1e6).toFixed(0)}M params, ${info.device})`);
      }
    });
  }, [fetchModelInfo, fetchModels, addLog]);

  // Auto-join session from ?session= URL param
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const sessionParam = params.get('session');
    if (sessionParam) {
      joinSession(sessionParam).catch(() => {});
    }
  }, [joinSession]);

  // Build dataOverride: playback > collab viewer > none
  const playbackOverride = playbackFrame ? {
    mode: playbackFrame.mode,
    structuralData: playbackFrame.structuralData,
    weightData: playbackFrame.weightData,
    activationData: playbackFrame.activationData,
    circuitData: playbackFrame.circuitData,
    anomalyData: playbackFrame.anomalyData,
    selectedTokenIdx: playbackFrame.selectedTokenIdx,
  } : undefined;

  const viewerOverride = collabRole === 'viewer' && remoteScanState ? {
    mode: remoteScanState.mode,
    structuralData: remoteScanState.structuralData,
    weightData: remoteScanState.weightData,
    activationData: remoteScanState.activationData,
    circuitData: remoteScanState.circuitData,
    anomalyData: remoteScanState.anomalyData,
    selectedTokenIdx: remoteScanState.selectedTokenIdx,
  } : undefined;

  const dataOverride = playbackOverride ?? viewerOverride;

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
          <div
            className="flex-1 flex justify-center items-start relative overflow-hidden"
            onMouseMove={(e) => {
              if (useCollabStore.getState().isInSession) {
                const rect = e.currentTarget.getBoundingClientRect();
                sendCursor(e.clientX - rect.left, e.clientY - rect.top);
              }
            }}
          >
            <ScanLineOverlay />
            {(isCompareMode || isCrossModelMode) ? <CompareView /> : <ScanCanvas dataOverride={dataOverride} />}
            <PeerCursors />
          </div>
          <TokenStepper />
          <RecordingBar />
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
          {(isCompareMode || isCrossModelMode) ? <DiffPanel /> : <LayerSummary />}
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <StimPanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <ComparisonPanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <CausalTracePanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <AttentionPanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <LogitLensPanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <BatteryPanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <SAEPanel />
          </div>
          <div style={{ borderTop: '1px solid var(--border)' }}>
            <CollabPanel />
          </div>
        </aside>
      </div>
      <GuideModal />
      <ReportModal />
      <BatteryDetailModal />
      <SettingsModal />
    </div>
  );
}
