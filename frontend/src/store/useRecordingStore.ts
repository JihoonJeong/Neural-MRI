import { create } from 'zustand';
import type { RecordingFrame, ScanRecording, PlaybackSpeed } from '../types/recording';
import { useScanStore } from './useScanStore';
import { useModelStore } from './useModelStore';
import { downloadFile } from '../utils/exportUtils';

interface RecordingState {
  // Recording
  isRecording: boolean;
  frames: RecordingFrame[];
  recordingStart: number;

  // Playback
  recording: ScanRecording | null;
  isPlaying: boolean;
  playbackSpeed: PlaybackSpeed;
  currentFrameIdx: number;

  // Computed-like getter
  currentFrame: RecordingFrame | null;

  // Actions — recording
  startRecording: () => void;
  stopRecording: () => void;
  saveRecording: () => void;
  loadRecording: (file: File) => void;

  // Actions — playback
  play: () => void;
  pause: () => void;
  stop: () => void;
  seekTo: (frameIdx: number) => void;
  setSpeed: (speed: PlaybackSpeed) => void;
}

let _recordUnsub: (() => void) | null = null;
let _playbackRaf = 0;
let _playbackStartTime = 0;
let _playbackStartOffset = 0;

function _captureFrame(startTime: number): RecordingFrame {
  const s = useScanStore.getState();
  return {
    timestamp: Date.now() - startTime,
    mode: s.mode,
    selectedTokenIdx: s.selectedTokenIdx,
    structuralData: s.structuralData,
    weightData: s.weightData,
    activationData: s.activationData,
    circuitData: s.circuitData,
    anomalyData: s.anomalyData,
  };
}

function _framesAreSame(a: RecordingFrame, b: RecordingFrame): boolean {
  return (
    a.mode === b.mode &&
    a.selectedTokenIdx === b.selectedTokenIdx &&
    a.structuralData === b.structuralData &&
    a.weightData === b.weightData &&
    a.activationData === b.activationData &&
    a.circuitData === b.circuitData &&
    a.anomalyData === b.anomalyData
  );
}

export const useRecordingStore = create<RecordingState>((set, get) => ({
  isRecording: false,
  frames: [],
  recordingStart: 0,
  recording: null,
  isPlaying: false,
  playbackSpeed: 1,
  currentFrameIdx: 0,
  currentFrame: null,

  startRecording: () => {
    // Stop any playback
    get().stop();

    const startTime = Date.now();
    const initialFrame = _captureFrame(startTime);

    set({
      isRecording: true,
      frames: [initialFrame],
      recordingStart: startTime,
    });

    // Subscribe to scan store changes
    if (_recordUnsub) _recordUnsub();

    let prevIsScanning = useScanStore.getState().isScanning;

    _recordUnsub = useScanStore.subscribe((state) => {
      const { isRecording, frames, recordingStart } = get();
      if (!isRecording) return;

      // Capture on scan completion
      const scanCompleted = prevIsScanning && !state.isScanning;
      prevIsScanning = state.isScanning;

      if (state.isScanning) return; // Don't capture mid-scan

      const newFrame = _captureFrame(recordingStart);
      const lastFrame = frames[frames.length - 1];

      // Skip duplicate frames
      if (lastFrame && _framesAreSame(lastFrame, newFrame) && !scanCompleted) {
        return;
      }

      set({ frames: [...frames, newFrame] });
    });

    useScanStore.getState().addLog('Recording started');
  },

  stopRecording: () => {
    if (_recordUnsub) {
      _recordUnsub();
      _recordUnsub = null;
    }

    const { frames, recordingStart } = get();
    if (frames.length === 0) {
      set({ isRecording: false });
      return;
    }

    const modelInfo = useModelStore.getState().modelInfo;
    const scanState = useScanStore.getState();

    const recording: ScanRecording = {
      version: 1,
      modelId: modelInfo?.model_id ?? 'unknown',
      prompt: scanState.prompt,
      createdAt: new Date(recordingStart).toISOString(),
      duration: Date.now() - recordingStart,
      frames,
    };

    set({
      isRecording: false,
      recording,
      currentFrameIdx: 0,
      currentFrame: frames[0] ?? null,
    });

    useScanStore.getState().addLog(`Recording stopped: ${frames.length} frames`);
  },

  saveRecording: () => {
    const { recording } = get();
    if (!recording) return;

    const json = JSON.stringify(recording, null, 2);
    const ts = new Date().toISOString().slice(0, 10);
    downloadFile(
      new Blob([json], { type: 'application/json' }),
      `nmri-recording-${recording.modelId}-${ts}.json`,
    );
    useScanStore.getState().addLog('Recording saved');
  },

  loadRecording: (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string) as ScanRecording;
        if (!data.version || !data.frames || data.frames.length === 0) {
          useScanStore.getState().addLog('Invalid recording file');
          return;
        }
        set({
          recording: data,
          currentFrameIdx: 0,
          currentFrame: data.frames[0],
          isPlaying: false,
        });
        useScanStore
          .getState()
          .addLog(`Recording loaded: ${data.frames.length} frames, ${data.modelId}`);
      } catch {
        useScanStore.getState().addLog('Failed to parse recording file');
      }
    };
    reader.readAsText(file);
  },

  play: () => {
    const { recording, isPlaying, currentFrameIdx, playbackSpeed } = get();
    if (!recording || recording.frames.length === 0 || isPlaying) return;

    const currentTimestamp = recording.frames[currentFrameIdx]?.timestamp ?? 0;
    _playbackStartTime = performance.now();
    _playbackStartOffset = currentTimestamp;

    set({ isPlaying: true });

    const tick = () => {
      const state = get();
      if (!state.isPlaying || !state.recording) return;

      const elapsed = (performance.now() - _playbackStartTime) * playbackSpeed;
      const targetTime = _playbackStartOffset + elapsed;

      // Find the frame closest to targetTime
      let frameIdx = state.currentFrameIdx;
      const frames = state.recording.frames;
      while (frameIdx < frames.length - 1 && frames[frameIdx + 1].timestamp <= targetTime) {
        frameIdx++;
      }

      if (frameIdx !== state.currentFrameIdx) {
        set({
          currentFrameIdx: frameIdx,
          currentFrame: frames[frameIdx],
        });
      }

      // Stop at end
      if (frameIdx >= frames.length - 1) {
        set({ isPlaying: false });
        return;
      }

      _playbackRaf = requestAnimationFrame(tick);
    };

    _playbackRaf = requestAnimationFrame(tick);
  },

  pause: () => {
    cancelAnimationFrame(_playbackRaf);
    set({ isPlaying: false });
  },

  stop: () => {
    cancelAnimationFrame(_playbackRaf);
    const { recording } = get();
    set({
      isPlaying: false,
      currentFrameIdx: 0,
      currentFrame: recording?.frames[0] ?? null,
    });
  },

  seekTo: (frameIdx: number) => {
    const { recording } = get();
    if (!recording) return;
    const clamped = Math.max(0, Math.min(frameIdx, recording.frames.length - 1));
    set({
      currentFrameIdx: clamped,
      currentFrame: recording.frames[clamped],
    });
  },

  setSpeed: (speed: PlaybackSpeed) => {
    const { isPlaying, currentFrameIdx, recording } = get();
    set({ playbackSpeed: speed });

    // If playing, restart timing from current position
    if (isPlaying && recording) {
      _playbackStartTime = performance.now();
      _playbackStartOffset = recording.frames[currentFrameIdx]?.timestamp ?? 0;
    }
  },
}));
