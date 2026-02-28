import { useRef } from 'react';
import { useRecordingStore } from '../store/useRecordingStore';
import { useLocaleStore } from '../store/useLocaleStore';
import type { TranslationKey } from '../i18n/translations';
import type { PlaybackSpeed } from '../types/recording';

function formatTime(ms: number): string {
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

export function RecordingBar() {
  const {
    isRecording,
    frames,
    recordingStart,
    recording,
    isPlaying,
    playbackSpeed,
    currentFrameIdx,
    startRecording,
    stopRecording,
    saveRecording,
    loadRecording,
    play,
    pause,
    stop,
    seekTo,
    setSpeed,
  } = useRecordingStore();

  const { t } = useLocaleStore();
  const fileRef = useRef<HTMLInputElement>(null);

  const hasRecording = recording !== null && recording.frames.length > 0;
  const isPlaybackMode = hasRecording && !isRecording;
  const totalFrames = recording?.frames.length ?? 0;
  const elapsed = isRecording ? Date.now() - recordingStart : 0;
  const playbackTime = recording?.frames[currentFrameIdx]?.timestamp ?? 0;
  const totalTime = recording?.duration ?? 0;

  const btnStyle = (active = false, danger = false) => ({
    background: active
      ? danger ? 'rgba(255,68,102,0.2)' : 'rgba(0,255,170,0.15)'
      : 'rgba(255,255,255,0.04)',
    border: `1px solid ${
      active
        ? danger ? 'rgba(255,68,102,0.5)' : 'rgba(0,255,170,0.3)'
        : 'var(--border)'
    }`,
    color: active
      ? danger ? '#ff4466' : 'var(--accent-active)'
      : 'var(--text-secondary)',
    padding: '2px 8px',
    fontSize: 'var(--font-size-xs)',
    fontFamily: 'var(--font-primary)',
    cursor: 'pointer',
    borderRadius: 3,
    letterSpacing: '0.5px',
  });

  const handleFileLoad = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) loadRecording(file);
    e.target.value = '';
  };

  const speeds: PlaybackSpeed[] = [0.5, 1, 2, 4];

  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5"
      style={{
        borderTop: '1px solid var(--border)',
        background: isRecording ? 'rgba(255,68,102,0.03)' : undefined,
      }}
    >
      {/* Recording controls */}
      {!isPlaybackMode && (
        <>
          <button
            onClick={isRecording ? stopRecording : startRecording}
            style={btnStyle(isRecording, true)}
          >
            {isRecording ? (
              <span>
                <span style={{ color: '#ff4466' }}>{'\u25CF'} </span>
                {t('recording.stop' as TranslationKey)}
                <span style={{ marginLeft: 6, color: 'var(--text-secondary)' }}>
                  {formatTime(elapsed)} | {frames.length}f
                </span>
              </span>
            ) : (
              t('recording.rec' as TranslationKey)
            )}
          </button>

          {!isRecording && (
            <>
              <button
                onClick={() => fileRef.current?.click()}
                style={btnStyle()}
              >
                {t('recording.load' as TranslationKey)}
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".json"
                onChange={handleFileLoad}
                style={{ display: 'none' }}
              />
            </>
          )}
        </>
      )}

      {/* Playback controls */}
      {isPlaybackMode && (
        <>
          <button onClick={stop} style={btnStyle()}>
            {'\u23F9'}
          </button>
          <button
            onClick={isPlaying ? pause : play}
            style={btnStyle(isPlaying)}
          >
            {isPlaying ? '\u23F8' : '\u25B6'}
          </button>

          {/* Timeline slider */}
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            value={currentFrameIdx}
            onChange={(e) => {
              if (isPlaying) pause();
              seekTo(Number(e.target.value));
            }}
            style={{
              flex: 1,
              minWidth: 80,
              maxWidth: 200,
              accentColor: 'var(--accent-active)',
              cursor: 'pointer',
            }}
          />

          {/* Speed selector */}
          <select
            value={playbackSpeed}
            onChange={(e) => setSpeed(Number(e.target.value) as PlaybackSpeed)}
            style={{
              background: 'var(--bg-surface)',
              border: '1px solid var(--border)',
              color: 'var(--text-primary)',
              fontSize: 'var(--font-size-xs)',
              fontFamily: 'var(--font-primary)',
              padding: '1px 4px',
              borderRadius: 3,
            }}
          >
            {speeds.map((s) => (
              <option key={s} value={s}>{s}x</option>
            ))}
          </select>

          {/* Time display */}
          <span
            style={{
              fontSize: 'var(--font-size-xs)',
              color: 'var(--text-secondary)',
              whiteSpace: 'nowrap',
            }}
          >
            {formatTime(playbackTime)}/{formatTime(totalTime)}
            <span style={{ marginLeft: 4, opacity: 0.6 }}>
              {currentFrameIdx + 1}/{totalFrames}f
            </span>
          </span>

          {/* Save / Close */}
          <button onClick={saveRecording} style={btnStyle()}>
            {t('recording.save' as TranslationKey)}
          </button>
          <button
            onClick={() => {
              stop();
              useRecordingStore.setState({ recording: null, currentFrame: null });
            }}
            style={btnStyle()}
          >
            {'\u2715'}
          </button>
        </>
      )}
    </div>
  );
}
