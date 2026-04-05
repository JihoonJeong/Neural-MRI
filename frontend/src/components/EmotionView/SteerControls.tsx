import { useEmotionStore } from '../../store/useEmotionStore';
import type { EmotionActivation, SAEFeatureDiff } from '../../types/emotion';

const ACCENT = '#e879a0';

export function SteerControls({ prompt }: { prompt: string }) {
  const {
    probeResult,
    steerResult,
    steerSaeResult,
    isSteering,
    selectedEmotion,
    strength,
    steer,
    steerSae,
    setSelectedEmotion,
    setStrength,
    emotionList,
  } = useEmotionStore();

  const hasProbes = probeResult !== null;
  const emotions = emotionList?.emotions ?? probeResult?.emotions ?? [];

  return (
    <div>
      {/* Header */}
      <div style={{ fontSize: 'var(--font-size-sm)', color: ACCENT, fontWeight: 'bold', letterSpacing: '0.5px', marginBottom: 12 }}>
        STEERING CONTROLS
      </div>

      {!hasProbes ? (
        <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)' }}>
          Extracting probes...
        </div>
      ) : (
        <>
          {/* Probe info */}
          <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', marginBottom: 10 }}>
            {probeResult.n_emotions} emotions @ layer {probeResult.layer_idx}
          </div>

          {/* Emotion selector */}
          <label style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
            Emotion
          </label>
          <select
            value={selectedEmotion}
            onChange={(e) => setSelectedEmotion(e.target.value)}
            style={{
              width: '100%',
              fontSize: 'var(--font-size-xs)',
              background: 'var(--bg-primary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border)',
              borderRadius: 3,
              padding: '4px 6px',
              marginBottom: 10,
            }}
          >
            {emotions.map((e) => (
              <option key={e} value={e}>{e}</option>
            ))}
          </select>

          {/* Strength slider */}
          <label style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
            Strength: {strength > 0 ? '+' : ''}{strength.toFixed(2)}
          </label>
          <input
            type="range"
            min={-0.2}
            max={0.2}
            step={0.005}
            value={strength}
            onChange={(e) => setStrength(parseFloat(e.target.value))}
            style={{ width: '100%', accentColor: ACCENT, marginBottom: 12 }}
          />

          {/* Steer button */}
          <button
            onClick={() => { steer(prompt); steerSae(prompt); }}
            disabled={isSteering || !prompt}
            style={{
              width: '100%',
              padding: '6px 12px',
              fontSize: 'var(--font-size-xs)',
              background: isSteering ? 'transparent' : `${ACCENT}20`,
              color: isSteering ? 'var(--text-secondary)' : ACCENT,
              border: `1px solid ${ACCENT}60`,
              borderRadius: 4,
              cursor: isSteering || !prompt ? 'default' : 'pointer',
              marginBottom: 16,
            }}
          >
            {isSteering ? 'STEERING...' : 'STEER'}
          </button>

          {/* Results */}
          {steerResult && (
            <SteerResult
              original={steerResult.comparison.original_text}
              steered={steerResult.comparison.steered_text}
              emotion={steerResult.comparison.emotion}
              strength={steerResult.comparison.strength}
              originalEmotions={steerResult.original_emotions}
              steeredEmotions={steerResult.steered_emotions}
            />
          )}

          {/* SAE Feature Diff (Zone E) */}
          {steerSaeResult && (
            <SAEFeatureDiffPanel features={steerSaeResult.top_changed_features} layer={steerSaeResult.sae_layer_idx} />
          )}
        </>
      )}
    </div>
  );
}

function SteerResult({
  original,
  steered,
  emotion,
  strength,
  originalEmotions,
  steeredEmotions,
}: {
  original: string;
  steered: string;
  emotion: string;
  strength: number;
  originalEmotions: EmotionActivation[];
  steeredEmotions: EmotionActivation[];
}) {
  const label = `${emotion} ${strength > 0 ? '+' : ''}${strength.toFixed(2)}`;

  // Top 8 changes
  const origMap = new Map(originalEmotions.map((e) => [e.emotion, e.activation]));
  const steerMap = new Map(steeredEmotions.map((e) => [e.emotion, e.activation]));
  const diffs = [...origMap.keys()].map((e) => ({
    emotion: e,
    orig: origMap.get(e) ?? 0,
    steer: steerMap.get(e) ?? 0,
    diff: (steerMap.get(e) ?? 0) - (origMap.get(e) ?? 0),
  })).sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff)).slice(0, 8);

  return (
    <div>
      <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', marginBottom: 4 }}>
        Original:
      </div>
      <div style={{
        fontSize: 'var(--font-size-xs)',
        color: 'var(--text-primary)',
        background: 'var(--bg-primary)',
        padding: '4px 8px',
        borderRadius: 3,
        maxHeight: 60,
        overflow: 'auto',
        lineHeight: 1.5,
        marginBottom: 8,
      }}>
        {original || '(empty)'}
      </div>

      <div style={{ fontSize: 'var(--font-size-xs)', color: ACCENT, marginBottom: 4 }}>
        Steered ({label}):
      </div>
      <div style={{
        fontSize: 'var(--font-size-xs)',
        color: 'var(--text-primary)',
        background: 'var(--bg-primary)',
        padding: '4px 8px',
        borderRadius: 3,
        border: `1px solid ${ACCENT}40`,
        maxHeight: 60,
        overflow: 'auto',
        lineHeight: 1.5,
        marginBottom: 12,
      }}>
        {steered || '(empty)'}
      </div>

      {/* Activation changes table */}
      <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--text-secondary)', marginBottom: 4 }}>
        Top activation changes:
      </div>
      <table style={{ fontSize: 'var(--font-size-xs)', width: '100%', borderCollapse: 'collapse' }}>
        <tbody>
          {diffs.map((d) => (
            <tr key={d.emotion}>
              <td style={{ padding: '1px 4px', color: 'var(--text-secondary)' }}>{d.emotion}</td>
              <td style={{ padding: '1px 4px', textAlign: 'right', color: '#6b7280' }}>{d.orig.toFixed(1)}</td>
              <td style={{ padding: '1px 2px', textAlign: 'center', color: 'var(--text-secondary)' }}>→</td>
              <td style={{ padding: '1px 4px', textAlign: 'right', color: ACCENT }}>{d.steer.toFixed(1)}</td>
              <td style={{
                padding: '1px 4px',
                textAlign: 'right',
                color: d.diff > 0 ? '#4ade80' : '#f87171',
                fontWeight: 'bold',
              }}>
                {d.diff > 0 ? '+' : ''}{d.diff.toFixed(1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ---------- Zone E: SAE Feature Diff ---------- */

function SAEFeatureDiffPanel({
  features,
  layer,
}: {
  features: SAEFeatureDiff[];
  layer: number;
}) {
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 'var(--font-size-xs)', color: '#aa88ff', fontWeight: 'bold', marginBottom: 6 }}>
        SAE FEATURE DIFF (layer {layer})
      </div>
      <table style={{ fontSize: 'var(--font-size-xs)', width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: 'var(--text-secondary)' }}>
            <th style={{ textAlign: 'left', padding: '1px 4px', fontWeight: 'normal' }}>#</th>
            <th style={{ textAlign: 'right', padding: '1px 4px', fontWeight: 'normal' }}>orig</th>
            <th style={{ textAlign: 'right', padding: '1px 4px', fontWeight: 'normal' }}>steer</th>
            <th style={{ textAlign: 'right', padding: '1px 4px', fontWeight: 'normal' }}>diff</th>
          </tr>
        </thead>
        <tbody>
          {features.slice(0, 10).map((f) => (
            <tr key={f.feature_idx}>
              <td style={{ padding: '1px 4px', color: '#aa88ff' }}>
                {f.feature_idx}
              </td>
              <td style={{ padding: '1px 4px', textAlign: 'right', color: '#6b7280' }}>
                {f.original_activation.toFixed(1)}
              </td>
              <td style={{ padding: '1px 4px', textAlign: 'right', color: '#aa88ff' }}>
                {f.steered_activation.toFixed(1)}
              </td>
              <td style={{
                padding: '1px 4px',
                textAlign: 'right',
                color: f.diff > 0 ? '#4ade80' : f.diff < 0 ? '#f87171' : 'var(--text-secondary)',
                fontWeight: 'bold',
              }}>
                {f.diff > 0 ? '+' : ''}{f.diff.toFixed(1)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
