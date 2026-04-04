# Neural MRI Scanner — Developer Guide

> API reference, architecture, and extension guide for Neural MRI v2.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [API Reference](#api-reference)
4. [Backend Core Modules](#backend-core-modules)
5. [Frontend Architecture](#frontend-architecture)
6. [Extending Neural MRI](#extending-neural-mri)
7. [Environment & Configuration](#environment--configuration)
8. [Testing](#testing)

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│  React 18 + TypeScript + D3.js + Zustand         │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ ScanCanvas│ │ModeTabs  │ │ Panels (Perturb, │ │
│  │ (D3 viz) │ │(T1-FLAIR)│ │ CausalTrace,SAE) │ │
│  ├──────────┤ ├──────────┤ ├──────────────────┤ │
│  │  Emotion  │ │   SAE    │ │   SAE Providers  │ │
│  │  Engine   │ │ Manager  │ │ (Lens+EleutherAI)│ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│            ↕ REST + WebSocket ↕                  │
├─────────────────────────────────────────────────┤
│                   Backend                        │
│  FastAPI + TransformerLens + PyTorch             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Analysis  │ │  Model   │ │   Perturbation   │ │
│  │  Engine   │ │ Manager  │ │     Engine        │ │
│  ├──────────┤ ├──────────┤ ├──────────────────┤ │
│  │  Emotion  │ │   SAE    │ │    Battery       │ │
│  │  Engine   │ │ Providers│ │    Engine         │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│            ↕ TransformerLens ↕                   │
├─────────────────────────────────────────────────┤
│              Model Weights                       │
│  HuggingFace Hub / Local Cache                   │
└─────────────────────────────────────────────────┘
```

---

## Project Structure

```
Neural-MRI/
├── backend/
│   ├── neural_mri/
│   │   ├── main.py              # FastAPI app, singletons, lifespan
│   │   ├── config.py            # Settings (NMRI_* env vars)
│   │   ├── api/
│   │   │   ├── routes_model.py  # Model load/unload/list/search
│   │   │   ├── routes_scan.py   # T1/T2/fMRI/DTI/FLAIR scans
│   │   │   ├── routes_perturb.py # Zero/amplify/ablate/patch/causal-trace
│   │   │   ├── routes_sae.py    # SAE info/scan/support
│   │   │   ├── routes_emotion.py # Emotion extract/steer/list
│   │   │   ├── routes_battery.py # Functional test battery
│   │   │   ├── routes_report.py # Diagnostic report generation
│   │   │   └── ws_stream.py     # WebSocket token streaming
│   │   ├── core/
│   │   │   ├── model_manager.py  # Model loading/switching singleton
│   │   │   ├── model_registry.py # Built-in model definitions
│   │   │   ├── analysis_engine.py # All scan modalities
│   │   │   ├── perturbation_engine.py # Zero/amplify/ablate/patch
│   │   │   ├── emotion_engine.py # Emotion probe extraction + steering
│   │   │   ├── sae_manager.py   # SAE loading via providers
│   │   │   ├── sae_providers.py # SAELens + EleutherAI adapters
│   │   │   ├── sae_registry.py  # Model → SAE mapping
│   │   │   ├── battery_engine.py # Functional test suite
│   │   │   └── scan_cache.py    # LRU scan result cache
│   │   ├── schemas/             # Pydantic request/response models
│   │   └── data/
│   │       └── emotion_comprehension_texts.csv
│   ├── tests/
│   └── pyproject.toml
├── frontend/
│   └── src/
│       ├── App.tsx
│       ├── api/client.ts        # Backend HTTP client
│       ├── store/               # Zustand stores
│       ├── components/Panels/   # Right sidebar panels
│       ├── types/               # TypeScript interfaces
│       └── i18n/translations.ts
└── docs/
```

---

## API Reference

Base URL: `http://localhost:8000/api`

### Model Management

#### Load Model

```bash
curl -X POST /api/model/load \
  -H "Content-Type: application/json" \
  -d '{"model_id": "gpt2"}'
```

Response: `ModelInfo` with layer config, device, dtype.

#### List Models

```bash
curl /api/model/list
```

Response: Array of models with `is_loaded`, `tl_compat`, `gated` flags.

#### Search HuggingFace Hub

```bash
curl "/api/model/search?query=pythia&limit=5"
```

#### Unload Model

```bash
curl -X DELETE /api/model/unload
```

### Scan Endpoints

All scan endpoints accept `POST` with a prompt and return mode-specific data.

#### T1 — Structural

```bash
curl -X POST /api/scan/structural
```

No body needed. Returns layers, connections, parameter counts.

#### T2 — Weights

```bash
curl -X POST /api/scan/weights \
  -H "Content-Type: application/json" \
  -d '{"layer_ids": ["blocks.0.attn", "blocks.11.mlp"]}'
```

Optional `layer_ids` filter. Returns weight statistics + histograms.

#### fMRI — Activations

```bash
curl -X POST /api/scan/activation \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

Returns per-layer per-token activations (L2 norm, 0-1 normalized).

#### DTI — Circuits

```bash
curl -X POST /api/scan/circuits \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "target_token_idx": -1}'
```

Returns component importance scores + attention patterns.

#### FLAIR — Anomaly

```bash
curl -X POST /api/scan/anomaly \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The first president of Mars was"}'
```

Returns KL divergence, entropy, anomaly scores, logit lens predictions.

### Perturbation Endpoints

#### Zero-Out

```bash
curl -X POST /api/perturb/zero \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "component": "blocks.9.mlp",
    "target_token_idx": -1
  }'
```

Returns original vs perturbed predictions, logit diff, KL divergence.

#### Amplify

```bash
curl -X POST /api/perturb/amplify \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "component": "blocks.9.mlp",
    "target_token_idx": -1,
    "factor": 2.0
  }'
```

#### Ablate (Mean)

```bash
curl -X POST /api/perturb/ablate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The capital of France is",
    "component": "blocks.9.mlp",
    "target_token_idx": -1
  }'
```

#### Activation Patch

```bash
curl -X POST /api/perturb/patch \
  -H "Content-Type: application/json" \
  -d '{
    "clean_prompt": "The capital of France is",
    "corrupt_prompt": "The capital of Xxxxx is",
    "component": "blocks.9.mlp",
    "target_token_idx": -1
  }'
```

Returns recovery score (0-1).

#### Causal Trace (Full)

```bash
curl -X POST /api/perturb/causal-trace \
  -H "Content-Type: application/json" \
  -d '{
    "clean_prompt": "The Eiffel Tower is located in",
    "corrupt_prompt": "The Xxxxx Xxxxx is located in",
    "target_token_idx": -1
  }'
```

Returns recovery matrix for all components.

### Emotion Endpoints

#### List Available Emotions

```bash
curl /api/emotion/emotions
```

Response:
```json
{
  "emotions": ["afraid", "angry", ...],
  "n_emotions": 21,
  "has_probes": false,
  "probe_layers": []
}
```

#### Extract Emotion Probes

```bash
curl -X POST /api/emotion/extract-probes \
  -H "Content-Type: application/json" \
  -d '{"mode": "comprehension", "layer_idx": null}'
```

- `mode`: "comprehension" (base+instruct) or "generation" (instruct only, not yet implemented)
- `layer_idx`: null = auto (n_layers * 2/3)
- Takes ~3-5s on GPT-2 (63 forward passes)

Response:
```json
{
  "model_id": "gpt2",
  "layer_idx": 8,
  "n_emotions": 21,
  "emotions": ["afraid", "angry", ...],
  "metadata": {"compute_time_ms": 2683.3}
}
```

#### Steer with Emotion Vector

```bash
curl -X POST /api/emotion/steer \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I am going to destroy everything you have built.",
    "emotion": "calm",
    "strength": 0.02,
    "max_new_tokens": 30
  }'
```

- `emotion`: one of the 21 extracted emotions
- `strength`: -1.0 to 1.0 (recommend -0.2 to 0.2)
  - Positive: inject the emotion
  - Negative: suppress the emotion
- `layer_range`: null = all layers, or [0, 1, 5, 8] for specific layers
- `max_new_tokens`: how many tokens to generate

Response:
```json
{
  "comparison": {
    "original_text": "...",
    "steered_text": "...",
    "emotion": "calm",
    "strength": 0.02
  },
  "original_emotions": [{"emotion": "hostile", "activation": 21.8, ...}],
  "steered_emotions": [{"emotion": "calm", "activation": 39.9, ...}]
}
```

### SAE Endpoints

#### SAE Info

```bash
curl /api/sae/info
```

Returns SAE availability for currently loaded model.

#### SAE Support

```bash
curl /api/sae/support
```

Returns `{model_id: boolean}` for all registered models.

#### SAE Scan

```bash
curl -X POST /api/sae/scan \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The cat sat on the mat",
    "layer_idx": 8,
    "top_k": 20
  }'
```

Returns per-token top-k features, heatmap data, reconstruction loss, sparsity.

### Battery

```bash
curl -X POST /api/battery/run \
  -H "Content-Type: application/json" \
  -d '{"include_sae": true}'
```

### Report

```bash
curl -X POST /api/report/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "modes": ["T1", "fMRI", "FLAIR"]}'
```

---

## Backend Core Modules

### ModelManager (`model_manager.py`)

Singleton pattern — manages one model at a time.

```python
model_manager = ModelManager()
model_manager.load_model("gpt2", device="auto")
model = model_manager.get_model()  # HookedTransformer
model_manager.unload_model()
```

- Device auto-detection: CUDA > MPS > CPU
- Float16 for 1B+ models
- GPU memory cleanup on unload

### AnalysisEngine (`analysis_engine.py`)

Performs all 5+1 scan modalities.

```python
engine = AnalysisEngine(model_manager)
data = engine.scan_activation(ActivationScanRequest(prompt="Hello"))
data = engine.scan_sae(SAEScanRequest(prompt="Hello", layer_idx=8), sae_manager)
```

### EmotionEngine (`emotion_engine.py`)

Extracts emotion probes and steers generation.

```python
engine = EmotionEngine(model_manager)
probes = engine.extract_probes(ExtractProbesRequest(mode="comprehension"))
result = engine.steer(SteerRequest(prompt="...", emotion="calm", strength=0.02))
```

**How probes are computed:**
1. Forward-pass each passage from `emotion_comprehension_texts.csv`
2. Extract residual stream activation at last token, at layer ≈ 2n/3
3. Average across 3 passages per emotion
4. Subtract global mean across all emotions → emotion vector

**How steering works:**
1. Normalize emotion vector to unit length
2. For each forward pass: add `direction * strength * residual_norm` at `hook_resid_post`
3. Manual greedy token loop using `run_with_hooks()` (TransformerLens `generate()` doesn't support hooks)

### SAE Provider System (`sae_providers.py`)

Adapter pattern for multiple SAE backends.

```python
# Unified interface
provider = load_sae_provider("saelens", model_id, layer_idx, device, registry_entry)
enc = provider.encode(activations)  # → EncodeResult(top_acts, top_indices, full_acts)
recon = provider.decode_from_top(enc.top_acts, enc.top_indices)
print(provider.d_sae, provider.hook_name)
```

**Providers:**
- `SAELensProvider`: Wraps `sae_lens.SAE` — dense encode/decode
- `EleutherAIProvider`: Wraps `sparsify.Sae` — natively sparse (top-k)

### PerturbationEngine (`perturbation_engine.py`)

Stateless perturbations via TransformerLens hooks.

```python
engine = PerturbationEngine(model_manager)
result = engine.zero_out(ZeroOutRequest(prompt="...", component="blocks.9.mlp"))
trace = engine.causal_trace(CausalTraceRequest(clean_prompt="...", corrupt_prompt="..."))
```

---

## Frontend Architecture

### Store Pattern (Zustand)

```typescript
// store/useEmotionStore.ts
export const useEmotionStore = create<EmotionState>((set, get) => ({
  steerResult: null,
  isSteering: false,
  steer: async (prompt) => {
    set({ isSteering: true });
    const result = await api.emotion.steer(prompt, get().selectedEmotion, get().strength);
    set({ steerResult: result, isSteering: false });
    useScanStore.getState().addLog('Steer complete');
  },
}));
```

### API Client (`api/client.ts`)

```typescript
export const api = {
  emotion: {
    extractProbes: (mode, layerIdx?) => request<ExtractProbesResponse>('/emotion/extract-probes', {
      method: 'POST',
      body: JSON.stringify({ mode, layer_idx: layerIdx ?? null }),
    }),
    steer: (prompt, emotion, strength, maxNewTokens) => request<SteerResponse>('/emotion/steer', {
      method: 'POST',
      body: JSON.stringify({ prompt, emotion, strength, max_new_tokens: maxNewTokens }),
    }),
  },
  // ...
};
```

### Panel Component Pattern

```tsx
export function EmotionPanel() {
  const t = useLocaleStore((s) => s.t);
  const { probeResult, extractProbes, steer } = useEmotionStore();
  const isLoaded = useModelStore((s) => s.modelInfo !== null);

  if (!isLoaded) return <div>Load a model first</div>;

  return (
    <div className="px-3 py-2">
      {/* Header + controls + visualization */}
    </div>
  );
}
```

---

## Extending Neural MRI

### Adding a New Scan Mode

1. **Schema**: Add request/response models in `schemas/scan.py`
2. **Engine**: Add scan method in `analysis_engine.py`
3. **Route**: Create `routes_newmode.py` with FastAPI router
4. **Register**: Add router in `main.py`
5. **Frontend types**: Add to `types/scan.ts`
6. **API client**: Add to `api/client.ts`
7. **Store**: Create `useNewModeStore.ts`
8. **Panel**: Create `NewModePanel.tsx`
9. **i18n**: Add translation keys

### Adding a New SAE Provider

1. Create a new class implementing `SAEProvider` in `sae_providers.py`
2. Add provider type to `load_sae_provider()` factory
3. Add registry entries in `sae_registry.py` with `"provider": "your_provider"`

### Adding a New Emotion

1. Add passages to `data/emotion_comprehension_texts.csv`:
   ```
   your_emotion,1,"First passage describing the emotion..."
   your_emotion,2,"Second passage..."
   your_emotion,3,"Third passage..."
   ```
2. Restart server — the emotion is automatically available

---

## Environment & Configuration

Settings are loaded from environment variables with `NMRI_` prefix.

| Variable | Default | Description |
|---|---|---|
| `NMRI_DEFAULT_MODEL` | `gpt2` | Model to load on startup |
| `NMRI_DEVICE` | `auto` | Device: auto, cuda, mps, cpu |
| `NMRI_HF_TOKEN` | — | HuggingFace token for gated models |
| `NMRI_MAX_CACHE_ENTRIES` | `100` | Scan result cache size |
| `NMRI_CORS_ORIGINS` | `["*"]` | CORS allowed origins |

Set in `backend/.env`:
```
NMRI_DEFAULT_MODEL=gpt2
NMRI_HF_TOKEN=hf_your_token_here
```

---

## Testing

### Backend

```bash
cd backend
uv run pytest tests/ -v          # all tests
uv run pytest tests/ -x -q       # stop on first failure
uv run ruff check .              # lint
uv run ruff format --check .     # format check
```

### Frontend

```bash
cd frontend
pnpm tsc --noEmit    # type check
pnpm build           # production build
```

### Useful Test Commands

```bash
# Quick API smoke test
curl http://localhost:8000/api/model/info
curl -X POST http://localhost:8000/api/scan/structural
curl -X POST http://localhost:8000/api/emotion/extract-probes \
  -H "Content-Type: application/json" -d '{"mode":"comprehension"}'
```
