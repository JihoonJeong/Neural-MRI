# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-04-04

### SAE Provider Abstraction
- Multi-backend SAE support via adapter pattern: **SAELens** + **EleutherAI Sparsify**
- New `sae_providers.py` — `SAEProvider` ABC with `SAELensProvider` and `EleutherAIProvider`
- SAE registry now includes `provider` field to route loading per model
- Factory-based loading: `load_sae_provider()` handles both backends transparently

### Emotion Vector Analysis (NEW)
- **Emotion probe extraction** — comprehension-mode forward pass over 20 emotion categories (3 passages each) with mean subtraction
- **Emotion steering** — inject emotion vectors into residual stream during generation, compare original vs steered output
- Bundled dataset: `emotion_comprehension_texts.csv` (20 emotions + neutral baseline)
- API endpoints: `POST /api/emotion/extract-probes`, `POST /api/emotion/steer`, `GET /api/emotion/emotions`
- Inspired by [Anthropic's "Emotion Concepts" research](https://transformer-circuits.pub/2026/emotions/index.html), adapted for open-weight SLMs

### Models
- Added Llama 3.1 8B with EleutherAI SAE support (layers 23, 29 MLP)
- Added Llama 3 8B with EleutherAI SAE support (all 32 layers)
- Marked Gemma-2-2B as `tl_compat: false` due to TransformerLens segfault
- 11 models in registry (up from 5 in v0.1.0)

### Clinical Cases
- Case 1: Gemma baseline — 5-mode normal anatomy scan
- Case 2: Cross-model comparison (Gemma vs Llama vs Qwen)
- Case 3: Perturbation stress test on Gemma-2-2B
- Case 4: Base vs instruct perturbation comparison (6 models, GPU results)

### Infrastructure
- HuggingFace Spaces live demo: [Hiconcep/Neural-MRI](https://huggingface.co/spaces/Hiconcep/Neural-MRI)
- Position paper published
- GitHub Pages docs site with CODE_OF_CONDUCT and SECURITY policy

---

## [0.1.0] - 2026-03-01

First public release.

### Scan Modes
- **T1 Topology** — static architecture visualization (layers, parameters, structure)
- **T2 Tensor** — weight distribution & magnitude analysis with histograms
- **fMRI Activations** — per-layer per-token activation patterns (L2 norm, 0-1 normalized)
- **DTI Circuits** — zero-ablation importance scoring + attention pattern flow
- **FLAIR Anomaly** — bias, hallucination & anomaly detection with logit lens

### Core Features
- Token-by-token streaming via WebSocket (`/ws/stream`)
- Perturbation engine — zero, amplify, ablate with KL divergence & logit shift metrics
- Causal tracing — clean/corrupt prompt comparison with layer-by-layer recovery
- SAE feature explorer via SAE-Lens integration
- Cross-model comparison — two models side-by-side on the same prompt
- Multi-prompt comparison mode

### Models
- 5 built-in models: GPT-2, GPT-2 Medium, Pythia-1.4B, Gemma-2-2B, Llama-3.2-3B
- HuggingFace Hub dynamic model search with TransformerLens compatibility filter
- Float16 auto-casting for 1B+ models, GPU→CPU fallback
- HuggingFace token support for gated models

### UI & Visualization
- 4 layout modes: vertical, brain, network, radial
- D3.js scan canvas with pulse/glow animations (fMRI) and flow animations (DTI)
- Token stepper with chip navigation and keyboard ←→ support
- Attention heads heatmap panel
- Logit lens panel
- Diagnostic report generation (Markdown)
- Functional test battery
- Prompt templates
- Keyboard shortcuts

### Export & Recording
- WebM video export (VP9)
- Animated GIF export
- SVG/PNG snapshots (single & compare mode)
- JSON data export
- Markdown diagnostic reports
- Recording playback with adjustable speed (0.5x–4x)

### Collaboration
- Real-time session sharing via WebSocket
- Host/viewer roles with peer cursors
- Scan state synchronization

### Infrastructure
- FastAPI backend with scan result caching (LRU by model/mode/prompt)
- Docker Compose deployment (backend + nginx frontend)
- HuggingFace Spaces deployment (`Dockerfile.spaces`)
- GitHub Actions CI (backend lint/test, frontend type check/build)
- i18n — English and Korean
- Medical dark theme (DICOM viewer aesthetic)

### Documentation
- README with badges, screenshots, architecture diagram
- CONTRIBUTING.md, INSTALL.md
- GitHub issue templates (bug report, feature request)
- Pull request template
