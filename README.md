<p align="center">
  <h1 align="center">Neural MRI Scanner</h1>
  <p align="center"><strong>Model Resonance Imaging for AI Interpretability</strong></p>
</p>

<p align="center">
  <a href="https://github.com/JihoonJeong/Neural-MRI/actions/workflows/ci.yml"><img src="https://github.com/JihoonJeong/Neural-MRI/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/node-20+-339933.svg?logo=node.js&logoColor=white" alt="Node 20+">
  <a href="https://huggingface.co/spaces/Hiconcep/Neural-MRI"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Live%20Demo-yellow.svg" alt="HF Spaces"></a>
</p>

---

Neural MRI Scanner visualizes the internals of open-source LLMs like a **brain MRI** — mapping five medical imaging modalities to AI model analysis techniques. Feed a model and a prompt, get back a full diagnostic scan of what's happening inside.

<p align="center">
  <img src="docs/screenshots/fmri-scan.png" alt="Neural MRI — fMRI activation scan" width="800">
</p>

## Scan Modes

| Mode | Full Name | What It Shows |
|------|-----------|---------------|
| **T1** | Topology Layer 1 | Static architecture — layers, parameters, structure |
| **T2** | Tensor Layer 2 | Weight distribution & magnitude |
| **fMRI** | functional Model Resonance Imaging | Activation patterns for a given prompt |
| **DTI** | Data Tractography Imaging | Information flow pathways & circuits |
| **FLAIR** | Feature-Level Anomaly Identification & Reporting | Bias, hallucination & anomaly detection |

<p align="center">
  <img src="docs/screenshots/dti-scan.png" alt="DTI circuit tracing" width="400">
  <img src="docs/screenshots/flair-scan.png" alt="FLAIR anomaly detection" width="400">
</p>

## Features

- **5 scan modes** with real-time visualization (T1, T2, fMRI, DTI, FLAIR)
- **Token-by-token streaming** via WebSocket — watch activations unfold live
- **Perturbation engine** — zero, amplify, or ablate individual components and measure impact (KL divergence, logit shift)
- **Causal tracing** — clean/corrupt prompt comparison with layer-by-layer recovery scores
- **SAE integration** — multi-backend Sparse Autoencoder feature analysis (SAELens + EleutherAI)
- **Emotion vector analysis** — extract emotion probes from model internals and steer behavior with emotion vectors (inspired by [Anthropic's emotion concepts research](https://transformer-circuits.pub/2026/emotions/index.html))
- **Cross-model comparison** — run two models side-by-side on the same prompt
- **4 layout modes** — vertical, brain, network, radial
- **Recording & export** — WebM video, animated GIF, SVG/PNG snapshots, JSON data, Markdown reports
- **Real-time collaboration** — share scan sessions with peers via WebSocket
- **HuggingFace Hub search** — dynamically discover and load TransformerLens-compatible models
- **i18n** — English and Korean
- **Medical dark theme** — DICOM viewer aesthetic with CRT scan effects

## Quick Start

### Local

```bash
# Backend
cd backend
uv sync
uv run uvicorn neural_mri.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:5173

### Docker

```bash
docker compose up --build
```

Open http://localhost

> See [INSTALL.md](INSTALL.md) for detailed setup instructions including GPU configuration and environment variables.

## Supported Models

| Model | Params | SAE | 8GB | 12GB | 16GB+ | Mac MPS |
|-------|--------|-----|-----|------|-------|---------|
| GPT-2 | 124M | SAELens | All | All | All | All |
| GPT-2 Medium | 355M | — | All | All | All | All |
| Pythia-1.4B | 1.4B | — | All | All | All | All |
| Gemma-2-2B | 2.6B | SAELens | Scan | All | All | All |
| Llama-3.2-3B | 3B | — | Scan | Scan | All | Scan |
| Qwen-2.5-3B | 3B | — | Scan | Scan | All | Scan |
| Phi-3 Mini | 3.8B | — | — | Scan | All | Scan |
| Llama-3.1-8B | 8B | EleutherAI | — | — | — | — |
| Mistral-7B | 7.2B | — | — | — | — | — |

> **All** = scan + SAE + emotion steering. **Scan** = model loads, scans work, steering may OOM. **—** = cannot load.
>
> 7B+ models require 24GB+ VRAM (fp16). See [Hardware Requirements](docs/hardware-requirements.md) for details.

Additional models can be loaded dynamically via **HuggingFace Hub search** — any model with a TransformerLens-compatible architecture works.

## Emotion Vector Analysis

Neural MRI can extract and manipulate emotion representations inside language models, based on [Anthropic's emotion concepts research](https://transformer-circuits.pub/2026/emotions/index.html).

```bash
# 1. Extract emotion probes (20 emotions x 3 passages, comprehension mode)
POST /api/emotion/extract-probes  {"mode": "comprehension"}

# 2. Steer model behavior with an emotion vector
POST /api/emotion/steer  {
  "prompt": "I'm going to destroy everything.",
  "emotion": "calm",
  "strength": 0.05
}
```

The engine supports **comprehension mode** (works on both base and instruct models) using pre-written emotional passages, and returns side-by-side original vs steered outputs with full emotion activation profiles.

Available emotions: happy, sad, calm, desperate, afraid, angry, proud, guilty, nervous, hopeful, brooding, gloomy, reflective, enthusiastic, hostile, loving, exasperated, blissful, anxious, grateful, neutral.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   Frontend                       │
│  React 18 + TypeScript + D3.js + Zustand         │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ ScanCanvas│ │ModeTabs  │ │ Panels (Perturb, │ │
│  │ (D3 viz) │ │(T1-FLAIR)│ │ CausalTrace,SAE) │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│            ↕ REST + WebSocket ↕                  │
├─────────────────────────────────────────────────┤
│                   Backend                        │
│  FastAPI + TransformerLens + PyTorch             │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │  Scanner  │ │  Model   │ │   Perturbation   │ │
│  │  Engine   │ │ Registry │ │     Engine        │ │
│  ├──────────┤ ├──────────┤ ├──────────────────┤ │
│  │  Emotion  │ │   SAE    │ │   SAE Providers  │ │
│  │  Engine   │ │ Manager  │ │ (Lens+EleutherAI)│ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
│            ↕ TransformerLens ↕                   │
├─────────────────────────────────────────────────┤
│              Model Weights                       │
│  HuggingFace Hub / Local Cache                   │
└─────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, TypeScript, D3.js, Zustand, Tailwind CSS v3 |
| Backend | FastAPI, TransformerLens, PyTorch, SAE-Lens, EleutherAI Sparsify |
| Infra | Docker Compose, GitHub Actions CI |
| Theme | Medical Dark (DICOM viewer aesthetic) |

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)
