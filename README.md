# Neural MRI Scanner

**Model Resonance Imaging for AI Interpretability**

Neural MRI Scanner visualizes the internals of open-source LLMs like a brain MRI — mapping five medical imaging modalities (T1, T2, fMRI, DTI, FLAIR) to AI model analysis techniques.

## Scan Modes

| Mode | Full Name | What It Shows |
|------|-----------|---------------|
| **T1** | Topology Layer 1 | Static architecture — layers, parameters, structure |
| **T2** | Tensor Layer 2 | Weight distribution & magnitude |
| **fMRI** | functional Model Resonance Imaging | Activation patterns for a given prompt |
| **DTI** | Data Tractography Imaging | Information flow pathways & circuits |
| **FLAIR** | Feature-Level Anomaly Identification & Reporting | Bias, hallucination & anomaly detection |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [pnpm](https://pnpm.io/) (Node package manager)

### Backend

```bash
cd backend
uv sync
uv run uvicorn neural_mri.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
pnpm install
pnpm dev
```

Open http://localhost:5173

## Supported Models (MVP)

- GPT-2 small (124M) — P0
- GPT-2 medium (355M) — P0
- Pythia-1.4B — P0
- Gemma-2-2B — P1
- Llama-3.2-3B — P1

## Tech Stack

- **Frontend:** React 18, TypeScript, D3.js, Zustand, Tailwind CSS
- **Backend:** FastAPI, TransformerLens, PyTorch
- **Theme:** Medical Dark (DICOM viewer aesthetic)

## License

MIT
