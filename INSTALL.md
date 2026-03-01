# Installation Guide

Detailed setup instructions for Neural MRI Scanner.

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.11+ | [python.org](https://www.python.org/downloads/) |
| Node.js | 20+ | [nodejs.org](https://nodejs.org/) |
| uv | latest | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| pnpm | 10+ | `npm install -g pnpm` |

## GPU Requirements

Neural MRI Scanner runs model inference locally. GPU acceleration is recommended for models >1B parameters.

| Setup | Device Flag | Notes |
|-------|------------|-------|
| NVIDIA GPU | `cuda` | Requires CUDA 11.8+ and PyTorch with CUDA support |
| Apple Silicon | `mps` | M1/M2/M3/M4 — automatic, no extra setup |
| CPU only | `cpu` | Works for GPT-2 (124M). Larger models will be slow |
| Auto-detect | `auto` | Default — picks best available device |

Set via environment variable:
```bash
NMRI_DEVICE=auto  # auto | cuda | mps | cpu
```

**Memory guidelines:**
- GPT-2 (124M): ~500MB — runs on any device
- GPT-2 Medium (355M): ~1.5GB
- Pythia-1.4B: ~3GB (loaded in float16)
- Gemma-2-2B / Llama-3.2-3B: ~6GB (loaded in float16)

## Backend Setup

```bash
cd backend

# Install dependencies
uv sync

# Install dev dependencies (for testing/linting)
uv sync --extra dev

# Start the server
uv run uvicorn neural_mri.main:app --reload --port 8000
```

The API will be available at http://localhost:8000. API docs at http://localhost:8000/docs.

## Frontend Setup

```bash
cd frontend

# Install dependencies
pnpm install

# Start dev server
pnpm dev
```

The UI will be available at http://localhost:5173. It proxies `/api` requests to the backend on port 8000.

## Docker Setup

The simplest way to run both services:

```bash
docker compose up --build
```

This starts:
- Backend on port 8000
- Frontend on port 80

Open http://localhost to use the app.

To use GPU in Docker, update `docker-compose.yml` to add NVIDIA runtime and set `NMRI_DEVICE=cuda`.

## Environment Variables

Copy the example file and customize:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `NMRI_DEFAULT_MODEL` | `gpt2` | Model to load on startup |
| `NMRI_DEVICE` | `auto` | Compute device (`auto`, `cuda`, `mps`, `cpu`) |
| `NMRI_ENVIRONMENT` | `local` | Environment profile (`local`, `docker`, `huggingface`) |
| `NMRI_CORS_ORIGINS` | `["http://localhost:5173"]` | Allowed CORS origins (JSON array) |
| `NMRI_HF_TOKEN` | _(empty)_ | HuggingFace token for gated models |
| `VITE_API_BASE_URL` | _(empty)_ | Override API base URL in frontend |

## HuggingFace Token (Gated Models)

Gemma-2-2B and Llama-3.2-3B are gated models that require a HuggingFace token:

1. Create a token at https://huggingface.co/settings/tokens
2. Accept model access on each model's page:
   - https://huggingface.co/google/gemma-2-2b
   - https://huggingface.co/meta-llama/Llama-3.2-3B
3. Set the token:
   ```bash
   # In .env file
   NMRI_HF_TOKEN=hf_your_token_here

   # Or export directly
   export NMRI_HF_TOKEN=hf_your_token_here
   ```

## Troubleshooting

### Model loading fails

- **"Out of memory"**: Try a smaller model (GPT-2) or set `NMRI_DEVICE=cpu` to avoid GPU memory limits. Models >1B are loaded in float16 automatically.
- **"Token required"**: Gated models (Gemma, Llama) need a HuggingFace token. See above.
- **Slow first load**: Model weights are downloaded from HuggingFace Hub on first use. Subsequent loads use the local cache at `~/.cache/huggingface/`.

### TransformerLens compatibility

- `transformers` is pinned to `<5` for TransformerLens compatibility. If you see import errors, check your transformers version: `uv run python -c "import transformers; print(transformers.__version__)"`.
- Not all HuggingFace models are supported — only architectures that TransformerLens can hook into (GPT-2, GPT-Neo, Pythia, Gemma, Llama, Mistral, etc.).

### Frontend build errors

- Make sure you're using Node.js 20+ and pnpm 10+.
- If `pnpm install` fails, try deleting `node_modules` and `pnpm-lock.yaml`, then run `pnpm install` again.
- TypeScript errors: run `pnpm tsc --noEmit` to see type errors in isolation.

### Docker issues

- **Port conflict**: Make sure ports 80 and 8000 are free, or change them in `docker-compose.yml`.
- **Model cache**: The compose file mounts `~/.cache/huggingface` to reuse downloaded models. Ensure this directory exists.
