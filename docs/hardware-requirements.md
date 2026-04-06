# Neural MRI — Hardware Requirements

## Measured VRAM Usage (RTX 4070 Ti, 12GB)

Actual peak VRAM measurements on RTX 4070 Ti (12GB VRAM, 16GB RAM).
All models loaded in fp16 via TransformerLens.

| Model | Params | Load | Probe | Steer | SAE | VRAM Peak | Headroom |
|---|---|---|---|---|---|---|---|
| **GPT-2** | 124M | OK | OK | OK | OK | **2,160 MB** | 10.1 GB |
| **GPT-2 Medium** | 355M | OK | OK | OK | N/A | **3,022 MB** | 9.0 GB |
| **Pythia 1.4B** | 1.4B | OK | OK | OK | N/A | **4,358 MB** | 7.6 GB |
| **Qwen 2.5 3B** | 3B | OK | OK | OK | N/A | **8,435 MB** | 3.6 GB |
| **Llama 3.2 3B** | 3.2B | OK | OK | OK | N/A | **8,659 MB** | 3.3 GB |
| **Llama 3.2 3B Inst** | 3.2B | OK | OK | OK | N/A | **8,659 MB** | 3.3 GB |
| **Phi-3 Mini 3.8B** | 3.8B | OK | OK | OK | N/A | **9,617 MB** | 2.4 GB |
| Gemma-2-2B | 2.6B | FAIL | — | — | — | OOM | OOM |
| Gemma-2-2B-IT | 2.6B | FAIL | — | — | — | OOM | OOM |
| Mistral 7B v0.3 | 7.2B | N/A | — | — | — | TL unsupported | — |
| Llama 3.1 8B | 8B | FAIL | — | — | — | Process killed | — |

**Notes:**
- Gemma-2-2B fails during TransformerLens weight processing (not model size itself) — requires >12GB
- Phi-3 Mini at 9.6GB is the practical limit for 12GB GPUs (2.4GB headroom)
- Mistral 7B is not supported by TransformerLens architecture
- Llama 3.1 8B crashes the process (16GB+ VRAM required)

## Compatibility Matrix

| Model | 8GB GPU | 12GB GPU | 16GB GPU | 24GB GPU | Mac MPS |
|---|---|---|---|---|---|
| GPT-2 (124M) | All | All | All | All | All |
| GPT-2 Medium (355M) | All | All | All | All | All |
| Pythia 1.4B | All | All | All | All | All |
| Qwen 2.5 3B | — | **All** | All | All | All |
| Llama 3.2 3B | — | **All** | All | All | All |
| Phi-3 Mini 3.8B | — | **All** (tight) | All | All | All |
| Gemma-2-2B | — | — | **All** | All | All |
| Mistral 7B | — | — | — | — | — (TL unsupported) |
| Llama 3.1/3 8B | — | — | — | Scan only | — |

**Legend:**
- **All**: Full functionality — scan + emotion probe + steering + sweep
- **All (tight)**: Works but <3GB headroom — close other GPU apps
- **Scan only**: Model loads, basic scans work, but steering may OOM
- **—**: Cannot load model

## Recommendations

### For Paper #6 Experiments (Emotion Steering)

| Budget | GPU | Full Steering Support |
|---|---|---|
| **Low** | 8GB (RTX 3060) | GPT-2, GPT-2 Medium, Pythia 1.4B |
| **Mid** | 12GB (RTX 4070 Ti) | + Qwen 2.5 3B, Llama 3.2 3B, Phi-3 Mini |
| **High** | 16GB+ (RTX 4080) | + Gemma-2-2B (with SAE) |
| **Full** | 24GB (RTX 4090/A10G) | + 8B models (scan only) |

### Mac (Apple Silicon, MPS)

Mac unified memory is shared between CPU and GPU. Effective VRAM is roughly 60-70% of total.

- **M1/M2 (8GB unified)**: GPT-2, GPT-2 Medium, Pythia 1.4B
- **M1/M2 Pro (16GB)**: + 3B models (all features)
- **M1/M2 Max (32GB)**: + Gemma-2-2B, all 3B models comfortably
- **M2 Ultra (64GB+)**: + 8B models

### Key Findings

1. **3B models are the sweet spot for 12GB GPUs** — Qwen, Llama 3.2, Phi-3 all run full emotion steering at 8.4-9.6GB peak
2. **Gemma-2-2B is misleading** — despite being 2.6B params, TransformerLens weight processing needs >12GB
3. **SAE adds ~800MB** (gemma-scope width_16k) — only feasible on GPT-2 or with 16GB+ VRAM
4. **7B+ models require 24GB+** — no quantization workaround with TransformerLens

### Gemma-2-2B Notes

Gemma-2-2B OOM on 12GB is caused by TransformerLens internal processing (weight folding, LayerNorm centering), not the raw model size. The model itself is ~5.2GB in fp16, but TL processing temporarily needs significantly more.

- **16GB GPU**: Should work (not yet verified)
- **12GB GPU**: FAIL — use Qwen 2.5 3B or Llama 3.2 3B instead
- **Workaround**: None within Neural-MRI. Use raw `transformers` for Gemma on 12GB.

### 7B+ Models

TransformerLens does **not** support INT4/INT8 quantization (BitsAndBytes packed weights incompatible with TL state_dict). 7B+ models require fp16 with 24GB+ VRAM.

Standalone emotion steering using `transformers` + hooks can work with INT4 on 12GB, but must run outside Neural-MRI.
