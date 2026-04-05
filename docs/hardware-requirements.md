# Neural MRI — Hardware Requirements

## VRAM Estimates (fp16)

All values in MB. Estimates based on model parameters and operational overhead.

| Model | Params | Load | Scan | SAE | Steer | Total |
|---|---|---|---|---|---|---|
| **GPT-2** | 124M | 248 | 372 | 275 | 446 | **722** |
| **GPT-2 Medium** | 355M | 710 | 1,065 | — | 1,278 | **1,278** |
| **Pythia 1.4B** | 1.4B | 2,800 | 4,200 | — | 5,040 | **5,040** |
| **Gemma-2-2B** | 2.6B | 5,200 | 7,800 | 351 | 9,360 | **9,711** |
| **Llama 3.2 3B** | 3.2B | 6,400 | 9,600 | — | 11,520 | **11,520** |
| **Qwen 2.5 3B** | 3B | 6,000 | 9,000 | — | 10,800 | **10,800** |
| **Phi-3 Mini** | 3.8B | 7,600 | 11,400 | — | 13,680 | **13,680** |
| **Mistral 7B** | 7.2B | 14,400 | 21,600 | — | 25,920 | **25,920** |
| **Llama 3.1 8B** | 8B | 16,000 | 24,000 | — | 28,800 | **28,800** |

**Column definitions:**
- **Load**: Model weights in fp16
- **Scan**: Load + TransformerLens activation cache (T1/T2/fMRI/DTI/FLAIR)
- **SAE**: Additional VRAM for SAE encoder+decoder weights
- **Steer**: Scan + generation KV cache (emotion steering)
- **Total**: All features running simultaneously (worst case)

## Compatibility Matrix

| Model | 8GB GPU | 12GB GPU | 16GB GPU | 24GB GPU | Mac MPS |
|---|---|---|---|---|---|
| GPT-2 (124M) | All | All | All | All | All |
| GPT-2 Medium (355M) | All | All | All | All | All |
| Pythia 1.4B | All | All | All | All | All |
| **Gemma-2-2B** | Scan only | **All** | All | All | All |
| **Llama 3.2 3B** | Scan only | Scan only | **All** | All | Scan only |
| **Qwen 2.5 3B** | Scan only | Scan only | **All** | All | Scan only |
| **Phi-3 Mini 3.8B** | — | Scan only | **All** | All | Scan only |
| Mistral 7B | — | — | — | Scan only | — |
| Llama 3.1/3 8B | — | — | — | Scan only | — |

**Legend:**
- **All**: Full functionality — scan + SAE + emotion steering + sweep
- **Scan only**: Model loads, basic scans work, but steering/SAE may OOM
- **—**: Cannot load model

## Recommendations

### For Paper #6 Experiments (Emotion Steering)

| Budget | GPU | Recommended Models |
|---|---|---|
| **Low** | 8GB (RTX 3060) | GPT-2, Pythia 1.4B |
| **Mid** | 12GB (RTX 4070 Ti) | + Gemma-2-2B (full), Llama 3.2 3B (scan) |
| **High** | 16GB (RTX 4080) | + Llama 3.2 3B, Qwen 2.5 3B, Phi-3 (full) |
| **Full** | 24GB (RTX 4090/A10G) | + 7B models (scan only), all 3B (full) |

### Mac (Apple Silicon, MPS)

- **M1/M2 (8GB unified)**: GPT-2, GPT-2 Medium, Pythia 1.4B
- **M1/M2 Pro (16GB)**: + Gemma-2-2B (full), 3B models (scan)
- **M1/M2 Max (32GB)**: + 3B models (full), 7B (scan)
- **M2 Ultra (64GB+)**: All models full functionality

### Gemma-2-2B + SAE Notes

Gemma-2-2B with gemma-scope SAE (width_16k, d_sae=16384) requires ~351MB additional VRAM.
- **12GB GPU**: Works but tight — close to OOM if other processes use VRAM
- **8GB GPU**: SAE will fail — model alone uses ~5.2GB + scan cache
- **Recommendation**: On 12GB, reduce `max_new_tokens` to 20 when combining SAE + steering

### 7B+ Models

TransformerLens does **not** support INT4/INT8 quantization for 7B+ models (BitsAndBytes packed weights incompatible with TL state_dict). 7B+ models require fp16 with sufficient VRAM.

Standalone emotion steering scripts using `transformers` + hooks can work with INT4 on 12GB, but must run outside Neural-MRI.
