# Cloud GPU Extraction Guide (Runpod)

## Quick Start

```bash
# 1. Start Runpod instance (RTX 3090 24GB, PyTorch template)
# 2. SSH in
# 3. Set HF token + run setup
export HF_TOKEN=hf_your_token_here
curl -sL https://raw.githubusercontent.com/JihoonJeong/Neural-MRI/main/scripts/cloud_setup.sh | bash

# 4. Extract (Mistral 7B, ~5 min)
cd Neural-MRI
python3 scripts/paper5_extract.py \
  --models "mistralai/Mistral-7B-Instruct-v0.3,mistralai/Mistral-7B-v0.1"

# 5. Push results
git add scripts/paper5_output/
git commit -m "Paper #5: Mistral 7B fp16 cloud extraction"
git push

# 6. Stop instance (save money!)
```

## Model Queue

| Priority | Model | Est. Time | VRAM |
|---|---|---|---|
| **1** | Mistral 7B v0.3 Instruct | ~5 min | ~15 GB |
| **2** | Mistral 7B v0.1 Base | ~5 min | ~15 GB |
| **3** | Llama 3.1 8B Instruct | ~5 min | ~17 GB |
| **4** | Llama 3.1 8B Base | ~5 min | ~17 GB |

Total: ~20 minutes GPU time.

## Costs

- RTX 3090 24GB on-demand: ~$0.44/hr
- Estimated total: $0.15-0.30 for all 4 models
- Setup time: ~5 min (download deps + first model)

## Troubleshooting

### HF Token
```bash
# Check if set
echo $HF_TOKEN

# Set for current session
export HF_TOKEN=hf_your_token

# Persistent (add to ~/.bashrc)
echo 'export HF_TOKEN=hf_your_token' >> ~/.bashrc
```

### Git Push from Cloud
```bash
# Configure git (first time only)
git config user.name "JihoonJeong"
git config user.email "jihoon.jeong@gmail.com"

# Use token for push
git remote set-url origin https://<github_token>@github.com/JihoonJeong/Neural-MRI.git
```

### CUDA OOM
If a model fails with OOM:
- Check `nvidia-smi` for other processes
- The script auto-frees VRAM between models
- RTX 3090 has 24GB — Mistral 7B fp16 needs ~15GB, should work

### Partial Failure
The script preserves partial results. If model 2 of 4 fails:
- Models 1's data is already saved
- Fix the issue and re-run with `--models` for just the failed model
