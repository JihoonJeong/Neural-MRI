# Cloud GPU Extraction Guide (Runpod)

## Quick Start

```bash
# 1. Start Runpod instance (A40 48GB recommended, PyTorch template)
#    ⚠ Set container disk ≥ 50GB (7B models need ~15GB each)
# 2. Open Web Terminal
# 3. Set HF token + run setup
export HF_TOKEN=hf_your_token_here
curl -sL https://raw.githubusercontent.com/JihoonJeong/Neural-MRI/main/scripts/cloud_setup.sh | bash

# 4. Extract one model at a time (clear cache between runs to save disk)
cd Neural-MRI
python3 scripts/paper5_extract.py --models 'mistralai/Mistral-7B-v0.1' --no-generation
rm -rf /root/.cache/huggingface/hub

python3 scripts/paper5_extract.py --models 'meta-llama/Llama-3.1-8B' --no-generation
rm -rf /root/.cache/huggingface/hub

# 5. Push results
git config user.name "JihoonJeong"
git config user.email "jihoon.jeong@gmail.com"
git remote set-url origin https://JihoonJeong:<github_pat>@github.com/JihoonJeong/Neural-MRI.git
git add scripts/paper5_output/
git commit -m "Paper #5: cloud fp16 extraction"
git push

# 6. Stop or Terminate instance (save money!)
```

## Model Queue

All 4 cloud models completed (2026-04-08, A40 48GB):

| Model | Backend | Time | Regime | Status |
|---|---|---|---|---|
| Mistral 7B v0.1 Base | TransformerLens | 273s | surgical | ✅ Done |
| Mistral 7B v0.3 Instruct | HF fallback | 108s | N/A | ✅ Done |
| Llama 3.1 8B Base | TransformerLens | 284s | surgical | ✅ Done |
| Llama 3.1 8B Instruct | TransformerLens | 260s | surgical | ✅ Done |

## Costs

- A40 48GB on-demand: ~$0.41/hr (cheaper than RTX 3090, more VRAM)
- Actual total: ~$0.45 for all 4 models (~1 hour including setup)
- Disk storage (stopped pod): $0.01/hr

## Known Issues & Fixes

### PyTorch / CUDA Driver Mismatch
The setup script now auto-detects CUDA driver version and installs matching PyTorch.
If you still hit issues:
```bash
# Check CUDA driver version
nvidia-smi | grep "CUDA Version"

# For CUDA 12.4 driver, use torch for cu124
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### torchvision / torchaudio Conflicts
The setup script removes these automatically. If you see `torchvision::nms` errors:
```bash
pip uninstall torchvision torchaudio -y
```

### transformers v5 Breaks TransformerLens
TransformerLens requires `transformers<5`. The setup script pins this automatically.

### Disk Full (No space left on device)
7B models need ~15GB disk for download + GPU load. With 50GB container disk:
- Clear HF cache between models: `rm -rf /root/.cache/huggingface/hub`
- Check space: `df -h /`

### GPU Reassigned After Stop
RunPod may reassign your GPU if you stop the pod. Choose "Automatically migrate your Pod data" when restarting. Note: container packages will be reset — re-run the setup script.

### HF Token
```bash
export HF_TOKEN=hf_your_token
```
Required for gated models (Llama, Mistral). Get one at https://huggingface.co/settings/tokens

### Git Push from Cloud
```bash
git config user.name "JihoonJeong"
git config user.email "jihoon.jeong@gmail.com"
# Use GitHub Personal Access Token (classic), repo scope
git remote set-url origin https://JihoonJeong:<github_pat>@github.com/JihoonJeong/Neural-MRI.git
git push
```
