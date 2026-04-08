#!/bin/bash
# Neural-MRI Cloud GPU Setup — Runpod / any Linux GPU instance
# One-liner: curl -sL https://raw.githubusercontent.com/JihoonJeong/Neural-MRI/main/scripts/cloud_setup.sh | bash
#
# Usage:
#   1. Start Runpod A40 48GB or RTX 3090 24GB (PyTorch template)
#   2. Set container disk ≥ 50GB (7B models need ~15GB each for download + load)
#   3. Open Web Terminal or SSH in
#   4. Run this script
#   5. Execute extraction

set -e

echo "═══════════════════════════════════════════════"
echo "  Neural-MRI Cloud Setup"
echo "═══════════════════════════════════════════════"

# ── Clone repo ──
if [ ! -d "Neural-MRI" ]; then
    echo "Cloning Neural-MRI..."
    git clone https://github.com/JihoonJeong/Neural-MRI.git
fi
cd Neural-MRI

# ── HF Token ──
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "⚠ HF_TOKEN not set. Gated models (Llama, Mistral) will fail."
    echo "  Set it: export HF_TOKEN=hf_your_token_here"
    echo "  Or run: huggingface-cli login"
    echo ""
fi

# ── Python environment ──
echo "Setting up Python environment..."

# Use system pip if no venv (Runpod containers usually have torch pre-installed)
pip install --quiet --upgrade pip

# ── Fix torch/torchvision compatibility ──
# RunPod templates ship torchvision/torchaudio pinned to old torch.
# Remove them first to avoid conflicts (not needed for our workload).
pip uninstall -y torchvision torchaudio 2>/dev/null || true

# Detect CUDA driver version and install matching PyTorch
CUDA_DRIVER=$(python3 -c "
import subprocess, re
try:
    out = subprocess.check_output(['nvidia-smi'], text=True)
    m = re.search(r'CUDA Version:\s+([\d.]+)', out)
    print(m.group(1) if m else 'unknown')
except: print('unknown')
" 2>/dev/null)
echo "  CUDA driver: $CUDA_DRIVER"

if echo "$CUDA_DRIVER" | grep -q "^12\.[0-4]"; then
    echo "  Installing PyTorch for CUDA 12.4..."
    pip install --quiet torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
else
    echo "  Installing latest PyTorch..."
    pip install --quiet "torch>=2.6"
fi

# Core deps (transformers<5 required for TransformerLens compatibility)
pip install --quiet \
    "transformer-lens>=2.18" \
    "transformers>=4.40,<5" \
    "accelerate>=0.28" \
    scipy \
    numpy

# EleutherAI sparsify (optional, for SAE)
pip install --quiet "git+https://github.com/EleutherAI/sparsify.git" 2>/dev/null || echo "  sparsify install skipped (optional)"

# ── Verify GPU ──
echo ""
echo "GPU check:"
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
else:
    print('  ⚠ CUDA not available! Check driver compatibility.')
"

# ── Verify TransformerLens ──
python3 -c "from transformer_lens import HookedTransformer; print('  TransformerLens: OK')" 2>/dev/null || echo "  TransformerLens: FAILED"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Quick test (GPT-2, 30s):"
echo "    python3 scripts/paper5_extract.py --models gpt2 --no-generation"
echo ""
echo "  Mistral 7B fp16 (main task):"
echo "    export HF_TOKEN=hf_your_token"
echo "    python3 scripts/paper5_extract.py \\"
echo "      --models 'mistralai/Mistral-7B-Instruct-v0.3,mistralai/Mistral-7B-v0.1'"
echo ""
echo "  Llama 3.1 8B fp16 (after Mistral):"
echo "    python3 scripts/paper5_extract.py \\"
echo "      --models 'meta-llama/Llama-3.1-8B-Instruct,meta-llama/Llama-3.1-8B'"
echo ""
echo "  Push results:"
echo "    git add scripts/paper5_output/"
echo "    git commit -m 'Paper #5: cloud fp16 extraction (Mistral/Llama 3.1)'"
echo "    git push"
echo "═══════════════════════════════════════════════"
