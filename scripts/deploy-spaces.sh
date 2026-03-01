#!/usr/bin/env bash
# Deploy Neural MRI to HuggingFace Spaces
# Usage: ./scripts/deploy-spaces.sh
#
# Prerequisites:
#   - HF token with WRITE access in backend/.env (NMRI_HF_TOKEN=hf_...)
#   - Or: export HF_TOKEN=hf_...
set -euo pipefail

cd "$(dirname "$0")/.."

# Load token
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN="$HF_TOKEN"
elif [ -f backend/.env ]; then
  TOKEN=$(grep '^NMRI_HF_TOKEN=' backend/.env | cut -d= -f2)
fi

if [ -z "${TOKEN:-}" ]; then
  echo "ERROR: Set HF_TOKEN env var or NMRI_HF_TOKEN in backend/.env"
  echo "Get a WRITE token at: https://huggingface.co/settings/tokens"
  exit 1
fi

echo "Deploying to HuggingFace Spaces..."

cd backend
uv run python -c "
from huggingface_hub import HfApi, SpaceHardware
import os, sys

token = '$TOKEN'
api = HfApi(token=token)
user = api.whoami()['name']
space_id = f'{user}/Neural-MRI'
project = os.path.dirname(os.getcwd())

# Create space
try:
    api.repo_info(repo_id=space_id, repo_type='space')
    print(f'Space {space_id} exists, updating...')
except Exception:
    print(f'Creating space {space_id}...')
    api.create_repo(
        repo_id=space_id, repo_type='space',
        space_sdk='docker', space_hardware=SpaceHardware.CPU_BASIC,
        private=False,
    )

# Upload repo
print('Uploading files...')
api.upload_folder(
    folder_path=project, repo_id=space_id, repo_type='space',
    ignore_patterns=[
        '.git/*', '.venv/*', 'node_modules/*', '__pycache__/*', '*.pyc',
        '.env', '.env.local', 'backend/.env', 'frontend/dist/*',
        'frontend/test-results/*', '.ruff_cache/*', '.pytest_cache/*',
        '.mypy_cache/*', 'models/*', '*.pt', '*.bin', '*.safetensors',
        '.cache/*', '.DS_Store', 'docs/neural-mri-launch-plan.md',
        'scripts/*',
    ],
)

# Upload Space README (YAML frontmatter)
readme = '''---
title: Neural MRI Scanner
emoji: 🧠
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
license: mit
short_description: Visualize LLM internals like a brain MRI
---

# Neural MRI Scanner

**Model Resonance Imaging for AI Interpretability**

Visualize the internals of open-source LLMs like a brain MRI — 5 scan modes (T1, T2, fMRI, DTI, FLAIR).

See [GitHub](https://github.com/JihoonJeong/Neural-MRI) for full documentation.
'''
api.upload_file(
    path_or_fileobj=readme.encode(), path_in_repo='README.md',
    repo_id=space_id, repo_type='space',
)

# Upload Dockerfile (rename Dockerfile.spaces -> Dockerfile)
api.upload_file(
    path_or_fileobj=open(os.path.join(project, 'Dockerfile.spaces'), 'rb'),
    path_in_repo='Dockerfile',
    repo_id=space_id, repo_type='space',
)

print(f'\\nDone! https://huggingface.co/spaces/{space_id}')
"
