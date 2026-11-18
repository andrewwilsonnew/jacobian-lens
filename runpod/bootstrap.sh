#!/usr/bin/env bash
# Runs ON the pod: install uv, create the venv, and optionally prefetch a model.
#   bash runpod/bootstrap.sh [model]     e.g. bash runpod/bootstrap.sh qwen3.6-27b
set -euo pipefail
cd "$(dirname "$0")/.."

# Keep the HF cache and uv cache on the /workspace volume so they survive a pod stop.
# HF's Xet transfer backend stalled indefinitely on this pod's network (2026-11-18, three
# times, open connections with no data), so use plain HTTP downloads instead.
export HF_HOME=/workspace/hf-cache UV_CACHE_DIR=/workspace/uv-cache HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_DISABLE_XET=1
grep -q "HF_HOME=/workspace" ~/.bashrc 2>/dev/null ||
  echo 'export HF_HOME=/workspace/hf-cache UV_CACHE_DIR=/workspace/uv-cache HF_HUB_DISABLE_XET=1 PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv sync
# Optional fast kernels for Qwen3.5/3.6's linear-attention layers (pure Triton, CUDA only).
uv pip install flash-linear-attention || echo "flash-linear-attention not installed; using slower reference kernels"
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

if [ $# -ge 1 ]; then
  uv run python -c "
from huggingface_hub import hf_hub_download, snapshot_download
from common import LENS_REPO, MODELS
spec = MODELS['$1']
print(snapshot_download(spec.hf_id))
print(hf_hub_download(LENS_REPO, spec.lens_file, revision=spec.lens_revision))
"
fi
