#!/usr/bin/env bash
# Launch the Qwen3-ASR Python sidecar that the Rust asr-server proxies to
# when running with `--backend python` (the default).
#
# Usage:
#   ./python_sidecar/run.sh                              # vllm backend, 0.6B on cuda:0, port 8090
#   ASR_BACKEND=transformers ./python_sidecar/run.sh     # eager HF backend
#   ASR_MODEL_DIR=third_party/Qwen3-ASR/Qwen3-ASR-1.7B ./python_sidecar/run.sh
#   ASR_DEVICE=cuda:1 ASR_PORT=8090 ./python_sidecar/run.sh
#
# Env vars (see python_sidecar/server.py for the full list):
#   ASR_BACKEND    default vllm     (vllm | transformers)
#   ASR_MODEL_DIR  default third_party/Qwen3-ASR/Qwen3-ASR-0.6B
#   ASR_DEVICE     default cuda:0   (vllm requires CUDA; index is honored)
#   ASR_DTYPE      default float16  (bf16 unsupported on Tesla T4 / sm_75)
#   ASR_HOST       default 0.0.0.0
#   ASR_PORT       default 8090
#   vLLM-only: ASR_GPU_MEM_UTIL (0.85), ASR_MAX_MODEL_LEN (16384), ASR_ENFORCE_EAGER (false)
set -euo pipefail

# Resolve repo root from this script's location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate the conda env if one isn't already active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "qwen3-asr" ]]; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate qwen3-asr
fi

exec python python_sidecar/server.py \
  --backend   "${ASR_BACKEND:-vllm}" \
  --model-dir "${ASR_MODEL_DIR:-third_party/Qwen3-ASR/Qwen3-ASR-0.6B}" \
  --device    "${ASR_DEVICE:-cuda:0}" \
  --dtype     "${ASR_DTYPE:-float16}" \
  --host      "${ASR_HOST:-0.0.0.0}" \
  --port      "${ASR_PORT:-8090}"
