#!/usr/bin/env bash
# End to end on a synthetic Llama-3.3-geometry-scaled checkpoint: BF16 source -> qtip_quant (QTIP,
# adaptive QTIP, adaptive NVFP4) -> CUDA runtime vs CPU reference + fidelity against the source.
# usage: tools/llama_synthetic_e2e.sh [WORKDIR]   (run from the repo root after building build-cuda
# and runtime/build-cuda)
set -euo pipefail
W=${1:-/tmp/qtip_llama_e2e}
rm -rf "$W"
build-cuda/qtip_llama_tests --keep "$W"
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} runtime/build-cuda/behemoth_cuda_llama_tests \
  --checkpoint "$W/qtip" --checkpoint "$W/adaptive-qtip" --checkpoint "$W/nvfp4" \
  --fidelity "$W/qtip" "$W/source" 0.7 --fidelity "$W/nvfp4" "$W/source" 0.3
