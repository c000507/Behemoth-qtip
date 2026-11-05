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
  --checkpoint "$W/qtip" --checkpoint "$W/adaptive-qtip" --checkpoint "$W/nvfp4" --checkpoint "$W/adaptive-mixed" \
  --checkpoint "$W/fp8kv" --checkpoint "$W/adaptive-fp8kv-qtip" --checkpoint "$W/adaptive-fp8kv-nvfp4" \
  --fidelity "$W/fp8kv" "$W/source" 0.7 \
  --fidelity "$W/qtip" "$W/source" 0.7 --fidelity "$W/nvfp4" "$W/source" 0.3

# whole-model KL tool: KL of a model against its own cached log-probs must be 0, against another quantization > 0
K=runtime/build-cuda/behemoth_cuda_model_kl
python3 -c "
import random,array;random.seed(1);array.array('i',[random.randrange(320) for _ in range(4*64)]).tofile(open('$W/kl_tokens.i32','wb'))"
$K --model "$W/qtip" --tokens "$W/kl_tokens.i32" --seq-len 64 --ref-dir "$W/kl_ref" --write-ref
self=$($K --model "$W/qtip" --tokens "$W/kl_tokens.i32" --seq-len 64 --ref-dir "$W/kl_ref" 2>/dev/null | sed -n 's/^KL.*= \([0-9.]*\) +-.*/\1/p')
other=$($K --model "$W/adaptive-mixed" --tokens "$W/kl_tokens.i32" --seq-len 64 --ref-dir "$W/kl_ref" 2>/dev/null | sed -n 's/^KL.*= \([0-9.]*\) +-.*/\1/p')
echo "model_kl: self $self other $other"
python3 -c "import sys; s,o=float('$self'),float('$other'); sys.exit(0 if s<1e-6 and o>1e-3 else 1)"
