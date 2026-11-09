#!/usr/bin/env bash
# Two-GPU validation of the GLM tensor-parallel path (--tp 2) against the layer split (--layer-split 2)
# on a real glm5_next checkpoint (HF directory with config.json, safetensors and tokenizer.json):
#   1. the repository's CUDA tests (runtime and standalone GLM trees);
#   2. teacher-forced logits at every prompt position: TP2 (NCCL and copy collectives) vs layer split,
#      plus the single-GPU trace when the checkpoint fits one GPU (GLM_TP2_SINGLE=1);
#   3. decode tokens/s of TP2 vs layer split from the same prompt.
# Usage: scripts/glm_tp2_validate.sh MODEL_DIR [RESULTS_DIR]
# Environment: CUDA_ARCH (default 120), BUILD_RUNTIME / BUILD_GLM (build directories; configured and built
# when missing), CONTEXT (default 4096), MAX_TOKENS (default 64), PROMPT_FILE (text; default below),
# GLM_TP2_SINGLE=1 to add a single-GPU reference run, GLM53_RESIDENT_GIB to cap residency per GPU.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_DIR [RESULTS_DIR]" >&2
  exit 2
fi
model=$(realpath "$1")
root=$(cd "$(dirname "$0")/.." && pwd)
results=${2:-"$root/runtime/cuda/results/glm_tp2_$(date +%Y%m%d_%H%M%S)"}
arch=${CUDA_ARCH:-120}
build_runtime=${BUILD_RUNTIME:-"$root/build-runtime-cuda"}
build_glm=${BUILD_GLM:-"$root/build-glm53-cuda"}
context=${CONTEXT:-4096}
max_tokens=${MAX_TOKENS:-64}
mkdir -p "$results"
log="$results/validation.log"
exec > >(tee -a "$log") 2>&1
echo "== GLM TP2 validation $(date -Is) model=$model results=$results"
count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ "$count" -lt 2 ]]; then
  echo "Need two GPUs from --device 0; nvidia-smi lists $count" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo "(other processes on these GPUs make the timings upper bounds; note them with the results)"

# 1. Builds and tests.
if [[ ! -x "$build_runtime/behemoth_cuda" ]]; then
  cmake -S "$root/runtime" -B "$build_runtime" -G Ninja -DBEHEMOTH_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$arch"
  cmake --build "$build_runtime"
fi
if [[ ! -x "$build_glm/glm53" ]]; then
  cmake -S "$root/glm53" -B "$build_glm" -G Ninja -DGLM53_CUDA=ON -DGLM53_SERVER=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$arch"
  cmake --build "$build_glm"
fi
echo "== CTest: runtime"
OMP_NUM_THREADS=8 ctest --test-dir "$build_runtime" --output-on-failure
echo "== CTest: standalone GLM (includes TP2/TP4 tiny models against CPU and single GPU)"
OMP_NUM_THREADS=8 ctest --test-dir "$build_glm" --output-on-failure
echo "== Tiny TP2 models on two real GPUs with each collective (the CTests emulate ranks on one GPU)"
tiny=$(mktemp -d)
"$build_glm/glm53_tests" --tp-dense-fixture "$tiny/dense"
"$build_glm/glm53_tests" --tp-fixture "$tiny/moe"
"$build_glm/glm53" quantize --model "$tiny/moe" --out "$tiny/moe-nvfp4"
for ckpt in dense moe moe-nvfp4; do
  for transport in nccl copy; do
    echo "-- $ckpt, $transport"
    GLM53_TP_TRANSPORT=$transport OMP_NUM_THREADS=8 "$build_glm/glm53_gpu_tests" --model "$tiny/$ckpt" --tp 2
  done
done
rm -rf "$tiny"

# 2. Teacher-forced logits on the real checkpoint.
prompt_file=${PROMPT_FILE:-"$results/prompt.txt"}
if [[ ! -f "$prompt_file" ]]; then
  cat > "$prompt_file" <<'EOF'
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a concise assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

In two sentences, explain why the sky is blue, then name the capital of France.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The sky looks blue because air molecules scatter short wavelengths of sunlight more strongly than long ones, a process called Rayleigh scattering. The capital of France is Paris.<|eot_id|>
EOF
fi
"$build_glm/glm53" tokenize --model "$model" --input "$prompt_file" --out "$results/tokens.raw.json"
python3 - "$results/tokens.raw.json" "$results/tokens.json" <<'EOF'
import json, sys
ids = json.load(open(sys.argv[1]))["tokens"]
ids = ids[:128]  # --trace captures at most 128 positions
json.dump({"tokens": ids, "logit_positions": list(range(len(ids)))}, open(sys.argv[2], "w"))
print(f"teacher-forced prompt: {len(ids)} tokens")
EOF
run_trace() { # name, args...
  local name=$1; shift
  echo "-- trace $name: $*"
  rm -f "$results/trace_$name.json"
  "$build_runtime/behemoth_cuda" --model "$model" --tokens "$results/tokens.json" --context "$context" \
    --max-tokens 0 --trace "$results/trace_$name.json" "$@" 2> >(grep -v '^prefill ' >&2)
}
run_trace layer_split --layer-split 2
GLM53_TP_TRANSPORT=nccl run_trace tp2_nccl --tp 2
GLM53_TP_TRANSPORT=copy run_trace tp2_copy --tp 2
if [[ "${GLM_TP2_SINGLE:-0}" == 1 ]]; then
  run_trace single --device 0
fi
echo "== Logit comparison (reference: layer split)"
compare="$root/scripts/compare_glm_traces.py"
status=0
python3 "$compare" "$results/trace_layer_split.json" "$results/trace_tp2_nccl.json" --label "TP2 (NCCL) vs layer split" || status=1
python3 "$compare" "$results/trace_layer_split.json" "$results/trace_tp2_copy.json" --label "TP2 (copy) vs layer split" || status=1
python3 "$compare" "$results/trace_tp2_nccl.json" "$results/trace_tp2_copy.json" --label "TP2 copy vs NCCL (expect identical)" || status=1
if [[ -f "$results/trace_single.json" ]]; then
  python3 "$compare" "$results/trace_single.json" "$results/trace_tp2_nccl.json" --label "TP2 (NCCL) vs single GPU" || status=1
  python3 "$compare" "$results/trace_single.json" "$results/trace_layer_split.json" --label "layer split vs single GPU (expect identical)" || status=1
fi

# 3. Decode throughput.
echo "== Decode tokens/s ($max_tokens tokens, greedy, Llama-3 chat template)"
decode() { # name, args...
  local name=$1; shift
  echo "-- decode $name: $*"
  "$build_runtime/behemoth_cuda" --model "$model" --prompt "Explain in two sentences why the sky is blue." \
    --context "$context" --max-tokens "$max_tokens" "$@" > "$results/decode_$name.txt" \
    2> >(grep -v '^prefill ' | tee "$results/decode_$name.log" >&2)
  echo "   output: $(head -c 300 "$results/decode_$name.txt")"
  grep -o '[0-9.]* tok/s' "$results/decode_$name.log" | sed "s/^/   $name: /"
}
decode layer_split --layer-split 2
# Second run of each configuration after the first warms the resident weights and caches.
decode layer_split_warm --layer-split 2
decode tp2_nccl --tp 2
decode tp2_nccl_warm --tp 2
GLM53_TP_TRANSPORT=copy decode tp2_copy --tp 2
{
  echo "# GLM TP2 validation $(date -Is)"
  echo "model: $model"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
  echo
  grep -h 'vs layer split\|vs NCCL\|vs single' "$log" || true
  echo
  grep -h 'tok/s' "$results"/decode_*.log | sed 's/^/decode: /' || true
  grep -h '^GLM GPU' "$results"/decode_tp2_nccl.log || true
} > "$results/SUMMARY.md"
echo "== Summary written to $results/SUMMARY.md"
cat "$results/SUMMARY.md"
exit $status
