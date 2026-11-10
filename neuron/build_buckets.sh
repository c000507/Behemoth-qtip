#!/bin/bash
# Compiles the standard NEFF bucket set into OUT/<name>/{graph.neff,meta.json}.
#   neuron/build_buckets.sh OUT [PAGES=2049]
#   env: MODEL=/path/to/checkpoint (geometry; default Llama 3.1 8B), TP (default 2), HEAD_FP8=1 (E4M3 lm_head heads)
# Buckets: dec{R}_{P} (R decode rows x P table pages), pre{Q}_{P} (Q-token prefill chunk), head{T} (final norm + lm_head).
# The runtime picks the cheapest bucket covering a step, so short chats use dec1_32 and 14k contexts dec1_1024; decode
# buckets with R > 1 also verify speculative drafts of a single request (R = 1 + drafts).
set -e
OUT=${1:?output directory}; export PAGES=${2:-2049}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn1
export ATTN_DEC=${ATTN_DEC:-nki}  # decode graphs of up to 8 rows and 128+ pages use the NKI decode attention kernel (nki_decode.py); ATTN_DEC=xla for the torch-op loop
export WT=${WT:-1}  # transposed FP8 projections [in, out]: 21 % faster single-stream decode than [out, in] (PERF_NOTES); WT=0 for the old layout
export MLP=${MLP:-nki}  # decode MLP through nki_mlp.mlp_fp8_tiled (tiled weight layout, ~15 % faster 24B decode); MLP=xla for the compiler's MLP
export KV=${KV:-fp8}  # E4M3 KV cache + per-token scales: half the KV HBM, NKI fp8 decode kernel (8B ctx 14k 35.0 vs 38.9 ms/token); KV=bf16 for the bf16 cache
export UNSAFE_FP8FNCAST=1  # the NKI kernels take float8_e4m3fn tensors (all codes <= 224, where e4m3fn and e4m3 agree)
mkdir -p "$OUT"
build() { n=$1; shift; [ -f "$OUT/$n/graph.neff" ] || { (cd "$OUT" && python "$HERE/trace_model.py" "$OUT/$n" "$@" > "$n.log" 2>&1); }; }
for spec in "dec1_32 1 1 32" "dec1_128 1 1 128" "dec1_512 1 1 512" "dec1_1024 1 1 1024" "dec4_128 4 1 128" "dec4_512 4 1 512" \
            "dec8_32 8 1 32" "dec8_128 8 1 128" "dec8_512 8 1 512" "dec8_1024 8 1 1024" \
            "dec16_32 16 1 32" "dec16_128 16 1 128" "dec32_32 32 1 32" "dec32_128 32 1 128" "pre64_32 1 64 32" "pre64_128 1 64 128" \
            "pre256_32 1 256 32" "pre256_128 1 256 128" "pre256_256 1 256 256" "pre256_512 1 256 512" "pre256_768 1 256 768" "pre256_1024 1 256 1024"; do
  set -- $spec; n=$1; shift; build "$n" layer "$@"
done
# SHARED_VERIFY=1: shared-page-table verify buckets dec8s_P (trace_model.py ATTN_DEC_SHARED=1, meta "shared_table"): the runtime uses them only for
# single-request verify steps (the scheduled token + its drafts); docs/neuron-verify-kernel.md. Off by default: the graph set is unchanged without it.
if [ "${SHARED_VERIFY:-0}" = 1 ]; then
  for spec in "dec8s_32 8 1 32" "dec8s_128 8 1 128" "dec8s_512 8 1 512" "dec8s_1024 8 1 1024"; do
    set -- $spec; n=$1; shift; ATTN_DEC_SHARED=1 build "$n" layer "$@"
  done
fi
for spec in "head1 1 1" "head4 4 4" "head8 8 8" "head16 16 16" "head32 32 32" "head64 64 1" "head256 256 1"; do
  set -- $spec; n=$1; t=$2; s=$3; HEAD_ROWS=$t build "$n" head "$s"
done
echo "buckets in $OUT"
