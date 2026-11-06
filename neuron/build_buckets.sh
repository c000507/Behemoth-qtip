#!/bin/bash
# Compiles the standard NEFF bucket set for the 8B on a trn1.2xlarge into OUT/<name>/{graph.neff,meta.json}.
#   neuron/build_buckets.sh OUT [PAGES=2049]
# Buckets: dec{R}_{P} (R decode rows x P table pages), pre{Q}_{P} (Q-token prefill chunk), head{T} (final norm + lm_head).
# The runtime picks the cheapest bucket covering a step, so short chats use dec1_32 and 14k contexts dec1_1024.
set -e
OUT=${1:?output directory}; export PAGES=${2:-2049}
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=/opt/aws_neuronx_venv_pytorch_2_9/bin:/opt/aws/neuron/bin:$PATH NEURON_PLATFORM_TARGET_OVERRIDE=trn1
mkdir -p "$OUT"
build() { n=$1; shift; [ -f "$OUT/$n/graph.neff" ] || { (cd "$OUT" && python "$HERE/trace_model.py" "$OUT/$n" "$@" > "$n.log" 2>&1); }; }
for spec in "dec1_32 1 1 32" "dec1_128 1 1 128" "dec1_512 1 1 512" "dec1_1024 1 1 1024" "dec4_128 4 1 128" "dec4_512 4 1 512" \
            "dec8_32 8 1 32" "dec8_128 8 1 128" "dec8_512 8 1 512" "dec8_1024 8 1 1024" "pre64_32 1 64 32" "pre64_128 1 64 128" \
            "pre256_32 1 256 32" "pre256_128 1 256 128" "pre256_512 1 256 512" "pre256_1024 1 256 1024"; do
  set -- $spec; n=$1; shift; build "$n" layer "$@"
done
for spec in "head1 1 1" "head4 4 4" "head8 8 8" "head64 64 1" "head256 256 1"; do
  set -- $spec; n=$1; t=$2; s=$3; HEAD_ROWS=$t build "$n" head "$s"
done
echo "buckets in $OUT"
