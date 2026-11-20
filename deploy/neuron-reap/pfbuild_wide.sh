#!/bin/bash
# pfbuild_wide.sh OUT [Q=1024]: the WIDE prefill bucket of the Ornith v3 TP=2 fp8 set (docs/neuron-qwen35-moe.md section 14), to be
# ADDED to the faster-prefill set (graphs/ornith-reap20-v3-kv8-seg20-mtp-pf; its Q=256 / Q=64 buckets stay: prompt tails).
#   gdn_1_Q_32        chunked gated delta rule, bound state slots (as the Q=256 graph, Q / 128 chunks)
#   gqa_1_Q_<P>       8-bit KV attention in key blocks x query blocks of 256 tokens (ATT_QBLOCK)
#   moe_1_Q_32        MOE=all in token sub-blocks (the overflow rerun of the grouped MoE)
#   moeg_1_Q_32       grouped MoE, idx form (scatter slots): 32 hot experts dense, 128 slots per other expert per 1024 tokens;
#                     Q=4096: four independent blocks of 1024 tokens (MOE_GBLK)
#   head_Q_1
# Serve with --prefill-chunk Q --token-budget Q+8 and BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1 (tails run in the Q=256 / 64 buckets).
# Compile memory: Q=1024 <= 3.6 GB per job; Q=4096: gqa P=4096 13 GB (11 min), P=2048 5 GB, moe 7 GB, moeg 3.3 GB (8 min) - JOBS=2 on a 32 GB box.
# run on the box: SRC=/path/to/tree CONFIG=/path/to/prepared/config.json bash deploy/neuron-reap/pfbuild_wide.sh /mnt/scratch/gw1024 1024
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
OUT=$(realpath -m "${1:?usage: pfbuild_wide.sh OUT [Q]}"); Q=${2:-1024}; mkdir -p "$OUT"
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
export CONFIG=${CONFIG:-/mnt/scratch/model/config.json} TP=2 PAGES=${PAGES:-4097} SLOTS=${SLOTS:-9} WEIGHTS=fp8
G="MOE=group MOE_GF=idx MOE_GB=sc1 MOE_CAP=128 MOE_HOT=32 MOE_GE=24 MOE_TSUB=256"
[ "$Q" -gt 1024 ] && G="$G MOE_GBLK=1024"
specs=(
  "gdn_1_${Q}_32 GDN_CHUNK=128 GDN_INV=diag GDN_SLOT=bound gdn 1 $Q 32"
  "moe_1_${Q}_32 MOE_TSUB=256 moe 1 $Q 32"
  "moeg_1_${Q}_32 $G moe 1 $Q 32"
  "head_${Q}_1 head $Q 1"
)
for P in 128 256 512 1024 2048 4096; do
  [ $((P * 16)) -lt "$Q" ] && continue                      # the bucket must hold the chunk itself
  [ "$Q" -ge 4096 ] && [ "$P" = 128 ] && continue
  [ "$Q" -lt 4096 ] && [ "$P" = 256 ] && continue           # (the production page ladder: 128, 512, 1024, 2048, 4096)
  B=256; [ "$P" = 512 ] && [ "$Q" -lt 4096 ] && B=128
  specs+=("gqa_1_${Q}_$P KV8=1 ATT_QBLOCK=256 ATT_BLOCK=$B gqa 1 $Q $P")
done
b() { n=$1; shift; envs=(); while [[ $1 == *=* ]]; do envs+=("$1"); shift; done
  [ -f "$OUT/$n/graph.neff" ] && [ -f "$OUT/$n/meta.json" ] && { echo "$n exists"; return 0; }
  t0=$(date +%s)
  ( cd "$OUT" && env "${envs[@]}" python "$SRC/neuron/qwen35_graphs.py" "$OUT/$n" "$@" > "$OUT/$n.log" 2>&1 ) && echo "$n ok $(( $(date +%s) - t0 ))s" || echo "$n FAILED (see $OUT/$n.log)"; }
export -f b; export OUT SRC
printf '%s\n' "${specs[@]}" | xargs -P "${JOBS:-2}" -I{} bash -c 'b {}'
