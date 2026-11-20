#!/bin/bash
# pfbuild.sh OUT: the faster PREFILL NEFFs of the Ornith v3 TP=2 fp8 set (docs/neuron-qwen35-moe.md section 13).
#   gdn_1_{256,64}_32    chunked gated delta rule (GDN_CHUNK, GDN_INV=diag) with engine-bound state slots (GDN_SLOT=bound)
#   gqa_1_256_{512..4096} 8-bit KV attention in key blocks with an online softmax (ATT_BLOCK)
#   moeg_1_256_32        MoE by expert groups (MOE=group); needs the bucket's plain moe_1_256_32 next to it (overflow rerun)
# Drop the output directories into a copy of the production graph set (graphs/ornith-reap20-v3-kv8-seg20-mtp): the gdn / gqa
# directories REPLACE the same-named ones, moeg_1_256_32 is ADDED. The grouped MoE needs the engine of branch ornith-prefill
# (an older engine sees two moe NEFFs for the bucket); the gdn / gqa NEFFs run on the production engine as they are.
# run on the box: SRC=/path/to/tree CONFIG=/path/to/prepared/config.json bash deploy/neuron-reap/pfbuild.sh /mnt/scratch/gpf
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
OUT=$(realpath -m "${1:?usage: pfbuild.sh OUT}"); mkdir -p "$OUT"
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
export CONFIG=${CONFIG:-/mnt/scratch/model/config.json} TP=2 PAGES=${PAGES:-4097} SLOTS=${SLOTS:-9} WEIGHTS=fp8
specs=(
  "gdn_1_256_32 GDN_CHUNK=128 GDN_INV=diag GDN_SLOT=bound gdn 1 256 32"
  "gdn_1_64_32 GDN_CHUNK=64 GDN_INV=diag GDN_SLOT=bound gdn 1 64 32"
  "gqa_1_256_512 KV8=1 ATT_BLOCK=128 gqa 1 256 512"
  "gqa_1_256_1024 KV8=1 ATT_BLOCK=256 gqa 1 256 1024"
  "gqa_1_256_2048 KV8=1 ATT_BLOCK=256 gqa 1 256 2048"
  "gqa_1_256_4096 KV8=1 ATT_BLOCK=256 gqa 1 256 4096"
  "moeg_1_256_32 MOE=group MOE_CAP=32 MOE_HOT=24 MOE_GE=69 moe 1 256 32"
)
b() { n=$1; shift; envs=(); while [[ $1 == *=* ]]; do envs+=("$1"); shift; done
  [ -f "$OUT/$n/graph.neff" ] && [ -f "$OUT/$n/meta.json" ] && { echo "$n exists"; return 0; }
  t0=$(date +%s)
  ( cd "$OUT" && env "${envs[@]}" python "$SRC/neuron/qwen35_graphs.py" "$OUT/$n" "$@" > "$OUT/$n.log" 2>&1 ) && echo "$n ok $(( $(date +%s) - t0 ))s" || echo "$n FAILED (see $OUT/$n.log)"; }
export -f b; export OUT SRC
printf '%s\n' "${specs[@]}" | xargs -P "${JOBS:-4}" -I{} bash -c 'b {}'
