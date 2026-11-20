#!/bin/bash
# lbuild.sh OUT SPEC...: fewer-launch decode NEFFs for the Ornith v3 TP=2 fp8 KV8 set (docs/neuron-reap-launches.md).
#   SPEC = name:kind:R:Q:P[:ENV=V,...], e.g. seg4_1_32:seg_gdn_gdn_gdn_gqa:1:1:32  layer_gdn_moe_1_1_32:layer_gdn_moe:1:1:32
#   Settings of the production set: MOE=gather (decode), KV8=1 int8, GDN_SLOT=bound, PAGES 4097, SLOTS 9.
# run on the box: SRC=/path/to/tree bash deploy/neuron-reap/lbuild.sh /mnt/scratch/launches/g seg4_1_32:seg_gdn_gdn_gdn_gqa:1:1:32
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
OUT=$(realpath -m "${1:?usage: lbuild.sh OUT SPEC...}"); shift; mkdir -p "$OUT"
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
export CONFIG=${CONFIG:-/mnt/scratch/v3prep/config.json} TP=2 PAGES=${PAGES:-4097} SLOTS=${SLOTS:-9} WEIGHTS=fp8 GDN_SLOT=bound KV8=1 MOE=${MOE:-gather}
b() { IFS=: read -r n kind R Q P envs <<< "$1"
  [ -f "$OUT/$n/graph.neff" ] && { echo "$n exists"; return 0; }
  t0=$(date +%s)
  ( cd "$OUT" && /usr/bin/time -f "%M KB peak" -o "$OUT/$n.mem" env ${envs//,/ } python "$SRC/neuron/qwen35_graphs.py" "$OUT/$n" "$kind" "$R" "$Q" "$P" > "$OUT/$n.log" 2>&1 ) \
    && echo "$n ok $(( $(date +%s) - t0 ))s $(stat -c %s "$OUT/$n/graph.neff") B $(cat "$OUT/$n.mem" 2>/dev/null)" || echo "$n FAILED (see $OUT/$n.log)"; }
export -f b; export OUT SRC
printf '%s\n' "$@" | xargs -P "${JOBS:-2}" -I{} bash -c 'b {}'
