#!/bin/bash
# nfbuild.sh OUT [KIND...]: GDN_SLOT=bound gdn NEFFs for the Ornith v3 TP=2 set (PAGES 4097, SLOTS 9; docs/neuron-reap-nonfinite.md).
#   KIND: dec (gdn_{1,2,4,8}_1_32, default), verify (VERIFY gdn_1_{2,3,4}_32, for MTP), prefill (gdn_1_{64,256}_32).
#   Drop the output directories into the graph set in place of the same-named dynamic ones. They need the engine of
#   branch reap-nonfinite (deploy it BEFORE the graphs: an older engine mis-sizes the state tensors from a bound NEFF).
# run on the box from a source tree of that branch: SRC=/path/to/tree bash deploy/neuron-reap/nfbuild.sh /mnt/scratch/gbound dec verify
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
OUT=$(realpath -m "${1:?usage: nfbuild.sh OUT [dec|verify|prefill]...}"); shift; mkdir -p "$OUT"
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
export CONFIG=${CONFIG:-/mnt/scratch/v3prep/config.json} TP=2 PAGES=${PAGES:-4097} SLOTS=${SLOTS:-9} WEIGHTS=fp8 GDN_SLOT=bound
specs=()
for k in "${@:-dec}"; do
  case $k in
    dec) for r in 1 2 4 8; do specs+=("gdn_${r}_1_32 gdn $r 1 32"); done ;;
    verify) for q in 2 3 4; do specs+=("gdn_1_${q}_32 VERIFY=1 gdn 1 $q 32"); done ;;
    prefill) for q in 64 256; do specs+=("gdn_1_${q}_32 gdn 1 $q 32"); done ;;
    *) echo "unknown kind $k"; exit 1 ;;
  esac
done
b() { n=$1; shift; envs=(); while [[ $1 == *=* ]]; do envs+=("$1"); shift; done
  [ -f "$OUT/$n/graph.neff" ] && { echo "$n exists"; return 0; }
  t0=$(date +%s)
  ( cd "$OUT" && env "${envs[@]}" python "$SRC/neuron/qwen35_graphs.py" "$OUT/$n" "$@" > "$OUT/$n.log" 2>&1 ) && echo "$n ok $(( $(date +%s) - t0 ))s" || echo "$n FAILED (see $OUT/$n.log)"; }
export -f b; export OUT SRC
printf '%s\n' "${specs[@]}" | xargs -P "${JOBS:-4}" -I{} bash -c 'b {}'
