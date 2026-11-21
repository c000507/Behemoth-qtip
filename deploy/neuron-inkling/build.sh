#!/bin/bash
# build.sh OUT SPEC...: Inkling NEFFs (neuron/inkling_graphs.py), one per spec "NAME [ENV=V ...] KIND R Q P" or "NAME head T S".
#   SRC=<tree> CONFIG=<prepared>/config.json bash deploy/neuron-inkling/build.sh OUT "s_1_1_64 MOE=gather s 1 1 64" ...
# common env: TP=2 PAGES (pool pages incl. scratch) WEIGHTS=fp8 WT; JOBS parallel compiles (<= 4 on the trn1.2xlarge).
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
OUT=$(realpath -m "${1:?usage: build.sh OUT SPEC...}"); shift; mkdir -p "$OUT"
SRC=${SRC:-$(cd "$(dirname "$0")/../.." && pwd)}
export CONFIG=${CONFIG:?set CONFIG} TP=${TP:-2} PAGES=${PAGES:-2049} WEIGHTS=${WEIGHTS:-fp8} WT=${WT:-0}
b() { n=$1; shift; envs=(); while [[ $1 == *=* ]]; do envs+=("$1"); shift; done
  [ -f "$OUT/$n/graph.neff" ] && [ -f "$OUT/$n/meta.json" ] && { echo "$n exists"; return 0; }
  t0=$(date +%s)
  ( cd "$OUT" && env "${envs[@]}" python "$SRC/neuron/inkling_graphs.py" "$OUT/$n" "$@" > "$OUT/$n.log" 2>&1 ) \
    && echo "$n ok $(( $(date +%s) - t0 ))s" || echo "$n FAILED (see $OUT/$n.log)"; }
export -f b; export OUT SRC
printf '%s\n' "$@" | xargs -P "${JOBS:-3}" -I{} bash -c 'b {}'
