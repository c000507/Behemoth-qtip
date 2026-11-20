#!/bin/bash
# gb.sh OUT SPEC...   SPEC = name:kind:R:Q:P[:ENV=V,ENV=V...]   (head: name:head:T:S)
# Ornith v3 TP=2 fp8 NEFF builds with the production base settings (PAGES 4097, SLOTS 9); per-spec env on top. JOBS parallel.
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
OUT=$(realpath -m "${1:?usage}"); shift; mkdir -p "$OUT"
SRC=${SRC:-/mnt/scratch/src}
export CONFIG=${CONFIG:-/mnt/scratch/model/config.json} TP=2 PAGES=${PAGES:-4097} SLOTS=${SLOTS:-9} WEIGHTS=fp8
b() { IFS=: read -r n kind R Q P envs <<< "$1"
  [ -f "$OUT/$n/graph.neff" ] && [ -f "$OUT/$n/meta.json" ] && { echo "$n exists"; return 0; }
  t0=$(date +%s)
  ( cd "$OUT" && /usr/bin/time -f "%M KB peak" -o "$OUT/$n.mem" env ${envs//,/ } python "$SRC/neuron/qwen35_graphs.py" "$OUT/$n" "$kind" "$R" "$Q" $P > "$OUT/$n.log" 2>&1 ) \
    && echo "$n ok $(( $(date +%s) - t0 ))s $(stat -c %s "$OUT/$n/graph.neff") B $(cat "$OUT/$n.mem" 2>/dev/null)" || echo "$n FAILED $(( $(date +%s) - t0 ))s (see $OUT/$n.log)"; }
export -f b; export OUT SRC
printf '%s\n' "$@" | xargs -P "${JOBS:-4}" -I{} bash -c 'b {}'
echo GB_DONE
