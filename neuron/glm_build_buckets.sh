#!/bin/bash
# GLM-5.3 NEFF buckets (glm_graphs.py) for one prepared model geometry and parallel layout.
#   MODEL=/path/to/checkpoint-or-prepared TP=2 EP=2 neuron/glm_build_buckets.sh OUT
# env: DECODE_R ("1 4": decode rows), PREFILL_Q ("16": chunk tokens), P ("8": table pages per bucket; several -> one mla
#      graph each), PAGES (KV pages incl. scratch), SLOTS (KDA state slots incl. scratch), WEIGHTS fp8|bf16, KV bf16|fp8,
#      MOE all|gather (decode graphs; prefill graphs always run every local expert), JOBS (parallel compiles),
#      FUSED=1 (one NEFF per layer kind, attention + MLP: half the launches per step)
set -e
OUT=${1:?usage: glm_build_buckets.sh OUT}; mkdir -p "$OUT"
H=$(cd "$(dirname "$0")" && pwd)
: "${MODEL:?set MODEL to the checkpoint (or prepared) directory with config.json}"
DECODE_R=${DECODE_R:-"1 4"}; PREFILL_Q=${PREFILL_Q:-16}; P=${P:-8}; JOBS=${JOBS:-2}
export MODEL TP=${TP:-2} PAGES=${PAGES:-129} SLOTS=${SLOTS:-9} WEIGHTS=${WEIGHTS:-fp8} KV=${KV:-bf16}
KINDS=$(FUSED=${FUSED:-0} python3 -c "
import json,os; c=json.load(open('$MODEL/config.json')); c=c.get('text_config',c)
n=c['num_hidden_layers']; L=c.get('mlp_layer_types') or ['dense' if l < c.get('first_k_dense_replace', n) else 'sparse' for l in range(n)]
A=['kda' if t=='linear_attention' else 'mla' for t in c['layer_types']]; M=['moe' if m=='sparse' else 'dense' for m in L]
print(' '.join(sorted({f'layer_{a}_{m}' for a,m in zip(A,M)} if os.environ['FUSED']=='1' else set(A)|set(M))))")
specs=()
for p in $P; do
  for r in $DECODE_R; do for k in $KINDS; do [[ $k == *mla* ]] || [ "$p" = "${P%% *}" ] && specs+=("$k $r 1 $p dec"); done; done
  for q in $PREFILL_Q; do for k in $KINDS; do [[ $k == *mla* ]] || [ "$p" = "${P%% *}" ] && specs+=("$k 1 $q $p pre"); done; done
done
for r in $DECODE_R; do specs+=("head $r $r"); done
for q in $PREFILL_Q; do specs+=("head $q 1"); done
build() {
  set -- $1
  if [ "$1" = head ]; then n=head_$2_$3; args="head $2 $3"; else n=$1_$2_$3_$4; args="$1 $2 $3 $4"; fi
  [ -f "$OUT/$n/graph.neff" ] && return 0
  if [[ $1 == *moe ]] && [ "$5" = pre ]; then export MOE=all; else export MOE=${MOE_DEC:-${MOE:-all}}; fi
  ( cd "$OUT" && python "$H/glm_graphs.py" "$OUT/$n" $args > "$OUT/$n.log" 2>&1 ) && echo "$n ok" || echo "$n FAILED (see $OUT/$n.log)"
}
export -f build; export OUT H
printf '%s\n' "${specs[@]}" | xargs -P "$JOBS" -I{} bash -c 'build "{}"'
