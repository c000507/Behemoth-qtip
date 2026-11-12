#!/bin/bash
# Qwen3.5 MoE NEFF buckets (qwen35_graphs.py) for one config and TP.
#   CONFIG=config.json TP=2 neuron/qwen35_build_buckets.sh OUT
# env: DECODE_R ("1": decode rows), PREFILL_Q ("" : chunk tokens, none by default), P ("4": table pages per bucket; several -> one gqa
#      graph each), PAGES (KV pages incl. scratch), SLOTS (state slots incl. scratch), WEIGHTS fp8|bf16, MOE_DEC all|gather (decode MoE),
#      JOBS (parallel compiles), FUSED=1 (layer_gdn_moe / layer_gqa_moe instead of the three sub-block kinds), HEAD=1 (head graphs)
set -e
OUT=$(realpath -m "${1:?usage: qwen35_build_buckets.sh OUT}"); mkdir -p "$OUT"
H=$(cd "$(dirname "$0")" && pwd)
: "${CONFIG:?set CONFIG to the config.json}"
DECODE_R=${DECODE_R:-"1"}; PREFILL_Q=${PREFILL_Q:-""}; P=${P:-4}; JOBS=${JOBS:-2}
export CONFIG TP=${TP:-2} PAGES=${PAGES:-65} SLOTS=${SLOTS:-9} WEIGHTS=${WEIGHTS:-fp8}
if [ "${FUSED:-0}" = 1 ]; then KINDS="layer_gdn_moe layer_gqa_moe"; else KINDS="gdn gqa moe"; fi
specs=()
for p in $P; do
  for r in $DECODE_R; do for k in $KINDS; do [[ $k == *gqa* ]] || [ "$p" = "${P%% *}" ] && specs+=("$k $r 1 $p dec"); done; done
  for q in $PREFILL_Q; do for k in $KINDS; do [[ $k == *gqa* ]] || [ "$p" = "${P%% *}" ] && specs+=("$k 1 $q $p pre"); done; done
done
if [ "${HEAD:-0}" = 1 ]; then for r in $DECODE_R; do specs+=("head $r $r"); done; for q in $PREFILL_Q; do specs+=("head $q 1"); done; fi
build() {
  set -- $1
  if [ "$1" = head ]; then n=head_$2_$3; args="head $2 $3"; else n=$1_$2_$3_$4; args="$1 $2 $3 $4"; fi
  [ -f "$OUT/$n/graph.neff" ] && return 0
  if [[ $1 == *moe ]] && [ "$5" = pre ]; then export MOE=all ROUTE=topk; else export MOE=${MOE_DEC:-all} ROUTE=${ROUTE_DEC:-topk}; fi   # ROUTE_DEC=rank: decode buckets without torch.topk (required for R > 1 at E = 512)
  t0=$(date +%s)
  ( cd "$OUT" && python "$H/qwen35_graphs.py" "$OUT/$n" $args > "$OUT/$n.log" 2>&1 ) && echo "$n ok $(( $(date +%s) - t0 ))s" || echo "$n FAILED (see $OUT/$n.log)"
}
export -f build; export OUT H
printf '%s\n' "${specs[@]}" | xargs -P "$JOBS" -I{} bash -c 'build "{}"'
