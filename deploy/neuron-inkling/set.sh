#!/bin/bash
# set.sh OUT: the Inkling 2 Mini serving graph set (WT layout, 32k context: PAGES=2049), built one compile at a time
# (JOBS=1: the trn1.2xlarge has 32 GB of RAM) with deploy/neuron-inkling/build.sh. See docs/neuron-inkling.md section 7.
#   SRC=<tree> CONFIG=<model>/config.json bash deploy/neuron-inkling/set.sh OUT
#   env: DEC_R="1 2 4" (decode rows), DEC_P="64 512 2048", PRE_Q="256 1024" (prefill chunks), PRE_P="64 512 2048", JOBS
# Decode: 6-visit segments (attention + gathered MoE), far keys of the global visit in blocks of 256 pages.
# Prefill: every visit split into an attention and a MoE NEFF, loop-pass kinds only (S / G: the engine binds zero message
# gates on first-pass visits); attention in query blocks of 256 tokens, far keys in blocks of 128 pages; MoE grouped
# (routed experts only: MOE_HOT experts on every token, the others on MOE_CAP token slots; Q >= 1024) with the dense MoE
# as the overflow rerun.
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: set.sh OUT}
DEC_R=${DEC_R:-"1 2 4"}; DEC_P=${DEC_P:-"64 512 2048"}; PRE_Q=${PRE_Q:-"256 1024"}; PRE_P=${PRE_P:-"64 512 2048"}
specs=()
dec() { r=$1; rt=$([ $r -gt 1 ] && echo pad64 || echo topk)
  for p in $DEC_P; do for k in sssssg SSSSSG; do
    specs+=("seg_${k}_${r}_1_$p MOE=gather ROUTE=$rt ATT_BLOCK=256 seg_$k $r 1 $p"); done; done
  specs+=("head_${r}_${r} head $r $r"); }
first=1
for r in $DEC_R; do [ $first = 1 ] && dec $r; first=0; done     # the first decode bucket, then prefill, then the other rows
for q in $PRE_Q; do
  qb=$([ $q -gt 256 ] && echo "ATT_QBLOCK=256 MOE_TSUB=256" || echo "")
  specs+=("Sa_1_$q PART=attn MOE=all $qb S 1 $q 64")
  specs+=("Sm_1_$q PART=moe MOE=all WT_MOE=grp WT_GRP=8 $qb S 1 $q 64")
  # grouped MoE only where it pays (wide chunks: at Q = 256 it measured the same as the dense form)
  [ $q -ge 1024 ] && specs+=("Smg_1_$q PART=moe MOE=group MOE_HOT=${MOE_HOT:-16} MOE_CAP=${MOE_CAP:-128} MOE_GE=${MOE_GE:-24} $GRP_EXTRA WT_MOE=grp WT_GRP=8 $qb S 1 $q 64")
  for p in $PRE_P; do specs+=("Ga_1_${q}_$p PART=attn MOE=all ATT_BLOCK=128 $qb G 1 $q $p"); done
  specs+=("head_${q}_1 head $q 1")
done
first=1
for r in $DEC_R; do [ $first = 0 ] && dec $r; first=0; done
WT=1 PAGES=${PAGES:-2049} JOBS=${JOBS:-1} bash "$HERE/build.sh" "$OUT" "${specs[@]}"
