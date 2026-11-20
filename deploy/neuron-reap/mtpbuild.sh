#!/bin/bash
# mtpbuild.sh OUT: MTP head + verify buckets (Q 2..4) for the Ornith v3 TP=2 KV8 PAGES=4097 set; main set linked from v3gkv8_4k
. /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
S=/mnt/scratch; cd $S/mtpsrc; OUT=$1; mkdir -p $OUT; L=$OUT/build.log
for b in $S/v3gkv8_4k/*/; do ln -sfn $(readlink -f $b) $OUT/$(basename $b); done
export CONFIG=$S/v3prep/config.json TP=2 PAGES=4097 SLOTS=9 WEIGHTS=fp8 MOE=gather KV_CODE=int8
specs=()
for q in 2 3 4; do specs+=("VERIFY=1 gdn 1 $q 32" "moe 1 $q 32"); for p in 32 128 512 1024 2048 4096; do specs+=("KV8=1 gqa 1 $q $p"); done; done
for q in 1 2 3 4; do specs+=("mtp 1 $q 32"); done
specs+=("head 3 3")
build() {
  set -- $1; envs=(); while [[ $1 == *=* ]]; do envs+=("$1"); shift; done
  if [ $1 = head ]; then n=head_$2_$3; else n=$1_$2_$3_$4; fi
  [ -f $OUT/$n/graph.neff ] && { echo "$n exists"; return 0; }
  rm -rf $OUT/$n; t0=$(date +%s)
  ( cd $OUT && env "${envs[@]}" nice -n 10 python $S/mtpsrc/neuron/qwen35_graphs.py $OUT/$n "$@" > $OUT/$n.log 2>&1 ) && echo "$n ok $(( $(date +%s)-t0 ))s" || echo "$n FAILED"
}
export -f build; export OUT S
printf '%s\n' "${specs[@]}" | xargs -P ${JOBS:-3} -I{} bash -c 'build "{}"' >> $L 2>&1
echo MTPBUILD_DONE >> $L
