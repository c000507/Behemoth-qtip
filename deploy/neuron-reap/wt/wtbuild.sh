#!/bin/bash
# wtbuild.sh OUT SRCSET [ENV=V...]: rebuild every NEFF of SRCSET (meta flags) + extra env into OUT, plus the R=8 bound segments
# (WTB_R8="P P ..." pages, default the full ladder) built with GDN_RSPLIT=4. Small NEFFs JOBS_S (2) in parallel, segments and
# Q>=1024 MoE one at a time (RAM), both queues concurrently. -> OUT/build.txt (WTB_DONE)
OUT=$1; SET=$2; shift 2; mkdir -p $OUT
EX=$(IFS=,; echo "${*//*@*/}" | sed "s/,,*/,/g; s/^,//; s/,$//")
[ -n "$WTB_WAIT" ] && while ! grep -q "${WTB_WAIT#*:}" "${WTB_WAIT%%:*}" 2>/dev/null; do sleep 20; done   # WTB_WAIT=FILE:MARK
python3 /root/op/wtspecs.py $SET "$@" | grep -vE "^(${WTB_SKIP:-NONE}):" > $OUT/specs.txt
seg=seg_gdn_gdn_gdn_gqa_gdn_gdn_gdn_gqa_gdn_gdn_gdn_gqa_gdn_gdn_gdn_gqa_gdn_gdn_gdn_gqa
for P in ${WTB_R8-32 128 512 1024 2048 4096}; do echo "seg20_8_$P:$seg:8:1:$P:MOE=gather,GDN_SLOT=bound,KV8=1,GDN_RSPLIT=4${EX:+,$EX}" >> $OUT/specs.txt; done
grep -v ':seg_\|^moe_1_1024\|^moeg_1_1024' $OUT/specs.txt > $OUT/small.txt
grep ':seg_\|^moe_1_1024\|^moeg_1_1024' $OUT/specs.txt > $OUT/big.txt
( JOBS=${JOBS_S:-2} bash /root/op/gb.sh $OUT $(cat $OUT/small.txt) > $OUT/small.log 2>&1 ) &
JOBS=1 bash /root/op/gb.sh $OUT $(cat $OUT/big.txt) > $OUT/big.log 2>&1
wait
cat $OUT/small.log $OUT/big.log > $OUT/build.txt; echo "failed: $(grep -c FAILED $OUT/build.txt)" >> $OUT/build.txt; echo WTB_DONE >> $OUT/build.txt
