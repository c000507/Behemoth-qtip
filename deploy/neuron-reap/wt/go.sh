#!/bin/bash
# go.sh: after the build + fallbacks: sets/wt = gwt; load test (HBM) of prod and WT with MTP; quiet segment layer_bench; then meas.sh
S=/mnt/scratch; L=$S/log/go.txt
while ! grep -q FB_DONE $S/log/fb.txt 2>/dev/null || ! grep -q W6_DONE $S/log/w6.txt 2>/dev/null; do sleep 20; done
bash /root/op/mkset.sh wt $S/gwt >> $L
for t in p w; do G=$S/gprod; B=$S/BUILD; [ $t = w ] && { G=$S/sets/wt; B=$S/BUILD2; }
  CHUNK=1024 BUDGET=1032 bash /root/op/srv.sh ${t}_load $G $B BEHEMOTH_NEURON_QWEN35_BUCKET_FIT=1 >> $L 2>&1
  echo "== $t load" >> $L; cat $S/run/${t}_load/mem.txt >> $L; grep -E "resident|MTP: |Server failed|NEFFs" $S/run/${t}_load/server.log | grep -v "Qwen35 NEFF " | head -5 >> $L
  pkill -f neuron_server; sleep 3; done
echo LOAD_DONE >> $L
bash /root/op/lbcmp.sh $S/gprod $S/sets/wt $(ls $S/gwt | grep '^seg') moe_1_256_32 moe_1_1024_32 gqa_1_64_32 gqa_1_256_4096 gqa_4_1_4096 gqa_1_1024_2048 >> $L 2>&1
echo GO_LB_DONE >> $L
[ "${GO_MEAS:-1}" = 1 ] && bash /root/op/meas.sh
