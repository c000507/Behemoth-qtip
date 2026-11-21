#!/bin/bash
# w6: the gqa_1_64_32 WT regression (0.47 -> 2.53 ms): variants; rebench of the other slower gqa NEFFs after the build
S=/mnt/scratch; O=$S/w6; mkdir -p $O; L=$S/log/w6.txt
B="MOE=gather,KV8=1,WT=1"
JOBS=2 bash /root/op/gb.sh $O ga_same:gqa:1:64:32:$B ga_cse:gqa:1:64:32:$B,LIN_CSE=1 ga_o1:gqa:1:64:32:$B,NCC_FLAGS=-O1 ga_flat:gqa:1:64:32:$B,ATT_GATHER=flat ga_b32:gqa:1:64:32:$B,ATT_BLOCK=32 ga_p128:gqa:1:64:128:$B >> $L 2>&1
while ! grep -q WTB_DONE $S/gwt/build.txt 2>/dev/null; do sleep 20; done
bash /root/op/lb.sh $S/gprod/gqa_1_64_32 $S/gwt/gqa_1_64_32 $O/ga_* >> $L 2>&1
for n in gqa_1_1024_128 gqa_1_1024_2048 gqa_1_256_4096 gqa_4_1_4096 gqa_1_256_32 gqa_1_64_128; do bash /root/op/lb.sh $S/gprod/$n $S/gwt/$n >> $L 2>&1; done
echo W6_DONE >> $L
