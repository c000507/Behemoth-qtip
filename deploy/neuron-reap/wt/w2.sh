#!/bin/bash
# w2: device-vs-mirror of the w1 WT MoE forms; Q=64 group sweep; Q=256 dense g41 (waits for w1 to finish: NeuronCores / RAM)
S=/mnt/scratch; O=$S/w2; mkdir -p $O; L=$S/log/w2.txt
while ! grep -q W1_DONE $S/log/w1.txt; do sleep 20; done
bash /root/op/devtest.sh $S/w1/m64_g41 $S/w1/g256_rel $S/w1/g256_g8 $S/gprod/moe_1_64_32 >> $L 2>&1
W="MOE=all,WT=1,WT_MOE=grp"
JOBS=3 bash /root/op/gb.sh $O m64_g5:moe:1:64:32:$W,WT_GRP=5 m64_g35:moe:1:64:32:$W,WT_GRP=35 m64_g52:moe:1:64:32:$W,WT_GRP=52 \
  m64_g69:moe:1:64:32:$W,WT_GRP=69 m64_g103:moe:1:64:32:$W,WT_GRP=103 m256_g41:moe:1:256:32:$W,WT_GRP=41 >> $L 2>&1
bash /root/op/lb.sh $O/m64_* $O/m256_* >> $L 2>&1
echo W2A_DONE >> $L
