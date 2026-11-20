#!/bin/bash
# wset.sh NAME Q DIR [MOEG]: graph set sets/NAME = gprod (the 256-chunk -pf set) + the Q-token prefill bucket from DIR
# (gdn_1_Q_32, moe_1_Q_32, gqa_1_Q_<P>, head_Q_1 and, when named, the grouped MoE directory MOEG as moeg_1_Q_32)
cd /mnt/scratch; N=$1; Q=$2; D=$3; M=$4
a="gdn_1_${Q}_32=$D/gdn_1_${Q}_32 moe_1_${Q}_32=$D/moe_1_${Q}_32 head_${Q}_1=$D/head_${Q}_1"
for p in 128 256 512 1024 2048 4096; do [ -d $D/gqa_1_${Q}_$p ] && a="$a gqa_1_${Q}_$p=$D/gqa_1_${Q}_$p"; done
[ -n "$M" ] && a="$a moeg_1_${Q}_32=$M"
bash /root/op/mkset.sh $N gprod $a
