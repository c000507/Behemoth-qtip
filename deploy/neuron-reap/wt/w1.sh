#!/bin/bash
# w1: prefill MoE under WT, forms (Q=64 / 256 dense + grouped), then Q=1024 alone; layer_bench vs the prod NEFFs
S=/mnt/scratch; O=$S/w1; mkdir -p $O; L=$S/log/w1.txt
W="WT=1"
JOBS=4 bash /root/op/gb.sh $O \
  m64_rel:moe:1:64:32:MOE=all,$W \
  m64_g4:moe:1:64:32:MOE=all,$W,WT_MOE=grp,WT_GRP=4 \
  m64_g8:moe:1:64:32:MOE=all,$W,WT_MOE=grp,WT_GRP=8 \
  m64_g16:moe:1:64:32:MOE=all,$W,WT_MOE=grp,WT_GRP=16 \
  m64_g41:moe:1:64:32:MOE=all,$W,WT_MOE=grp,WT_GRP=41 \
  g256_g8:moe:1:256:32:MOE=group,MOE_CAP=32,MOE_HOT=24,MOE_GE=69,$W,WT_MOE=grp,WT_GRP=8 \
  g256_rel:moe:1:256:32:MOE=group,MOE_CAP=32,MOE_HOT=24,MOE_GE=69,$W \
  m256_g8:moe:1:256:32:MOE=all,$W,WT_MOE=grp,WT_GRP=8 >> $L 2>&1
bash /root/op/lb.sh $S/gprod/moe_1_64_32 $O/m64_* $S/gprod/moeg_1_256_32 $O/g256_* $S/gprod/moe_1_256_32 $O/m256_* >> $L 2>&1
echo W1A_DONE >> $L
JOBS=1 bash /root/op/gb.sh $O \
  g1024_g8:moe:1:1024:32:MOE=group,MOE_GF=idx,MOE_GB=sc1,MOE_CAP=128,MOE_HOT=32,MOE_GE=24,MOE_TSUB=256,$W,WT_MOE=grp,WT_GRP=8 \
  m1024_g8:moe:1:1024:32:MOE=all,MOE_TSUB=256,$W,WT_MOE=grp,WT_GRP=8 >> $L 2>&1
bash /root/op/lb.sh $S/gprod/moeg_1_1024_32 $O/g1024_* $S/gprod/moe_1_1024_32 $O/m1024_* >> $L 2>&1
echo W1_DONE >> $L
