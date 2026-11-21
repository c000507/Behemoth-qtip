#!/bin/bash
# w3: decode gather MoE forms at R=4/8 under WT, moeg Q=1024 relayout hot tier, dense Q=1024 grp41 (alone); bench after e1
S=/mnt/scratch; O=$S/w3; mkdir -p $O; L=$S/log/w3.txt
while ! grep -q W2A_DONE $S/log/w2.txt; do sleep 20; done
D="MOE=gather,WT=1"
JOBS=3 bash /root/op/gb.sh $O m4_rel:moe:4:1:32:$D m4_g16:moe:4:1:32:$D,WT_MOE=grp,WT_GRP=16 m8_rel:moe:8:1:32:$D m8_g16:moe:8:1:32:$D,WT_MOE=grp,WT_GRP=16 \
  m8_g32:moe:8:1:32:$D,WT_MOE=grp,WT_GRP=32 m2_rel:moe:2:1:32:$D m1_rel:moe:1:1:32:$D >> $L 2>&1
JOBS=1 bash /root/op/gb.sh $O g1024_rel:moe:1:1024:32:MOE=group,MOE_GF=idx,MOE_GB=sc1,MOE_CAP=128,MOE_HOT=32,MOE_GE=24,MOE_TSUB=256,WT=1 \
  m1024_g41:moe:1:1024:32:MOE=all,MOE_TSUB=256,WT=1,WT_MOE=grp,WT_GRP=41 >> $L 2>&1
while ! grep -q E1_DONE $S/log/e1.txt 2>/dev/null; do sleep 20; done
bash /root/op/lb.sh $S/gprod/moe_1_1_32 $O/m1_* $S/gprod/moe_2_1_32 $O/m2_* $S/gprod/moe_4_1_32 $O/m4_* $S/gprod/moe_8_1_32 $O/m8_* $S/gprod/moeg_1_1024_32 $O/g1024_* $S/w1/g1024_g8 $S/gprod/moe_1_1024_32 $O/m1024_* $S/w1/m1024_g8 >> $L 2>&1
bash /root/op/devtest.sh $O/m8_rel $O/g1024_rel $O/m1024_g41 $S/w1/g1024_g8 >> $L 2>&1
echo W3_DONE >> $L
