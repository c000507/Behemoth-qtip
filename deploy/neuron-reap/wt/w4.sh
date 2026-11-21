#!/bin/bash
# w4: dense MOE=all Q=256 (the grouped MoE's overflow rerun) under WT: group sizes and token sub-blocks
S=/mnt/scratch; O=$S/w4; mkdir -p $O; L=$S/log/w4.txt
W="MOE=all,WT=1,WT_MOE=grp"
JOBS=2 bash /root/op/gb.sh $O m256_g5:moe:1:256:32:$W,WT_GRP=5 m256_g35:moe:1:256:32:$W,WT_GRP=35 m256_t64g5:moe:1:256:32:$W,WT_GRP=5,MOE_TSUB=64 \
  m256_t128g5:moe:1:256:32:$W,WT_GRP=5,MOE_TSUB=128 m256_t64g41:moe:1:256:32:$W,WT_GRP=41,MOE_TSUB=64 m256_rel:moe:1:256:32:MOE=all,WT=1 >> $L 2>&1
while ! grep -q W3_DONE $S/log/w3.txt 2>/dev/null; do sleep 20; done
bash /root/op/lb.sh $O/m256_* >> $L 2>&1
echo W4_DONE >> $L
