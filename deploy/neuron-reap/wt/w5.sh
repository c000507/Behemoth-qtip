#!/bin/bash
# w5: WT_MOE=tr (transposed-orientation MOE=all) at Q=64 / 256 / 1024
S=/mnt/scratch; O=$S/w5; mkdir -p $O; L=$S/log/w5.txt
W="MOE=all,WT=1,WT_MOE=tr"
JOBS=3 bash /root/op/gb.sh $O m64_tr1:moe:1:64:32:$W,WT_GRP=1 m64_tr5:moe:1:64:32:$W,WT_GRP=5 m64_tr41:moe:1:64:32:$W,WT_GRP=41 \
  m256_tr1:moe:1:256:32:$W,WT_GRP=1 m256_tr5:moe:1:256:32:$W,WT_GRP=5 m256_tr41:moe:1:256:32:$W,WT_GRP=41 >> $L 2>&1
bash /root/op/lb.sh $O/m64_* $O/m256_* >> $L 2>&1
echo W5A_DONE >> $L
JOBS=1 bash /root/op/gb.sh $O m1024_t256tr1:moe:1:1024:32:$W,WT_GRP=1,MOE_TSUB=256 m1024_tr1:moe:1:1024:32:$W,WT_GRP=1 >> $L 2>&1
bash /root/op/lb.sh $O/m1024_* >> $L 2>&1
echo W5_DONE >> $L
