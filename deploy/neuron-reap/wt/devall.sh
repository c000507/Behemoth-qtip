#!/bin/bash
# devall.sh SET [prod SET]: device-vs-mirror for every sub-block NEFF of SET (gdn / gqa / moe; heads, segments, mtp are checked end to end)
S=/mnt/scratch; G=$1; L=$S/log/devall_$(basename $G).txt
while ! grep -q GB_DONE $G/small.log 2>/dev/null; do sleep 30; done
for d in $G/*/; do n=$(basename $d); case $n in gdn_*|gqa_*|moe_*|moeg_*) [ -f $d/graph.neff ] && bash /root/op/devtest.sh $d >> $L 2>&1;; esac; done
echo DEVALL_DONE >> $L
