#!/bin/bash
# lbcmp.sh A B [names...]: layer_bench of every NEFF name in both sets (ms per launch, best of 8) -> log/lbcmp.txt "name A B"
S=/mnt/scratch; A=$1; B=$2; shift 2; L=$S/log/lbcmp.txt
names=${*:-$(ls $B | grep -v '\.' )}
for n in $names; do [ -f $A/$n/graph.neff ] && [ -f $B/$n/graph.neff ] || { echo "$n only in one set" >> $L; continue; }
  a=$(bash /root/op/lb.sh -n 32 $A/$n | awk '{print $2}'); b=$(bash /root/op/lb.sh -n 32 $B/$n | awk '{print $2}'); echo "$n $a $b" >> $L; done
echo LBCMP_DONE >> $L
