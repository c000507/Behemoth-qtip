#!/bin/bash
# lbm.sh [-n steps] DIR...: lb.sh per NEFF + the HBM it takes when loaded alone (core 0, MiB: shared scratchpad, model code, tensors incl. weights / IO)
N=32; [ "$1" = -n ] && { N=$2; shift 2; }
d=/sys/devices/virtual/neuron_device/neuron0/neuron_core0/stats/memory_usage/device_mem
for x in "$@"; do
  bash /root/op/lb.sh -n $N $x > /tmp/lbm.out 2>&1 & p=$!
  s=0; m=0; t=0
  while kill -0 $p 2>/dev/null; do a=$(cat $d/model_shared_scratchpad/total); b=$(cat $d/model_code/total); c=$(cat $d/tensors/total); [ $a -gt $s ] && s=$a; [ $b -gt $m ] && m=$b; [ $c -gt $t ] && t=$c; sleep 0.2; done
  echo "$(cat /tmp/lbm.out | cut -c1-230) | scratch $((s / 1048576)) code $((m / 1048576)) tensors $((t / 1048576)) MiB"
done
