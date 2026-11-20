#!/bin/bash
# mem.sh: HBM in use per NeuronCore by category (MiB), from the driver's sysfs counters (<category>/total = bytes in use now)
for c in 0 1; do d=/sys/devices/virtual/neuron_device/neuron0/neuron_core$c/stats/memory_usage/device_mem
  echo "core$c: total=$(( $(cat $d/total) / 1048576 )) peak=$(( $(cat $d/peak) / 1048576 )) $(for k in tensors model_shared_scratchpad nonshared_scratchpad model_code dma_rings runtime_memory constants collectives; do printf '%s=%d ' $k $(( $(cat $d/$k/total) / 1048576 )); done)"
done
